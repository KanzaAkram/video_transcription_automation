# Operations

Running it, watching it, and fixing it.

> **On the scripts referenced below.** `dry_run.py`, `verify_config.py` and
> `bootstrap_drive_map.py` exist today. The operational tooling — `status.py`,
> `explain.py`, `calibrate.py`, `revert.py`, `backfill.py` and the rest —
> is **Phase 2**: each depends on the job store, which is designed but not yet
> wired up, and writing them against a schema that may still change would be
> wasted effort. [scripts/README.md](../scripts/README.md) lists exactly what
> exists. The procedures below describe the intended operations either way,
> so they double as the specification for those scripts.

---

## Services

| Service | Command | Notes |
|---|---|---|
| API | `uvicorn pipeline.api.main:app --port 8080` | Must be publicly reachable over HTTPS for webhooks |
| Worker (I/O) | `celery -A pipeline.orchestration.tasks worker -Q io -c 8` | Scale horizontally |
| Worker (GPU) | `celery -A pipeline.orchestration.tasks worker -Q gpu -c 1` | Concurrency = GPU count |
| Scheduler | `celery -A pipeline.orchestration.tasks beat` | Exactly one instance |

The scheduler runs three periodic jobs:

| Job | Interval | Purpose |
|---|---|---|
| `renew_watch_channels` | 30 min | Drive watch channels expire. Renew before they do — this is the most common cause of "it stopped working". |
| `reconcile` | 10 min | Find inbox files with no job, and jobs stuck past timeout. |
| `retry_dead_letters` | 1 hour | Retry poisoned jobs; transient upstream outages self-heal. |

---

## Daily check

```bash
python scripts/status.py
```

```
Recordings (last 24h)      12
  DONE                     10
  NEEDS_REVIEW              2
  FAILED                    0
  in flight                 0

Watch channels          2 active, next expiry in 4h12m
Dead letter queue       0
Mean end-to-end        11m 04s
Auto-file rate         83%   (target: >90% once calibrated)
```

The number to watch over time is **auto-file rate**. Falling means either the taxonomy has drifted from what's being taught, or the threshold needs revisiting.

---

## Handling review items

Files in `_Needs Review` have a `?` in the name and a sidecar with the classifier's reasoning and evidence quotes.

**To correct one: move it into the right chapter folder in Drive.** That's the whole workflow. Within ten minutes the reconciler will:

1. Detect the move
2. Rename the file properly for its new location, dropping the `?`
3. Move the transcript and sidecar to follow it
4. Record the correction as a labelled training example

If it's genuinely unclassifiable — an exam debrief, an admin session — move it to `_Unclassified`. The pipeline leaves that folder alone.

---

## Failure modes

### Nothing has been processed for hours

Check in this order:

1. **Watch channels expired** — `scripts/status.py` shows expiry. Fix: `python scripts/renew_watches.py`. This is the most likely cause.
2. **Webhook endpoint unreachable** — check the Zoom app's Event Subscription for delivery failures; confirm the certificate is valid.
3. **Workers down** — `celery -A pipeline.orchestration.tasks inspect active`.
4. **Queue backed up** — check Redis depth. A stuck GPU job blocks the `gpu` queue.

The reconciler should catch 1 and 2 on its own within ten minutes. If it hasn't, the scheduler itself is down.

### Transcription failing

| Symptom | Cause | Fix |
|---|---|---|
| `ffmpeg not found` | Not on PATH | Install it; restart the worker |
| CUDA out of memory | Model too large for the card, or concurrent jobs | `gpu` concurrency to 1; drop to `medium`; or `compute_type: int8_float16` |
| Empty transcript | Silent or corrupt audio | Check the source; verify the M4A downloaded fully |
| Repeated identical sentences | `condition_on_previous_text` is on | Set it to `false` — this is exactly the loop it causes |
| Text over silence | VAD disabled | Enable `vad_filter` |

### Classification quality dropping

Check the correction log first — it tells you *how* it's wrong, which determines the fix. See [CLASSIFICATION.md § Failure modes](CLASSIFICATION.md#failure-modes-and-their-fixes).

Most common by far: teachers using vocabulary the taxonomy doesn't know. Fix by approving the aliases the correction loop proposes.

### Drive API errors

| Error | Meaning | Response |
|---|---|---|
| `403 rateLimitExceeded` / `429` | Too many requests | Automatic backoff. Persistent → check for a retry loop, then request more quota |
| `403 storageQuotaExceeded` | Service account tried to own a file | Auth mode is wrong — see [SETUP.md § 0](SETUP.md#0-decide-the-drive-authentication-mode-first) |
| `404` on a known folder | Folder deleted or moved out of scope | Re-run `bootstrap_drive_map.py` |
| `401` | Credentials expired or revoked | `oauth_user`: re-run consent. Others: check the key |

### A file was misfiled

1. Move it to the right folder in Drive. The pipeline follows it and records the correction.
2. To understand why: `python scripts/explain.py --recording-id <id>` prints the candidates, scores, evidence quotes, and reasoning.
3. If it's a pattern, fix the taxonomy rather than the individual file.

### Full reversal

Every Drive mutation is audited.

```bash
python scripts/audit.py --since 2026-08-01          # what changed
python scripts/revert.py --recording-id <id>        # undo one
python scripts/revert.py --since 2026-08-09T14:00   # undo a window
```

---

## Reprocessing

```bash
# Reclassify without re-transcribing — for after a taxonomy change.
python scripts/reprocess.py --stage classify --since 2026-08-01 --dry-run

# Re-transcribe — for after an ASR model upgrade. Expensive.
python scripts/reprocess.py --stage transcribe --recording-id <id>
```

Reclassification is cheap because transcripts are kept. After improving the taxonomy, reclassify in dry-run, look at what would change, then apply.

---

## Backfill

Running the pipeline over the existing archive.

```bash
python scripts/backfill.py --folder-id <id> --dry-run   # always first
python scripts/backfill.py --folder-id <id> --limit 10  # small batch
python scripts/backfill.py --folder-id <id>             # the rest
```

Backfill is also your best evaluation set: those files are already correctly filed, so agreement between the classifier and the existing placement is a direct accuracy measurement on real data. Run it in dry-run early even if you never intend to apply it.

---

## Tuning the confidence threshold

Start at `0.85`. Adjust on evidence, not feel.

```bash
python scripts/calibrate.py --since 2026-08-01
```

```
threshold  auto-filed  correct  wrong  → to review
0.95            42%      100%      0%        58%
0.90            67%       98%      2%        33%
0.85            83%       96%      4%        17%
0.80            91%       91%      9%         9%
```

The right threshold depends on the relative cost of a review versus a misfile. Misfiling is worse — it's silent — so bias toward review. Around 4% wrong at 0.85 is a reasonable operating point; 9% at 0.80 usually isn't.

Recalibrate after any taxonomy change, model change, or new subject.

---

## Adding a subject

1. Write `config/syllabi/<subject>.yaml`. Start from `scripts/ingest_syllabus.py --pdf <syllabus.pdf>` and edit.
2. Add the subject to `config/settings.yaml` with its Drive root folder ID.
3. `python scripts/bootstrap_drive_map.py --subject <id> --dry-run` — verify every node resolves to the right folder.
4. Run in shadow mode for a couple of weeks.
5. Calibrate the threshold for that subject specifically. Subjects differ: chemistry has dense distinctive vocabulary and classifies easily; a discursive subject will need a lower threshold and more review.

---

## Monitoring

Worth alerting on:

| Signal | Threshold |
|---|---|
| Watch channel expiring | < 1 hour remaining |
| No recording processed | > 24h during term |
| Dead letter queue | Any entry |
| Auto-file rate | Below 70% over a week |
| GPU queue depth | > 10 jobs |
| End-to-end latency | p95 > 1 hour |

Logs are structured JSON with a `recording_id` on every line, so tracing one lecture through all five services is a single filter.
