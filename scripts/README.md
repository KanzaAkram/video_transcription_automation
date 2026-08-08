# Scripts

What exists today, and what is deliberately deferred. Documented plainly so
the runbooks don't promise tools that aren't written yet.

## Available now

| Script | Purpose |
|---|---|
| `dry_run.py` | Full pipeline against a local fake Drive and fixture transcripts. **No credentials.** Start here. |
| `verify_config.py` | Validate `.env`, `settings.yaml`, taxonomies and local tooling. No network calls. |
| `bootstrap_drive_map.py` | Read-only. Resolve every syllabus node to a Drive folder and report how each match was made. Run before anything writes. |

Run order when connecting real accounts is in [docs/SETUP.md § 5](../docs/SETUP.md).

## Deferred to Phase 2

These are referenced in [docs/OPERATIONS.md](../docs/OPERATIONS.md) and are
**not yet implemented**. Each depends on the job store, which is designed
([docs/ARCHITECTURE.md § 12](../docs/ARCHITECTURE.md#12-data-model-and-state-machine))
but not wired up — writing them against a schema that may still change would
be wasted work.

| Script | Purpose | Depends on |
|---|---|---|
| `process_one.py` | Run one real recording end to end | Drive + Zoom credentials |
| `verify_zoom.py` | List recent Zoom recordings, read-only | Zoom credentials |
| `google_oauth_setup.py` | Browser consent flow, prints a refresh token | `oauth_user` mode only |
| `ingest_syllabus.py` | Extract a first-pass taxonomy from a syllabus PDF | — |
| `status.py` | Daily health summary | job store |
| `explain.py` | Why a given recording was classified as it was | job store |
| `calibrate.py` | Confidence threshold sweep against corrections | job store + corrections |
| `reprocess.py` | Reclassify without re-transcribing | job store |
| `backfill.py` | Run the pipeline over the existing archive | job store |
| `audit.py` / `revert.py` | Inspect and undo Drive mutations | audit log |
| `renew_watches.py` | Force renewal of Drive watch channels | Drive credentials |
| `benchmark_asr.py` | Measure real transcription throughput on your hardware | GPU host |

`benchmark_asr.py` is worth running early once a GPU exists — the throughput
figures in the architecture document are estimates, and capacity planning
should use measurements from your own hardware.
