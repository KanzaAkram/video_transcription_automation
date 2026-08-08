# Architecture Decision Records

Each record states the decision, what it was chosen over, and what would make us revisit it. Recorded so that six months from now the reasoning is recoverable.

---

## ADR-001 — Zoom webhook as the primary trigger

**Decision.** Trigger on Zoom's `recording.completed` webhook. Keep a Google Drive change watcher as a secondary path and a scheduled reconciler as a safety net.

**Alternatives.**

| Option | Why not primary |
|---|---|
| Polling Drive on a schedule | Latency proportional to poll interval; wastes quota; sees a filename and nothing else |
| Drive push notifications alone | Doesn't say *what* changed (you must still diff the changes feed); channels expire and need renewal; loses all Zoom metadata |
| Watching a synced local folder | Ties the pipeline to one machine and to a sync client's reliability |

**Why.** The Zoom event fires earliest, carries the metadata that makes everything downstream better (host, meeting UUID, scheduled time, duration), and exposes the **audio-only M4A** — roughly a tenth the size of the MP4, which is a large recurring saving since transcription needs only audio.

**Consequences.** Requires a publicly reachable HTTPS endpoint and a Zoom Server-to-Server OAuth app. The reconciler makes the whole thing degrade gracefully: if the webhook is never wired up, the system still works from Drive alone, just later and with less metadata.

**Revisit if.** Zoom access is unavailable, or recordings arrive by a route that bypasses Zoom cloud recording entirely.

---

## ADR-002 — Three detection mechanisms, not one

**Decision.** Webhook + change watcher + reconciliation sweep, all three.

**Why.** Every push mechanism fails eventually and fails *silently*. Drive watch channels expire. Webhook deliveries get dropped. Endpoints have outages. A periodic sweep that asks "is there anything in the inbox with no job row?" is what turns "usually automatic" into "actually unattended".

The cost is one cron task and one Drive `list` call every ten minutes. The alternative is discovering three weeks later that nothing has been filed since a certificate renewal.

**Revisit if.** Never. This pattern is the point.

---

## ADR-003 — Move and rename, never re-upload

**Decision.** Once a file is in Drive, reposition it with `files.update` metadata calls. Download audio only, and only for transcription.

**Why.** A rename plus a reparent is a few hundred bytes of HTTP. Downloading and re-uploading a 2 GB lecture is minutes of transfer, a resumable-upload failure surface, and a new file ID that breaks every existing share link. Preserving the file ID also means existing links keep working after reorganisation.

**Consequences.** The pipeline needs Drive write scope on files it did not create — see ADR-009.

---

## ADR-004 — Transcription engine

**Decision.** `faster-whisper` with `large-v3`, behind an `ASRProvider` interface.

**Alternatives.**

| Option | Verdict |
|---|---|
| `openai-whisper` (reference) | Same accuracy, materially slower and heavier on VRAM. No reason to prefer it. |
| `whisper.cpp` | Excellent on CPU/Apple Silicon. The fallback if no GPU is available and hosted APIs are ruled out. |
| **WhisperX** | Adds VAD, forced alignment, and speaker diarization. Genuinely wanted eventually — separating teacher from student speech improves classification, because student questions are noise for topic detection. Deferred: heavier dependencies, and diarization needs a gated Hugging Face model with a token. **Planned Phase 3.** |
| NVIDIA Parakeet / Canary | Very fast, top-tier English accuracy. **English-only** — disqualifying if lectures code-switch to Urdu. |
| `distil-large-v3` | ~2× faster for a small accuracy cost. English-only. Good option if English is confirmed. |
| Hosted APIs (Groq, Deepgram, AssemblyAI) | No GPU needed, low per-minute cost, trivial to operate. Audio leaves your infrastructure. **The right choice below ~20 lectures/week.** |

**Why `large-v3`.** Multilingual, robust to accents, handles code-switching. In this teaching context an English-only model is a real risk: on Urdu spans it will not fail loudly, it will emit confident wrong English, which then corrupts classification.

**Consequences.** Needs a GPU for practical throughput. The provider interface means swapping is a config change.

**Revisit if.** Lectures are confirmed English-only (→ `distil-large-v3` or Parakeet), or volume stays low (→ hosted API, no GPU).

---

## ADR-005 — Hybrid retrieval before LLM classification

**Decision.** Narrow to ~8 candidate syllabus nodes with BM25 + embeddings, then have Claude choose among them with structured output.

**Alternatives.**

| Option | Why not |
|---|---|
| LLM sees the whole taxonomy | Expensive per lecture; long-context discrimination across dozens of similar chemistry topics is exactly where models blur |
| Pure keyword matching | Fails on paraphrase. A teacher can explain a concept thoroughly without using its syllabus name |
| Pure embeddings | Fails on the decisive rare terms. "Halogenoalkane" is a near-certain signal and embeddings dilute it |
| Fine-tuned classifier | Needs labelled data that doesn't exist yet. The correction log is being built precisely to make this possible later |

**Why hybrid.** The two retrieval methods fail differently — lexical misses paraphrase, semantic misses jargon — so fusing them is strictly better than either. The LLM then does what it's genuinely best at: weighing evidence and explaining itself.

**Consequences.** More moving parts than a single LLM call, but far cheaper per lecture and considerably more accurate.

---

## ADR-006 — Windowed transcripts and coverage profiles

**Decision.** Split the transcript into windows of a few minutes, classify each, and aggregate into a coverage profile.

**Why.** Real lectures aren't single-topic. A lecture might spend 20 minutes finishing the previous chapter and 50 minutes starting the next. Whole-transcript classification produces one answer and hides that. Windowing produces "78% relative atomic mass, 18% atomic structure", which supports a sound primary choice, a record of secondaries, and detection of continuation lectures.

**Consequences.** Slightly more compute during retrieval. Worth it — this is what makes multi-topic lectures work at all.

---

## ADR-007 — Never file below the confidence threshold

**Decision.** Below-threshold classifications go to `_Needs Review` with the best guess recorded. They are never filed silently.

**Why.** A misfiled lecture is worse than an unfiled one. Unfiled is visible in the inbox and gets fixed. Misfiled is invisible until somebody needs it, looks in the right folder, doesn't find it, and concludes the recording was lost. The failure mode of over-confidence is silent archive corruption.

**Consequences.** Human review effort, largest early on. That's the intended shape: [OPERATIONS.md](OPERATIONS.md) covers calibrating the threshold, and the correction loop is what drives review volume down.

---

## ADR-008 — Corrections are made by moving the file in Drive

**Decision.** Fixing a misclassification means dragging the file to the right folder. No admin UI.

**Why.** Teachers already know how to move files in Drive. Any custom interface is one more thing to log into and one more thing to teach, and it would be used less. The reconciler detects the move and turns it into a labelled training example for free.

**Consequences.** The reconciler must distinguish deliberate corrections from unrelated reorganisation. Handled by only treating moves of pipeline-managed files (tracked by file ID) as corrections.

---

## ADR-009 — Full Drive scope, tightly constrained

**Decision.** Request `https://www.googleapis.com/auth/drive`. Constrain everything else.

**Why.** The narrower `drive.file` scope grants access only to files the application created. Reorganising a pre-existing folder tree is exactly what that scope excludes, so it cannot work here.

**Mitigations**, all implemented in the storage layer:

- Every operation is confined to the configured root folder ID; a resolved path outside it is refused before any API call.
- No delete operation exists in the client. Move and rename only.
- Every mutation is written to an audit log with before/after state.
- A dry-run mode logs intended changes without making them.

**Revisit if.** Google introduces a narrower scope covering folder reorganisation.

---

## ADR-010 — Claude for classification

**Decision.** `claude-opus-5` with adaptive thinking and structured outputs.

**Why.** The task is evidence-weighing, not pattern-matching: deciding whether a lecture that mentions isotopes throughout is *about* relative atomic mass or *about* atomic structure requires judgement, and this is where a strong model earns its cost. Structured outputs mean the response is schema-validated at the API layer — the pipeline never parses free text and cannot receive a malformed reply. The cost is a few cents per lecture, negligible against the GPU time already spent transcribing it.

The classifier sits behind an interface, so a cheaper model can be used for easy cases and escalation reserved for ambiguous ones if cost ever matters.

---

## ADR-011 — SQLite now, Postgres later

**Decision.** SQLAlchemy against SQLite by default, Postgres by changing `DATABASE_URL`.

**Why.** At tens of lectures per week, SQLite is entirely sufficient and removes a service from the deployment. The ORM means the migration is a connection-string change plus running migrations. Postgres becomes necessary when workers run on more than one machine — concurrent writers are where SQLite's limits bite.

---

## ADR-012 — Separate GPU and I/O queues

**Decision.** Two queues. `gpu` has concurrency equal to GPU count; `io` runs at high concurrency.

**Why.** Transcription is GPU-bound and everything else is network-bound. A single queue means either the GPU idles while workers wait on Drive, or downloads are throttled to GPU concurrency. Splitting them keeps the expensive resource saturated and the cheap resource parallel.

---

## ADR-013 — Three independent numbering fields in the taxonomy

**Decision.** Store `teaching_order`, `coursebook_chapter`, and `syllabus_topic` as separate fields. Never derive one from another.

**Why.** Read directly off your Drive: `3. Chemical Bonding (Chapter 4)` and `4. Moles and Stoichiometry (Chapter 3)` — the ordinals and chapter numbers are transposed relative to each other, and neither matches the 9701 topic numbering. Any code that assumes they agree will file bonding lectures under stoichiometry.

**Consequences.** The taxonomy YAML is more verbose. That verbosity is the point: it makes the mismatch explicit rather than latent.

---

## ADR-014 — A fake Drive so the pipeline runs with no credentials

**Decision.** `DrivePort` is an interface with two implementations: `GoogleDriveAdapter` and `LocalDriveAdapter` (a local directory tree).

**Why.** Directly responsive to "I don't want to give credentials yet" — the entire pipeline can be built, demonstrated, and tested end-to-end before any account is connected. It also makes tests fast and deterministic, and gives a genuinely safe rehearsal environment: mirror the real folder structure locally and watch what the system does to it before letting it near the real thing.

**Consequences.** One interface to maintain. Worth it several times over.
