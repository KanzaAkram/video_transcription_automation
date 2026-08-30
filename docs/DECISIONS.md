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

> **Superseded by ADR-017.** Retrieval no longer narrows the field before the
> model sees it; it runs alongside and is reported. The reasoning below is kept
> because two of its three objections turned out to be right, and the third is
> what changed.

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

> **Superseded by ADR-017.** The default provider is now OpenAI. The reasoning
> below — that this is an evidence-weighing task where a strong model earns its
> cost, and that structured output is what makes the reply safe to consume — is
> unchanged and is why the classifier sits behind an interface at all.

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

---

## ADR-015 — A lesson layer beneath the syllabus, generated from an overlay

**Decision.** The taxonomy carries three levels: syllabus topic, syllabus subtopic, and **lesson**. Topics, subtopics and learning-outcome text are extracted verbatim from the official PDF by `scripts/ingest_syllabus.py` and marked `verified: true`. Lessons come from a hand-written overlay, are marked `verified: false`, and must cite the learning-outcome numbers they cover.

**Why.** A syllabus subtopic is not a lecture. Cambridge 9700 topic 12 has two subtopics, *Energy* and *Respiration*, and the corresponding Drive folder holds six recordings. Naming files after subtopics gives six files two names between them, one of which duplicates the folder name — no better than `class 4 (9 aug).mp4`. The granularity teaching actually happens at is the learning outcome, and lessons are the natural grouping of those.

Keeping the two sources separate is what makes the split safe. The syllabus half is never typed, so 134 outcomes arrive without transcription errors in the text that retrieval scores against. The invented half is never presented as Cambridge's, and because each lesson cites outcome numbers, the ingester can copy real syllabus prose onto it — a lesson is matched against the syllabus, not against a title this repo made up.

**Consequences.** Adding a subject is two files, not one, and the overlay has to be kept in step with the outcome numbering (the ingester fails loudly if a lesson cites an outcome that does not exist). The lesson layer is also the part most likely to need revision after seeing real lectures: the first draft grouped 12.1's outcomes 1–3 into one lesson, and the first two recordings turned out to be exactly the 1 / 2–3 split, so they were being handed the same name. That is a cheap edit to the overlay and a regeneration — no code change.

---

## ADR-016 — Verification warnings track outcome text, not node flags

**Decision.** `load_taxonomy` warns about nodes carrying unverified *learning-outcome text*, not about every node whose `verified` flag is false.

**Why.** The lesson layer (ADR-015) is unverified by design and permanent. Counting it would print "89 of 117 nodes unverified" on every single load — an expected, unfixable warning, which is the fastest way to teach an operator to ignore warnings. The actual risk being tracked is outcome text that was typed rather than extracted, because that is the text retrieval scores against.

**Consequences.** The generated Biology taxonomy loads clean; the hand-written Chemistry one still warns, correctly, until its outcomes are regenerated from the 9701 PDF.

---

## ADR-017 — ChatGPT reads the whole lecture against the whole syllabus

**Decision.** `classify.provider` defaults to `openai`. One call per lecture carries the entire syllabus subtree and the entire transcript, and returns one chapter id and one to three subtopic ids. Retrieval still runs, but only to report; it no longer narrows what the model may choose from. Supersedes ADR-005 and ADR-010.

**Why.** Keyword coverage decides from vocabulary overlap, and vocabulary overlap cannot express two distinctions the task depends on.

*Taught versus mentioned.* A teacher who explains glycolysis for fifty minutes without saying "glycolysis" loses to a chapter whose keywords happen to be spoken. Counting terms answers "which topic's words appear most", which is a different question from "what was this lecture about" and only usually has the same answer.

*Content versus recap.* Every chapter-12 recording after the first opens with five to fifteen minutes revising the previous lesson. Those windows are real syllabus content and score exactly like it. On 11 August the recap won: the link reaction owns its own ten-minute window 51 terms to 9, but glycolysis leads across the file 69 to 85, and the lecture was proposed as *Glycolysis* — the name of the **previous** lesson. The confidence gate caught it at 0.46, so it routed to review rather than being misfiled. The gate working is not the same as the answer being right.

Neither failure is visible in the filename it produces, which is what makes them worth paying to remove.

**Why the whole syllabus and not a shortlist.** ADR-005's three objections, revisited:

| Objection | Now |
|---|---|
| Expensive per lecture | 15,000–19,000 tokens for a full syllabus, 2,700 scoped to one chapter. A few cents unscoped. |
| Slow | One call, no longer in a per-window loop. |
| Long-context discrimination across similar topics is where models blur | **Still true**, and answered directly: the tree is nested so sibling subtopics are adjacent, each node carries its verbatim learning outcomes, and the reply must cite timestamped quotes. |

Against that, narrowing has a cost that is not recoverable: a keyword shortlist can drop the right answer before anything has read the lecture, and no amount of downstream judgement gets it back. That is the same failure as above, moved one step earlier and made harder to see.

**Why the model returns ids and never names.** The reply schema declares the legal node ids as a JSON Schema `enum` and is sent with `strict: true`, so a conforming reply cannot contain an id the syllabus does not have; the filename is then rendered from the taxonomy's own `short_title`. There is no field a made-up name could travel in. Every id is re-checked against the taxonomy anyway, because the enum is dropped for a syllabus too large to enumerate and because a schema constrains the reply and not the world.

The alternative — a free-text `suggested_topic_label`, which the Claude path has — buys precision the taxonomy cannot express and costs a field with nothing behind it. It is also where the subtopic cap gets quietly lost: an abstaining reply still carries a label, because the schema marks every property required.

**Consequences.** The renamer needs `OPENAI_API_KEY` to do its main job; `--classifier keyword` still runs offline and is what the tests and the dry run use, so nothing about the build depends on a key. The keyword path stays as the baseline the model is measured against — and printing its top three next to every decision, as the `keywords` line, is how the two are compared on real folders rather than in principle.

---

## ADR-018 — Three subtopics in a filename, not two

**Decision.** `MAX_SUBTOPICS_IN_NAME` is 3. `naming.max_length` rises from 80 to 120, and a name over the cap drops a whole subtopic rather than cutting one in half.

**Why.** The two-subtopic cap was the right bound for a classifier that ranked by coverage. Ranking can order subtopics; it cannot say a third was taught, so a third name would have been a guess presented as a fact — and `select_subtopics` still stops at two for exactly that reason. A model that has read the lecture can make that judgement, and lectures that genuinely work through three subtopics of one chapter exist.

The length cap moves because its job changed. It was sized to keep a Cambridge syllabus sentence out of a filename; that is now structural — labels are rendered from each node's `short_title` and the classifier is never asked to write one — so the number is sized for what it now governs: three real labels joined. The longest such triple in the shipped syllabi is 155 characters, and dropping a whole subtopic to fit is what keeps "A + B" true where "A + B + Muscle Structure and the Neuro" would not be.

**Consequences.** A fourth subtopic still raises rather than truncating, at the `Classification` boundary, so a model that stops following the instruction is visible instead of silently trimmed. Three-part filenames are longer than the archive's existing convention; that is the cost of naming a lecture accurately, and the two-part and one-part names are unchanged.
