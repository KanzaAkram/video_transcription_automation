# Architecture

**Goal:** a teacher finishes a Zoom lecture and does nothing else. Within roughly an hour the recording is transcribed, matched to the right place in the syllabus, renamed to your existing convention, moved into the correct Google Drive folder, and has its transcript sitting beside it. Nobody runs a script.

This document describes how that happens, why each piece is built the way it is, and where the system deliberately stops and asks a human.

---

## Contents

1. [What I read off your Drive](#1-what-i-read-off-your-drive)
2. [System overview](#2-system-overview)
3. [Ingestion — detecting a new recording](#3-ingestion--detecting-a-new-recording)
4. [Media preparation](#4-media-preparation)
5. [Transcription](#5-transcription)
6. [Syllabus taxonomy](#6-syllabus-taxonomy)
7. [Classification](#7-classification)
8. [Naming](#8-naming)
9. [Placement in Drive](#9-placement-in-drive)
10. [Publishing the transcript](#10-publishing-the-transcript)
11. [Review, correction, and the feedback loop](#11-review-correction-and-the-feedback-loop)
12. [Data model and state machine](#12-data-model-and-state-machine)
13. [Reliability](#13-reliability)
14. [Scaling](#14-scaling)
15. [Security](#15-security)
16. [Cost](#16-cost)
17. [Deployment options](#17-deployment-options)
18. [Rollout plan](#18-rollout-plan)
19. [Open decisions I need from you](#open-decisions-i-need-from-you)

---

## 1. What I read off your Drive

Three facts from the screenshots drive real design decisions, so they're stated up front.

### 1.1 The filename convention already exists

```
Aug, 5  - Atomic Structure.mp4
Aug, 7  - AOD and some basic concepts.mp4
Aug, 10 - Relative Atomic Mass.mp4
```

`{Month abbreviation}, {day} - {topic}.mp4`. The renamer reproduces this exactly as its default. It is a template in config, not a hardcoded string, so you can change it later without touching code — but the default is *your* format, because a pipeline that renames everything into some new scheme would make the existing archive inconsistent.

### 1.2 Folder names carry two numbering systems, and neither is the syllabus

```
1.  Atomic Structure (Chapter 1)
2.  Electrons in Atoms (Chapter 2)
3.  Chemical Bonding (Chapter 4)      ← 3rd folder, chapter 4
4.  Moles and Stoichiometry (Chapter 3)  ← 4th folder, chapter 3
7.  Hydrocarbons (Chapter 15)
8.  Alcohols (Chapter 17)
9.  Halogenoalkanes (Chapter 16)      ← 9th folder, chapter 16
```

The leading ordinal is **teaching order**. The parenthesised number is the **Cambridge coursebook chapter**. Neither matches the **9701 syllabus topic number**. They diverge in both directions, repeatedly.

This is the single most common way a system like this quietly corrupts a folder tree: it infers that "Chapter 4" means "the fourth thing" and files bonding lectures under stoichiometry. So the taxonomy stores all three identifiers as independent fields and never derives one from another:

```yaml
- id: chem.bonding
  title: Chemical Bonding
  teaching_order: 3
  coursebook_chapter: 4
  syllabus_topic: "3"
  drive_folder_name: "3. Chemical Bonding (Chapter 4)"
```

Folder resolution matches on the recorded folder name, confirmed once and then cached by folder ID. Nothing is computed from an ordinal.

### 1.3 Recordings sit directly in chapter folders

There are no subtopic subfolders. `1. Atomic Structure (Chapter 1)` contains the PDF and the three MP4s directly.

So placement uses a **deepest-existing-folder** rule: resolve the classification as deep as the syllabus goes, then walk down the Drive tree only as far as folders that already exist, and drop the file there. Creating subtopic folders is off by default and opt-in per subject. The system fits your structure rather than imposing one.

---

## 2. System overview

Five services, each independently scalable. The split matters because the transcription step is a GPU bottleneck and everything else is I/O-bound waiting on network calls — mixing them wastes whichever resource is scarcer.

```
┌──────────────┐   recording.completed webhook   ┌─────────────────────┐
│    Zoom      │ ──────────────────────────────► │   API service       │
└──────────────┘                                 │  (FastAPI)          │
                                                 │  · verify signature │
┌──────────────┐   change notification           │  · dedupe           │
│ Google Drive │ ──────────────────────────────► │  · enqueue          │
└──────────────┘                                 └──────────┬──────────┘
       ▲                                                    │
       │                                          ┌─────────▼─────────┐
       │                                          │  Job store        │
       │                                          │  (Postgres/SQLite)│
       │                                          │  one row per      │
       │                                          │  recording        │
       │                                          └────┬─────────┬────┘
       │                                               │         │
       │                                    ┌──────────▼──┐   ┌──▼──────────┐
       │                                    │ io workers  │   │ gpu workers │
       │                                    │ (many)      │   │ (= #GPUs)   │
       │                                    │             │   │             │
       │  rename / move / upload            │ download    │   │ transcribe  │
       └────────────────────────────────────┤ classify    │   │             │
                                            │ place       │   └─────────────┘
                                            │ publish     │
                                            └──────┬──────┘
                                                   │
                                            ┌──────▼──────┐
                                            │  scheduler  │
                                            │ · renew     │
                                            │   watches   │
                                            │ · reconcile │
                                            │ · retry DLQ │
                                            └─────────────┘
```

| Service | Responsibility | Scales with |
|---|---|---|
| **API** | Receive webhooks, verify signatures, deduplicate, enqueue. Never does slow work. | Nothing — it's tiny. One instance is plenty. |
| **io workers** | Everything network-bound: download, Drive calls, LLM calls, uploads. | Lectures per hour. High concurrency, cheap. |
| **gpu workers** | Transcription only. | Number of GPUs. Expensive, so kept busy and nothing else runs here. |
| **scheduler** | Renews Drive watch channels before expiry, runs the reconciliation sweep, retries the dead-letter queue. | Nothing. One instance. |
| **job store** | The single source of truth for what state every recording is in. | Total recordings. |

The API service never transcribes, never calls Drive beyond a dedupe lookup, and never blocks. Webhook senders — Zoom especially — retry aggressively on slow responses, and a webhook handler that does real work turns one lecture into five duplicate jobs.

---

## 3. Ingestion — detecting a new recording

You asked which detection mechanism to use. **Zoom's `recording.completed` webhook is the primary trigger, with a Drive change watcher as a secondary path and a scheduled reconciler as the safety net.** Three mechanisms, because each covers a different failure.

### 3.1 Primary: the Zoom webhook

When a cloud recording finishes processing, Zoom POSTs to your endpoint. This is better than watching Drive for four reasons:

- **It fires earlier.** No waiting for a Zoom→Drive sync to run.
- **It carries the metadata you actually want** — host email, meeting topic, scheduled start time, duration, and a per-file `download_url` with a short-lived `download_token`. A Drive file gives you a filename and a timestamp.
- **It offers an audio-only file.** Zoom produces an `M4A` alongside the `MP4`. Transcription only needs audio, and the M4A is roughly a tenth the size. Over a term that is the difference between hundreds of gigabytes of transfer and tens.
- **It gives a stable identity.** The Zoom meeting UUID is the natural deduplication key. Drive filenames are not unique and can be edited.

Setup requires a **Server-to-Server OAuth app** (Zoom retired JWT apps in 2023) and an Event Subscription. Two implementation details that bite people:

- Zoom validates your endpoint with an `endpoint.url_validation` event: it sends a `plainToken` and you must reply with an HMAC-SHA256 of it keyed by your webhook secret token. Implemented in `src/pipeline/ingest/zoom_webhook.py`.
- Every subsequent request carries `x-zm-signature` and `x-zm-request-timestamp`. **Verify both** — signature and timestamp freshness — before doing anything. An unverified webhook endpoint is an open invitation to enqueue arbitrary work.

### 3.2 Secondary: the Drive change watcher

Recordings may also arrive in Drive by other routes — a teacher uploading a local recording, a backfill of old material, or your existing Zoom→Drive sync. So the designated inbox folder is also watched.

The important thing about Drive push notifications, which is not obvious from the API docs: **the notification does not tell you what changed.** It is a "something happened" ping. The correct pattern is:

1. Once, at setup: call `changes.getStartPageToken` and persist the token.
2. Register a watch channel on the changes feed (or the inbox folder).
3. On any notification: call `changes.list` from the saved token, walk the results, act on the ones that are new video files in the inbox, and persist the new token.

Watch channels also **expire** — commonly within a day, sometimes much sooner — so the scheduler renews them well before expiry. This is the number one reason "it worked for a week then stopped" happens with Drive-watching systems.

### 3.3 Safety net: the reconciler

A cron sweep, every 10 minutes, that asks two questions:

- Are there video files in the inbox with no matching job row? (A webhook was missed or the channel silently died.)
- Are there jobs stuck in a non-terminal state past their timeout? (A worker died mid-flight.)

Anything it finds gets enqueued or reset for retry. This is what makes the system genuinely unattended: every push mechanism fails eventually, and polling is the thing that notices.

Note this makes the pipeline resilient to your Zoom→Drive sync entirely — if the webhook path is never connected, the reconciler alone is sufficient, just slower and without the rich metadata.

### 3.4 On "xScript"

Your developer mentioned xScript. I could not identify a tool by that name in this space with enough confidence to build on it, and I'd rather say so than guess. If it refers to a specific product you already license, tell me what it does and I'll assess whether it replaces the ingestion layer — the `IngestSource` interface exists precisely so a different trigger can be dropped in without touching anything downstream. If it was a mishearing of a generic "a script", then the answer to the underlying question is: **a bare scheduled script is the weakest of the three options** and is included here only as the safety net, not the primary mechanism.

---

## 4. Media preparation

The worker downloads **audio only** and never touches the video where it can avoid it.

```
Zoom M4A ──► ffmpeg ──► 16 kHz mono PCM WAV ──► ASR
```

```bash
ffmpeg -i input.m4a -vn -ac 1 -ar 16000 -c:a pcm_s16le output.wav
```

16 kHz mono is what Whisper-family models consume internally; feeding anything else just makes the model resample it. Extracting once up front saves that work on every retry.

**The video file is never downloaded and never re-uploaded.** This is worth stating plainly because it is the difference between a pipeline that costs hours and one that costs seconds. Once a file is in Drive, renaming it and moving it are *metadata operations* — `files.update` with a new name, and `addParents`/`removeParents` for the move. A 2 GB lecture is repositioned in the tree with a few hundred bytes of HTTP traffic.

The only case where the video is downloaded is when the recording is not yet in Drive at all and the pipeline is doing the Zoom→Drive transfer itself. That path uses resumable uploads with checkpointing so a dropped connection resumes rather than restarts.

---

## 5. Transcription

**Recommendation: `faster-whisper` running `large-v3`**, behind an `ASRProvider` interface so the engine can be swapped without touching the pipeline.

### Why this one

`faster-whisper` is a CTranslate2 reimplementation of OpenAI's Whisper. Same weights, same accuracy, substantially faster and lighter on VRAM than the reference implementation. For lecture-length audio that matters: it turns a per-lecture cost of tens of minutes into a few minutes.

`large-v3` rather than a distilled or English-only model because of a specific risk: **if your teachers code-switch between English and Urdu**, which is common in this teaching context, English-only models (NVIDIA Parakeet, `distil-large-v3`) will produce confident nonsense on the non-English spans. Multilingual `large-v3` handles code-switching far more gracefully. If you confirm the lectures are English-only, `distil-large-v3` is roughly twice as fast for a small accuracy cost, and it's a one-line config change.

Alternatives considered and why they lost are in [DECISIONS.md](DECISIONS.md#adr-004--transcription-engine). The short version: WhisperX adds speaker diarization, which is genuinely useful for separating teacher from student questions but adds a heavy dependency and a Hugging Face gated-model login; it's the planned Phase 3 upgrade, not the Phase 1 default. Hosted APIs (Groq, Deepgram, AssemblyAI) need no GPU at all and are the right choice at low volume — the provider interface supports them.

### Three settings that matter more than the model choice

**1. Prime the decoder with syllabus vocabulary.** Chemistry terms are exactly what generic ASR fumbles — "stoichiometry", "halogenoalkane", "electronegativity", "Avogadro". Both `initial_prompt` and `hotwords` are populated from the vocabulary of the syllabus subject the lecture is expected to belong to, so the model has already seen the right spellings before it hears them. This is cheap and it measurably improves the terms that the classifier then depends on.

**2. `condition_on_previous_text=False`.** Whisper's default is to feed each window its own prior output as context. On hour-long audio with silences this causes repetition loops — the model gets stuck emitting the same sentence for minutes. For long-form lecture audio, turn it off.

**3. Voice-activity detection.** Lectures start with several minutes of "can everyone hear me". VAD filtering drops the silence before it reaches the model, which cuts runtime and prevents Whisper's well-known tendency to hallucinate text over silence.

### Outputs

One transcription produces four artifacts, all persisted:

| Artifact | Purpose |
|---|---|
| Segment list with timestamps | Canonical form, stored in the job store |
| `.md` | Human-readable, uploaded to Drive with timestamped headings |
| `.srt` / `.vtt` | Subtitles, playable directly against the video in Drive |
| Plain text | Fed to the classifier |

---

## 6. Syllabus taxonomy

The syllabus is a tree, loaded from YAML, one file per subject:

```
Subject  →  Chapter  →  Subtopic  →  Sub-subtopic
```

Each node carries what the rest of the system needs:

```yaml
- id: chem.atomic_structure
  title: Atomic Structure
  teaching_order: 1
  coursebook_chapter: 1
  syllabus_topic: "1"
  drive_folder_name: "1. Atomic Structure (Chapter 1)"
  aliases: ["structure of the atom", "AOD", "atomic orbital diagram"]
  keywords: [proton, neutron, electron, isotope, nucleon, mass number,
             relative atomic mass, mass spectrometer]
  learning_outcomes:
    - code: "1.1"
      text: "identify and describe protons, neutrons and electrons..."
  children:
    - id: chem.atomic_structure.rel_masses
      title: Relative Atomic Mass
      keywords: [relative isotopic mass, carbon-12, weighted average]
```

Four fields do the heavy lifting:

- **`keywords`** feed lexical retrieval. Domain jargon is high-signal — "halogenoalkane" appears in exactly one chapter.
- **`learning_outcomes`** are the actual syllabus text. They get embedded for semantic retrieval and are quoted to the LLM, so the classifier reasons against the real syllabus rather than a title.
- **`aliases`** capture what teachers *say*, which is not what the syllabus *writes*. Your own file `Aug, 7 - AOD and some basic concepts.mp4` is the proof: "AOD" is teacher shorthand that appears in no syllabus document anywhere.
- **`drive_folder_name`** is the observed folder name, resolved once to a folder ID and cached.

`config/syllabi/as-chemistry-9701.yaml` ships as a working template built from the coursebook chapter structure your folders follow. **It needs verifying against your actual syllabus PDF before production use** — you have `Data Section 9701-2025-2027-syllabus.pdf` in Drive, and `scripts/ingest_syllabus.py` extracts a first-pass taxonomy from it for you to review. I have not invented learning-outcome codes I could not verify; unverified entries are marked `verified: false` and the loader warns about them.

---

## 7. Classification

The most important requirement you gave: **classify on what was actually taught, not on the recording title.** A Zoom title of "Chemistry Class" carries no information. The transcript carries all of it.

Two stages on the default path — decide, then gate — with retrieval running
alongside as reporting rather than as a filter.

### 7.1 Retrieve candidates

This was originally the first stage and is now a second opinion. The reasoning
for it was that comparing a 12,000-word transcript against every syllabus node
with an LLM would be slow, expensive, and worse, because long-context
comparison across dozens of similar chemistry topics is where models blur
distinctions. Two of the three turned out not to hold: a full syllabus is
15,000–19,000 tokens and a lecture is ten to twenty thousand more, which is
neither slow nor expensive at current prices. The third is a real risk, and the
prompt answers it by nesting the tree, quoting each node's learning outcomes,
and requiring timestamped evidence for the choice.

What did hold is the cost of narrowing: a keyword shortlist can drop the right
answer before anything has read the lecture, and then nothing downstream can
recover it. So retrieval no longer gates the decision. It still runs, and its
coverage profile is reported next to the model's answer so the two can be seen
to disagree.

The transcript is split into windows of a few minutes each. Each window is scored against every syllabus node by two independent methods:

- **Lexical (BM25)** over keywords, aliases, and outcome text. Catches the decisive rare terms.
- **Semantic (embeddings)** over outcome text. Catches paraphrase — a teacher explaining ionisation energy without ever saying "ionisation energy".

Scores are fused and aggregated across windows into a ranked candidate list, typically the top 8 nodes. Aggregating across windows rather than over the whole transcript is deliberate: it produces a **coverage profile** showing how much of the lecture each candidate accounts for, which is what makes multi-topic lectures tractable.

### 7.2 Decide

One call to ChatGPT (`classify.openai_model`, default `gpt-5`, structured
output). It receives:

- **The whole syllabus subtree**, nested chapter → subtopic → lesson, with each
  node's id, title, filename label, aliases, keywords and learning outcomes
- **The whole transcript**, one timestamped line per 45 seconds
- The date, teacher, original filename and any Zoom meeting topic — as **weak**
  signals, explicitly labelled as such and explicitly not to be classified from

It returns a validated JSON object, not prose:

```json
{
  "chapter_id": "chem.atomic_structure",
  "subtopics": [{"node_id": "chem.2_1", "coverage": 0.78}],
  "confidence": 0.91,
  "evidence": [
    {"quote": "so the relative atomic mass is the weighted mean of the isotopes",
     "start_s": 412.5}
  ],
  "reasoning": "Two thirds of the lecture derives weighted averages from mass spectra...",
  "abstain": false,
  "abstain_reason": ""
}
```

Structured output means the response is schema-validated at the API layer, so the pipeline never parses free text and never has to handle a malformed reply.

**The reply names ids and never names topics.** `chapter_id` and each `node_id`
are declared in the schema as an `enum` of the actual syllabus ids, so a
conforming reply cannot contain something the syllabus does not have; the
filename is then rendered from the taxonomy's own `short_title` for those ids.
There is no field a made-up name could travel in. Every id is re-checked
against the taxonomy afterwards regardless, and one belonging to a different
chapter than the one just chosen is dropped.

**Evidence quotes with timestamps are required.** They are what makes a wrong classification debuggable — you can jump to 6:52 in the recording and see immediately whether the model was right. A classifier that says "Atomic Structure, trust me" is unauditable, and an unauditable classifier eventually silently degrades a folder tree.

### 7.3 Gate

Confidence never routes silently:

| Condition | Action |
|---|---|
| High confidence, single dominant topic | File it |
| High confidence, several topics with real coverage | File under the dominant one, record the others in metadata, optionally add Drive shortcuts |
| Below threshold, or `abstain` | Move to `_Needs Review`, name it with the best guess and a `?` marker, notify |
| Nothing matched at all | Leave in the inbox, flag it |

**The system never guesses quietly.** A misfiled lecture is worse than an unfiled one: unfiled is visible, misfiled is invisible until somebody goes looking for it in the wrong folder.

Full detail, including the prompt design and how the confidence threshold should be calibrated against your own data, is in [CLASSIFICATION.md](CLASSIFICATION.md).

---

## 8. Naming

Template-driven, defaulting to your existing convention:

```
{month_abbr}, {day} - {topic}{part_suffix}{ext}
→  "Aug, 5 - Atomic Structure.mp4"
```

Available fields: `date`, `year`, `month_abbr`, `day`, `subject`, `chapter`, `subtopic`, `topic`, `teacher`, `part`, `confidence_marker`.

Rules that keep the archive clean:

- **Illegal characters removed.** `/ \ : * ? " < > |` break in Drive, in Windows sync clients, and in shell scripts.
- **Length capped** to 120 characters, truncating the topic and never the date.
- **Collisions resolved by meaning, not by counter.** Same date and same topic almost always means a lecture split across two recordings, so it becomes `- Part 1` / `- Part 2` rather than `(1)` / `(2)`. Genuinely distinct same-day topics keep their own names.
- **Low-confidence files are marked.** A `?` in the filename makes anything that slipped past review visible at a glance in the folder listing.
- **The original name is preserved** in the job store and in the sidecar metadata, so any rename is reversible.

---

## 9. Placement in Drive

```
classification → syllabus node → Drive folder ID → files.update (rename)
                                                 → files.update (addParents / removeParents)
```

Both operations are metadata-only. No transfer of file content.

**Folder resolution** happens once per node and is then cached in the `folder_map` table:

1. Look up the node's `drive_folder_name` among children of the parent folder — exact match.
2. If that fails, normalised match: strip the leading ordinal and the `(Chapter N)` suffix, lowercase, collapse whitespace.
3. If that fails, fuzzy match above a similarity threshold — and **log it for confirmation** rather than trusting it.
4. If that fails, either create the folder (only if `allow_create` is on for that subject) or route to review.

After the first resolution everything is by folder ID, which is stable across renames. If somebody renames a folder in Drive, the cached ID keeps working; a periodic verification pass re-checks names and flags drift.

**Guardrails, because this component has write access to your file tree:**

- Every mutation is scoped under the configured root folder ID. A resolved path outside it is refused.
- The pipeline **never deletes** anything. Move and rename only.
- Every mutation is written to an audit log with the before and after state, so any action can be reversed.
- A `--dry-run` mode logs exactly what would change without changing it. Use it for the first week.

---

## 10. Publishing the transcript

Alongside the recording, matching its name:

```
1. Atomic Structure (Chapter 1)/
├── Aug, 10 - Relative Atomic Mass.mp4
├── Aug, 10 - Relative Atomic Mass.md      ← readable transcript
├── Aug, 10 - Relative Atomic Mass.srt     ← subtitles
└── .lecture-meta/
    └── Aug, 10 - Relative Atomic Mass.json
```

Configurable to a `Transcripts/` subfolder if you'd rather keep chapter folders visually clean.

The `.json` sidecar carries the full provenance: source Zoom UUID, original filename, ASR model and version, classification result with all evidence quotes, confidence, and every alternative the classifier considered. This is what makes the system auditable a year later, and what allows a reclassification pass when the taxonomy improves — without re-transcribing anything.

The `.md` transcript uses timestamped headings, which makes the file useful to students as a study aid rather than just a machine artifact.

---

## 11. Review, correction, and the feedback loop

Low-confidence results land in `_Needs Review` with the classifier's best guess, its confidence, and its evidence quotes in the sidecar.

**Correcting one is just moving the file in Drive.** A teacher drags it into the right chapter folder. No console, no login, no training. The reconciler notices the move, records it as a correction, and does three things:

1. Renames the file properly for its new home (dropping the `?` marker).
2. Moves the transcript and sidecar to follow it.
3. Writes a correction record: transcript, what the classifier said, what the human decided.

Corrections then improve the system in three ways:

- **Immediately**: distinctive phrases from the corrected transcript are proposed as new `aliases` or `keywords` on the correct node, for your approval. This is how "AOD" gets learned.
- **Per-run**: recent corrections are included as few-shot examples in the classifier prompt.
- **Over time**: the correction set is a labelled evaluation dataset. It's what tells you whether a threshold change or a model upgrade actually helped, rather than just feeling better.

The design intent is that the system gets more accurate the more it's used, without anybody doing anything they wouldn't do anyway.

---

## 12. Data model and state machine

One row per recording, one explicit state:

```
DISCOVERED → FETCHED → AUDIO_READY → TRANSCRIBED → CLASSIFIED
                                                        │
                                          ┌─────────────┴─────────────┐
                                          ▼                           ▼
                                       NAMED                    NEEDS_REVIEW
                                          ▼                           │
                                       PLACED                    (human moves file)
                                          ▼                           │
                                      PUBLISHED ◄────────────────────┘
                                          ▼
                                        DONE
```

Plus `FAILED` (terminal after retry exhaustion, goes to the dead-letter queue) reachable from any state.

Every transition is atomic and every step is **idempotent**: re-running a step that already completed is a no-op that returns the same result. That property is what makes crash recovery trivial — a worker that dies mid-transcription simply has its job picked back up at `AUDIO_READY`, and nothing needs to be undone.

Key tables:

| Table | Purpose |
|---|---|
| `recordings` | The job row: identity, state, attempts, timings, error |
| `transcripts` | Segments, language, model version, ASR runtime |
| `classifications` | Every classification attempt with full evidence — history, not just the latest |
| `folder_map` | Syllabus node → Drive folder ID, with resolution method and confidence |
| `corrections` | Human overrides, forming the evaluation set |
| `drive_audit` | Every mutation made to Drive, for reversal |
| `watch_channels` | Active Drive channels and their expiry, for renewal |

Deduplication is on `(source, source_id)` plus a content hash, so a replayed webhook, a reconciler double-detection, and a re-uploaded file all resolve to the same job.

---

## 13. Reliability

**Retries.** Every external call retries with exponential backoff and jitter. Google Drive returns `403 rateLimitExceeded` and `429` under load; backoff is mandatory, not optional, and a tight retry loop against Drive will get the project throttled hard.

**Poison messages.** Retries are capped per step. Exhausted jobs go to a dead-letter queue with full context rather than looping forever. The scheduler retries the DLQ on a slow cadence, so a transient upstream outage recovers by itself.

**Partial failure.** Each step commits its output before the next begins. A classifier failure never loses a transcript that took eight minutes to produce.

**Crash recovery.** State lives in the database, not in worker memory. Kill every worker mid-run and restart: each job resumes from its last committed state.

**Ordering.** No global ordering is required — lectures are independent. Same-lecture operations are serialized by a per-recording lock.

Failure modes and their responses are enumerated in [OPERATIONS.md](OPERATIONS.md#failure-modes).

---

## 14. Scaling

### Where the time goes

For a 90-minute lecture:

| Step | Time | Bound by |
|---|---|---|
| Download M4A (~80 MB) | 1–3 min | Network |
| Audio extraction | ~20 s | CPU |
| **Transcription** | **4–15 min** | **GPU** |
| Retrieval + classification | 20–60 s | LLM API |
| Rename + move | < 5 s | Drive API |
| Transcript upload | 5–15 s | Drive API |

Transcription dominates, so it gets its own queue sized to GPU count while everything else runs at high concurrency on cheap workers.

The GPU figures are estimates and depend heavily on hardware — a recent consumer GPU with batched inference sits at the fast end, an older datacentre card at the slow end, and CPU-only is roughly real-time or worse. **Measure on your actual hardware before sizing**; `scripts/benchmark_asr.py` exists for exactly this.

### Growing

- **One subject, a few lectures a week**: SQLite, one machine, or hosted ASR with no GPU at all.
- **Several subjects, tens per week**: the same single box, with transcription moved to a GPU or to a hosted API. Still SQLite, still no queue.
- **Institution-wide, hundreds per week**: only here does splitting transcription onto its own machine start to pay, and only then does a shared database become necessary. Batch overnight — lectures are not latency-sensitive, and an eight-hour turnaround is fine.

Nothing in the design assumes one subject or one syllabus. Subjects are configuration: adding A2 Chemistry, or Physics 9702, or a different exam board, means adding a YAML file and pointing it at a Drive root. Taxonomies are versioned so a syllabus change doesn't invalidate old classifications.

**Quotas to respect**: Google Drive's default per-project quota is on the order of 12,000 queries per minute — generous for this workload, but check the current figure in the Cloud console since it changes. The real risk is a runaway retry loop, not steady-state traffic. Every Drive call goes through one rate-limited client wrapper.

---

## 15. Security

**Credentials** live in environment variables in development and a secret manager in production. Never in the repository — `.gitignore` blocks the obvious filenames, and `.env.example` documents every variable with no real values.

**Google Drive scope.** This needs saying clearly: reorganising folders the application did not create requires the full `https://www.googleapis.com/auth/drive` scope. The narrower `drive.file` scope only grants access to files the app itself created, which cannot work here. That is a broad grant, so it is mitigated by:

- Confining every operation to the configured root folder ID
- Never deleting
- Auditing every mutation with before/after state
- A dry-run mode

**Webhooks** verify signatures before any processing. Zoom's HMAC-SHA256 signature and timestamp are both checked; Drive notifications are validated against the registered channel token.

**Transcripts are student data.** They contain classroom audio, including student voices and names. Treat them accordingly: encrypt at rest, set a retention policy, and restrict Drive permissions on transcript files if the recordings are shared more widely than the transcripts should be. Worth a deliberate decision rather than a default.

**Least privilege for Zoom**: the Server-to-Server app needs only `recording:read` and `user:read`.

---

## 16. Cost

Per 90-minute lecture, at Phase 2 scale:

| Component | Cost |
|---|---|
| Transcription, self-hosted GPU | Electricity plus amortised hardware — effectively free at low volume |
| Transcription, hosted API | Roughly $0.01–0.40 depending on provider |
| Classification (Claude) | A few cents — the transcript is the input, and only a digest of it is sent |
| Drive API | Free within quota |
| Storage | Transcripts are kilobytes next to gigabyte videos — negligible |

The dominant cost is GPU hardware or GPU rental, and it is only worth it above roughly 20 lectures a week. Below that, hosted ASR is cheaper and much simpler to operate. The provider interface means this is a config change, not a rewrite, so you can start hosted and move in-house when volume justifies it.

---

## 17. Deployment options

| Option | Good for | Trade-off |
|---|---|---|
| **A. Single machine + tunnel** | Getting started, a GPU you already own | Home internet must stay up; webhook endpoint needs a tunnel |
| **B. Small VPS + on-prem GPU worker** | Recommended for Phase 2 | Two things to operate, but the reliable public endpoint is separated from the beefy private one |
| **C. Fully cloud, serverless GPU** | No hardware, spiky volume | Per-second GPU billing, cold starts, more moving parts |
| **D. Fully managed, hosted ASR** | Lowest operational burden | Per-minute transcription cost; audio leaves your infrastructure |

**Option B is the recommendation** at your likely scale: a small always-on VPS runs the API and scheduler (a webhook endpoint must be reachable and stable), while the GPU worker sits wherever the GPU is and reaches out to the queue. The GPU box needs no inbound ports.

There is deliberately no compose file, no broker and no second database. One
machine runs the whole pipeline from a cron job against a SQLite file. That is
the right shape for tens of lectures a week, and the shape to keep until
something measured says otherwise.

---

## 18. Rollout plan

**Phase 1 — Shadow mode.** Everything runs, nothing is moved. The pipeline transcribes and classifies, then writes a report of what it *would* have done. Run it over lectures already correctly filed and measure agreement with the human filing. This is the honest way to establish the confidence threshold, and it costs nothing to get wrong.

**Phase 2 — Assisted.** Only high-confidence results are filed automatically; everything else goes to review. Corrections accumulate into the evaluation set. Expect to be reviewing a meaningful fraction at first — that number falling is the signal that thresholds can loosen.

**Phase 3 — Autonomous.** Thresholds tuned on real data, review handling the tail. Add subjects. Optionally add diarization to separate teacher from student speech, and a search index across all transcripts — which is a genuinely valuable by-product: once every lecture is transcribed, "which lecture covered Le Chatelier" becomes answerable.

**Backfill** is a separate one-off job that runs the same pipeline over existing recordings, in dry-run first. Your existing archive is also the best possible test set, because its correct answers are already known.

---

## Open decisions I need from you

Four questions. I have built so that all answers are supported, but each changes the recommended default, and the first one changes the authentication design materially.

### 1. Is the Google Drive account Workspace or a personal Gmail account?

This is the big one. **Service accounts cannot own files in a personal My Drive** — they have no storage quota, and uploads fail with a quota error that is confusing because the *user's* drive has plenty of space. Three resolutions:

| Your situation | Approach |
|---|---|
| Workspace + content on a **Shared Drive** | Service account as a member of the Shared Drive. Cleanest option by a wide margin. |
| Workspace + content in a user's My Drive | Service account with **domain-wide delegation**, impersonating that user. Needs Workspace admin access. |
| **Personal Gmail** | Service accounts don't work. Use OAuth user credentials with a stored refresh token. Workable, but re-consent is needed if the token is revoked. |

The screenshots show the Google Drive mobile app with a "Shared" tab, which is consistent with either. If it's a personal account, moving this content to a Shared Drive is worth considering before we wire anything up — it removes a whole category of ongoing problems.

`GOOGLE_AUTH_MODE` already supports all three.

### 2. What language are the lectures delivered in?

English only, or English mixed with Urdu? This decides the ASR model, and choosing wrong is not subtle — an English-only model on code-switched audio produces fluent, confident, wrong text. Default is multilingual `large-v3` because it's the safe choice; confirm English-only and it gets meaningfully faster.

### 3. Where should this run, and is there a GPU?

Determines whether transcription is self-hosted or hosted API. If there's no GPU and you'd rather not rent one, hosted ASR is genuinely fine and simpler — the interface supports both.

### 4. How many lectures per week, across how many subjects?

Sets whether this is SQLite-on-one-box or Postgres-with-workers. Under about 20 a week, keep it simple.

---

Answer these and I'll finalise the defaults, set the concrete instance sizes, and finish the adapters that are currently stubbed. Everything in the current scaffold runs without them.
