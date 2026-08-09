# Lecture Pipeline

Automated pipeline: **Zoom cloud recording → transcript → syllabus classification → renamed file in the correct Google Drive folder.**

Built for Cambridge International AS/A Level teaching, starting with **AS Chemistry (9701)**. Designed to run unattended once credentials are connected, and to scale to many subjects, teachers, and syllabuses.

---

## Status

**Phase 0 — architecture and scaffold. No credentials connected yet.**

Everything in this repo runs today against a **local filesystem stand-in for Google Drive** and a **fixture transcript**, so the whole pipeline can be demonstrated end-to-end before you hand over any Zoom or Google credentials. See [Dry run](#dry-run) below.

| Layer | State |
|---|---|
| Architecture, data model, state machine | Designed — see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) |
| Domain models, config, ports/interfaces | Implemented |
| Local (fake) Drive adapter | Implemented — pipeline runs offline |
| Naming, placement, syllabus taxonomy | Implemented |
| Google Drive adapter | Interface implemented, calls stubbed pending credentials |
| Zoom client + webhook receiver | Interface implemented, calls stubbed pending credentials |
| ASR (faster-whisper) | Provider implemented, needs GPU host + `ffmpeg` |
| LLM classifier | Implemented against the Claude API, needs `ANTHROPIC_API_KEY` |

## Read these first

| Document | What it answers |
|---|---|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | How the whole thing works, component by component. **Main deliverable.** |
| [docs/DECISIONS.md](docs/DECISIONS.md) | Why each technology was chosen, and what was rejected. |
| [docs/CLASSIFICATION.md](docs/CLASSIFICATION.md) | How a lecture gets matched to a syllabus topic, and how wrong matches are caught. |
| [docs/SETUP.md](docs/SETUP.md) | Exactly which credentials to create when you're ready, and what each is used for. |
| [docs/OPERATIONS.md](docs/OPERATIONS.md) | Running it, failure modes, recovery, what to do when a lecture is misfiled. |

## The flow

```
Zoom "recording.completed" webhook
        │
        ▼
  [ ingest ]  record the recording, deduplicate, enqueue
        │
        ▼
  [ media ]   download audio-only (M4A) — not the video
        │
        ▼
  [ asr ]     faster-whisper large-v3, primed with syllabus vocabulary
        │
        ▼
  [ classify ] retrieve candidate syllabus nodes → LLM picks + cites evidence
        │
        ├── low confidence ──► _Needs Review folder + notification (never silently misfiled)
        ▼
  [ name ]    "Aug, 5 - Atomic Structure.mp4"   (matches your existing convention)
        │
        ▼
  [ place ]   move + rename in Drive (metadata-only; the video is never re-uploaded)
        │
        ▼
  [ publish ] transcript .md/.srt/.vtt alongside the recording
```

## Two things this design takes from your current Drive

Both were read off the screenshots you provided, and both are load-bearing:

1. **Your filename convention is `Aug, 5 - Atomic Structure.mp4`** — month abbreviation, comma, day, space-hyphen-space, topic. The renamer reproduces this exactly by default; it is a configurable template, not hardcoded.

2. **Your folder names carry two different numbering systems at once.** `3. Chemical Bonding (Chapter 4)` is the *third* folder in teaching order but *chapter 4* of the Cambridge coursebook — and neither number matches the 9701 syllabus topic number. The taxonomy therefore stores all three independently (`teaching_order`, `coursebook_chapter`, `syllabus_topic`) and never assumes they agree. See [docs/ARCHITECTURE.md § Syllabus taxonomy](docs/ARCHITECTURE.md#5-syllabus-taxonomy).

Recordings also live **directly inside chapter folders** in your Drive, not in subtopic subfolders. The placement engine descends only as deep as folders that actually exist and does not invent new ones unless you turn that on.

## Dry run

Runs the full pipeline with no credentials: a fake Drive backed by a local directory, fixture transcripts, and a keyword-only classifier.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows;  source .venv/bin/activate on macOS/Linux
pip install -e ".[dev]"

python scripts/dry_run.py
```

It builds a miniature copy of your `AS Chemistry` tree under `.dryrun/`, drops two unclassified recordings into the inbox, and processes both — deliberately chosen to show each outcome:

| Lecture | Outcome |
|---|---|
| Halogenoalkanes — one topic throughout | **Filed.** Renamed `Sep, 14 - Halogenoalkanes.mp4`, moved into `9. Halogenoalkanes (Chapter 16)`, transcript and subtitles uploaded beside it. |
| Relative atomic mass — opens with a recap of the previous chapter | **Sent to review.** Correct topic identified, but not confidently enough to file. |

Both recordings carry the Zoom meeting topic `"Chemistry Class"`, which tells you nothing — the classification comes entirely from the transcript.

The second outcome is the safety mechanism, not a failure: a misfiled lecture is worse than an unfiled one because it's invisible. Add `--llm` to use the real Claude classifier (needs `ANTHROPIC_API_KEY`); `--keep` leaves `.dryrun/` in place so you can inspect the result.

## Other scripts

| Script | Purpose |
|---|---|
| `scripts/verify_config.py` | Validate `.env`, settings, taxonomies, local tooling. No network calls. |
| `scripts/bootstrap_drive_map.py` | Read-only. Resolve every syllabus node to a Drive folder and report how each match was made. |

[scripts/README.md](scripts/README.md) lists what exists today versus what is deferred to Phase 2.

## Layout

```
config/            settings, syllabus taxonomies, Drive folder mapping
docs/              architecture and runbooks
src/pipeline/
  config.py        typed settings (env + YAML)
  models.py        domain models and the processing state machine
  storage/         DrivePort interface + Google and local implementations
  ingest/          Zoom webhook + client, Drive change watcher, reconciler
  media/           ffmpeg audio extraction and probing
  asr/             ASR provider interface + faster-whisper implementation
  syllabus/        taxonomy loading and the searchable index
  classify/        candidate retrieval, LLM classifier, confidence policy
  naming/          filename templates, sanitising, collision handling
  placement/       syllabus node → Drive folder resolution
  orchestration/   task definitions and the state machine runner
  api/             FastAPI app: webhooks, health, admin
scripts/           dry run, syllabus ingestion, Drive mapping, backfill
tests/             unit and pipeline tests
```

## Requirements

**To run the dry run and the tests: Python 3.11+ and nothing else.** Verified working on 3.14.3, which is what this machine has. `pip install -e ".[dev]"` takes about two minutes and pulls no ML dependencies.

The transcription worker needs more, and only the worker:

| | Needed for | Note |
|---|---|---|
| **ffmpeg** on PATH | Audio extraction | Not currently installed here. Windows: `winget install Gyan.FFmpeg` |
| **Python 3.11–3.13** | `[asr]` extra | `ctranslate2`/`torch` wheels lag new CPython releases. Installing `[asr]` on 3.14 is a harmless no-op rather than a build failure. |
| **NVIDIA GPU** | Self-hosted transcription | Optional — set `asr.provider` to a hosted API instead. Both paths are supported ([ADR-004](docs/DECISIONS.md#adr-004--transcription-engine)). |
| **`[embeddings]` extra** | Semantic retrieval | Optional. Pulls torch (multi-GB). Without it, retrieval is lexical-only — worse at paraphrase, but everything still works. |

The split is deliberate: nothing you need to evaluate this design requires a GPU or a large download.

## What is deliberately not built yet

Held back until you decide the open questions in [docs/ARCHITECTURE.md § Open decisions](docs/ARCHITECTURE.md#open-decisions-i-need-from-you) — chiefly whether the Drive account is Google Workspace or personal, which changes the authentication design substantially.
