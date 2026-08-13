# Renaming a folder of recordings

The narrow job: a folder already sits in the right place in Drive, the videos
inside it are called `class 1 (2 aug).mp4`, and they should be called
`Aug, 2 - ATP as the Universal Energy Currency.mp4`.

This is a strict subset of the full pipeline — no move, no folder creation, no
transcript publishing, no Zoom — and it needs **no credentials at all**.

```bash
pip install -e ".[asr,syllabus]"

# dry run: prints what it would do, renames nothing
python scripts/rename_folder.py \
    --folder "C:/path/to/chap 12-Respiration" \
    --syllabus config/syllabi/a2-biology-9700.yaml \
    --scope bio.topic_12

# same command with --apply does the renames and writes an undo file
```

---

## Why the filename cannot come from a syllabus subtopic

This is the part worth getting straight before anything else.

Cambridge 9700 topic 12, *Energy and respiration*, has exactly **two**
subtopics:

| Code | Title |
|---|---|
| 12.1 | Energy |
| 12.2 | Respiration |

The folder has **six** recordings. Naming from syllabus subtopics gives six
files two names between them, and one of those names — "Respiration" — is the
name of the folder they are already in. That is not an improvement over
`class 4 (9 aug).mp4`; it is a different way of saying nothing.

The level teaching actually happens at is the **learning outcome**. 12.2 is
fourteen outcomes, and they group naturally into the lessons a teacher gives:
glycolysis, the link reaction, the Krebs cycle, oxidative phosphorylation, and
so on. So the taxonomy carries a third layer:

```
12 Energy and respiration          ← syllabus topic     (the Drive folder)
   12.1 Energy                     ← syllabus subtopic
      ATP as the Universal Energy Currency    ← lesson  (the filename)
      Respiratory Substrates and Energy Values
      The Respiratory Quotient
      Investigating RQ with a Respirometer
   12.2 Respiration                ← syllabus subtopic
      Overview of Aerobic Respiration         ← lesson
      Glycolysis
      The Link Reaction
      The Krebs Cycle
      Oxidative Phosphorylation and Chemiosmosis
      Structure and Function of Mitochondria
      Anaerobic Respiration and Energy Yield
      Adaptations of Rice to Waterlogged Conditions
      Investigating the Rate of Respiration
```

Topics, subtopics and learning-outcome text are extracted verbatim from the
official syllabus PDF and marked `verified: true`. The lesson layer is not in
the syllabus — it is teaching structure — so it is marked `verified: false`,
and every lesson **cites the outcome numbers it covers**. The ingester copies
that outcome text onto the lesson node, so a lesson is matched against real
Cambridge prose rather than against a title this repo invented.

`--level` selects which layer names the file: `lesson` (default), `subtopic`,
or `topic`.

---

## Where the syllabus comes from

```bash
python scripts/ingest_syllabus.py \
    --pdf 9700-2025-2027-syllabus.pdf \
    --overlay config/syllabi/overlays/a2-biology-9700.yaml \
    --out config/syllabi/a2-biology-9700.yaml
```

The split between the two inputs is the whole design:

| Source | Carries | Marked |
|---|---|---|
| The PDF | topic/subtopic numbers, titles, learning-outcome text | `verified: true` |
| The overlay | keywords, aliases, Drive folder names, teaching order, lessons | `verified: false` |

Learning outcomes are the highest-value text in the pipeline — they are what
retrieval scores against — and typing 134 of them by hand is the easiest
possible place to introduce an error that then misfiles lectures silently for a
year. So they are never typed.

### The one hard part of parsing the PDF

Cambridge sets subject content in two columns, and a bare number on its own
line is ambiguous three ways: `12` is a topic heading, an outcome number, and a
page number, all in the same document. Font *size* cannot separate them —
everything in the PDF is 10pt. Headings are set in the bold cut of the face and
body text in the light cut, so the parser reads spans rather than flat text and
carries a bold flag. That flag is the entire difference between a correct parse
and a taxonomy full of phantom topics numbered 38 and 40.

---

## What the classifier actually does

Full detail is in [CLASSIFICATION.md](CLASSIFICATION.md). In short, per lecture:

1. Split the transcript into ~4-minute overlapping windows.
2. Score every window against every candidate node (BM25, plus embeddings when
   `sentence-transformers` is installed).
3. Aggregate into a **coverage profile** — "55% ATP, 18% Respiration, 9% Energy".
4. Take the top node, or hand off to Claude to choose among the top candidates
   and cite evidence.
5. Anything under the confidence threshold gets ` (review)` appended to the
   proposed name rather than being filed silently.

### Writing keywords that actually work

The overlay's keywords are where most of the accuracy lives, and two rules
about them are not obvious. Both are enforced by tests in
`tests/test_biology_taxonomy.py`, because both failures are silent — nothing
errors, the classifier just gets quietly worse.

**A keyword must be a term someone says, not a description of the topic.**
Retrieval counts a keyword only when *every* one of its tokens appears in the
same window. "Proximal convoluted tubule" is fine — a teacher saying it puts
all three tokens in one window. "Explain why the energy yield is much greater
in aerobic" never fires at all.

**A keyword must separate a lesson from its siblings, which is not the same as
being what the lesson is about.** Bare `ATP` was originally a keyword on the
ATP lesson. Every lecture in chapter 12 says "ATP" constantly, so it fired on
all of them and pulled two unrelated lectures onto that node. Removing it took
the ATP lecture from 55% coverage to 82%. Shared terms are fine — all three
lessons under 13.2 legitimately discuss light intensity — but every lesson
needs *some* vocabulary its siblings lack, or nothing can ever select it.

### `--scope` is the cheapest accuracy you will ever buy

When you already know the folder is chapter 12, letting the classifier also
consider photosynthesis and gene therapy adds nothing but ways to be wrong.
The electron transport chain appears in respiration *and* photosynthesis, and
shares most of its vocabulary between them. `--scope bio.topic_12` cuts the
candidate set from 118 nodes to 17 and removes that entire failure mode.

You usually do not have to pass it. If the folder's name matches a node's
`drive_folder_name` — `chap 12-Respiration` is recorded in the overlay — the
scope is inferred and logged. When nothing matches, the script says so and
considers the whole syllabus rather than guessing at a chapter, because
silently scoping to the wrong one is worse than not scoping at all.

---

## Measured result on chapter 12

Six recordings, 5.6 hours of audio, keyword classifier, `--model small`,
scope inferred from the folder name:

| Original | Proposed | Coverage | Confidence | Correct? |
|---|---|---:|---:|:--:|
| `class 1 (2 aug).mp4` | `Aug, 2 - Structure and Properties of ATP.mp4` | 82% | 0.92 | ✅ |
| `class 2 (4 aug).mp4` | `Aug, 4 - The Need for Energy in Living Organisms (review).mp4` | 28% | 0.53 | ✅ |
| `class 3 (6 aug).mp4` | `Aug, 6 - Coenzymes - NAD and FAD (review).mp4` | 21% | 0.49 | ✅ |
| `class 4 (9 aug).mp4` | `Aug, 9 - Glycolysis (review).mp4` | 24% | 0.57 | ✅ |
| `class 5 (11 aug).mp4` | `Aug, 11 - Glycolysis (review).mp4` | 14% | 0.46 | ❌ — is the link reaction |
| `class 6 (12 aug).mp4` | `Aug, 12 - The Krebs Cycle (review).mp4` | 22% | 0.50 | ✅ |

**Five of six correct; one wrong and flagged.** Only the first file cleared the
0.85 threshold, and it is the only one that opens without revising the previous
lesson. That is the whole story of this chapter in one column.

Note also what the coverage numbers are *not* saying. A 22% coverage figure
does not mean 22% of the lecture was the Krebs cycle — it means 22% of the
four-minute windows had the Krebs cycle as their top match, with the rest split
across revision, exam chat, "can you guys hear me", and the reflection exercise
at the end. Real lessons are mostly not syllabus content, and any threshold
tuned as though they were will be wrong.

## Keyword-only or an LLM?

Both are implemented, behind one interface, and `--classifier` switches between
them. They are not equivalent, and the difference is not subtle.

**Keyword retrieval gets most of the way there for free.** On the chapter-12
recordings it names the right lesson on five of six, with no API key and no
per-lecture cost.

**It is not enough to decide unsupervised — and the sixth file shows why.**
The 11 Aug lecture spends its first twenty minutes revising the previous
lesson's glycolysis before teaching the link reaction. Counting topic terms
across the file:

| 10-minute window | glycolysis | link reaction |
|---|---:|---:|
| 10–20 (revision) | 20 | 6 |
| 20–30 (new content) | 9 | **51** |
| 30–40 | 30 | 15 |
| **whole file** | **69** | **85** |

The link reaction owns its own window 51-to-9, but glycolysis leads for long
enough that retrieval called the lecture *Glycolysis* — the name of the
**previous** lesson. It flagged the result at 0.46 confidence, so the file
routes to review instead of being misfiled silently, and the safety mechanism
did its job. The label was still wrong.

Three specific things go wrong:

1. **Confidence lands in the grey zone.** Coverage is diluted by everything a
   real lesson contains that is not syllabus content — roll call, "can you
   guys hear me", homework, the reflection exercise at the end. A lecture that
   is unambiguously about ATP scores 55% coverage, not 95%, and the honest
   confidence that follows sits just under the review threshold. Keyword
   matching cannot tell "taught this" from "mentioned this a lot", so it
   *should* be uncertain — but that means a human clears the queue.

2. **Recaps are indistinguishable from content.** This is the dominant effect,
   not an edge case. Every chapter-12 recording except the first opens with a
   recap of the previous lesson — "let us quickly revise whatever we have done
   in the last class and then I will begin" — running five to fifteen minutes.
   Those windows are real syllabus content and score exactly like real
   syllabus content, so they split the coverage and depress confidence on
   every file after the first.

   The system prompt already tells the LLM that a lecture spending five
   minutes on a recap and fifty on new material belongs to the new material.
   Keyword counting has no way to express that, and no threshold tuning fixes
   it: the recap genuinely *is* the previous topic, so the retriever is not
   wrong, it is just answering a different question.

   This was deliberately not patched with a heuristic. Discounting the opening
   windows would break genuine part-2 lectures, which look identical until you
   read them — and over-fitting a baseline classifier to six recordings buys
   nothing that the LLM step does not buy properly.

3. **The label is only ever a node title.** The keyword path can return
   "The Krebs Cycle" because that node exists. It cannot notice that a lesson
   covered the link reaction *and* the Krebs cycle and propose a name saying
   so. `suggested_topic_label` from the LLM path is what produces a filename
   more precise than the taxonomy.

The practical recommendation: **run keyword-only first**, look at the plan, and
add the LLM when you want the queue of `(review)` files to shrink.

The cost is not the reason to hesitate. One call per lecture, measured on the
real chapter-12 prompts: **~6,900 input tokens** (eight candidate nodes with
their learning outcomes, plus a windowed transcript digest) and one to two
thousand output. At Claude Opus 5's $5 / $25 per million tokens that is **about
seven cents a lecture** — roughly forty cents to classify this entire chapter,
against the hour of CPU the transcription itself took. Classification is the
cheap half of this pipeline by a wide margin.

The repo's LLM path calls **Claude** (`ClassifySettings.model`, default
`claude-opus-5`) with a JSON schema on the response, so a malformed answer is
impossible rather than merely unlikely. Nothing about the design is
Claude-specific — it needs structured output and long context, which any
frontier model provides.

---

## Choosing an ASR model

`--model` takes any faster-whisper model name. Measured on this machine
(12 CPU threads, no GPU, `compute_type=int8`):

| Model | Speed | On this audio |
|---|---|---|
| `small` | **5.2–6.1x realtime** (5.6 h of lecture in 59 min) | Correct sentence boundaries and capitalisation; got `ribose` right |
| `distil-large-v3` | **1.9x realtime** — about 3x slower | Long un-punctuated lowercase runs; wrote `ribos` and `iboes` for `ribose` |

**The bigger model was not better here.** That is worth stating plainly because
it is the opposite of the usual advice. On the same 3-minute clip,
`distil-large-v3` cost three times the compute and lost to `small` on the terms
that matter, largely because it degrades to run-on lowercase text — which also
makes the windowed transcript harder to read when you are checking a
classification by hand. `small` is the right default for this workload, and the
6.1x figure means a week of lectures transcribes over lunch.

Both models fail on the same words: `pyrimidine` came out as "primidine" /
"primideen", `pentose` as "panto" / "pantos". These are AS-level terms that
recur inside A2 lessons.

### A fix that did not work

The obvious remedy is to add those terms to the ASR priming prompt — the
decoder conditions on `initial_prompt`, so in principle it should see the right
spelling before it has to guess. **Tested, and it made no difference**: the same
clip transcribed with `adenine, guanine, ribose, purine, pyrimidine, pentose`
appended to the prompt produced byte-identical output for all six terms.

Priming is still worth keeping — it costs nothing and is the documented
mechanism — but do not assume it rescues a term the model gets wrong. Verify on
your own audio before relying on it. The reassuring part is that it does not
matter much here: "pyrimidine" is not a chapter-12 retrieval keyword, so
mangling it changed no classification.

What did measurably matter:

- **Turn off `condition_on_previous_text`.** Left on, one hallucinated line
  conditions the next and a lecture can spiral into repeated text — which looks
  like content to the classifier.
- **Enable the VAD filter.** It drops the dead air before a lesson starts,
  where Whisper otherwise invents speech.

A GPU changes the arithmetic completely: `large-v3` on CUDA runs faster than
`small` does on this CPU, at which point the comparison above stops applying.

---

## Getting the names back into Drive

`rename_folder.py` renames **local files**. To close the loop today:

1. Download the folder (Drive → right-click → Download gives a zip).
2. Run the script with `--apply`.
3. Rename in Drive to match, or re-upload.

Doing it in place through the Drive API is a metadata-only `files.update` call
— the video is never re-uploaded — and needs the credentials described in
[SETUP.md](SETUP.md).

**Check who owns the files first.** A folder under *Shared with me* is owned by
whoever shared it, and renaming requires Editor access. If you only have Viewer
access the rename will fail no matter how the request is made, and the fix is a
permission change, not a code change.

---

## Undo

`--apply` writes `rename_undo.json` into the folder, mapping every new name
back to its original. Renames go through a staging name first, so a plan that
legitimately swaps two names does not clobber one of them.
