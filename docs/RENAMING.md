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

## Which layer names the file, and why it differs by subject

This is the part worth getting straight before anything else, and the answer is
**not the same for every syllabus**. It depends on one measurable thing: how
many recordings the folder holds per syllabus subtopic.

### AS Chemistry and IGCSE Maths: the subtopic is the name

9701 has 51 subtopics across 23 Drive folders, and 0580 has 72 across 9. A
chapter folder holds a handful of recordings and each one is about one
subtopic, so the subtopic name *is* the lesson name:

```
3. Chemical Bonding (Chapter 4)
   Sep, 2 - Shapes of Molecules.mp4          <- 3.5
   Sep, 4 - Sigma and Pie Bonds.mp4          <- 13.3
   Sep, 23 - Intermolecular Forces.mp4       <- 3.6
```

Neither syllabus has a lesson layer, so this is what the adaptive default picks
for them (equivalent to `--level subtopic`). Names come from the overlay's
`short_title` — keywords only, never the verbatim syllabus sentence. 3.6 is
titled *"Intermolecular forces, electronegativity and bond properties"* in the
syllabus and `Intermolecular Forces` in a filename.

### A2 Biology: the subtopic is too coarse

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

So for this subject, and only where the ratio demands it, the taxonomy carries
a hand-curated third layer. The level teaching actually happens at is the
**learning outcome**. 12.2 is
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

### Choosing the level

`--level` selects which layer names the file:

| Level | Depth | Names the file after | Use when |
|---|---|---|---|
| `chapter` | 0 | the Drive folder itself | you only want files grouped, not distinguished |
| `subtopic` | 1 | one syllabus subtopic | roughly one recording per subtopic — 9701, 0580 |
| `lesson` | below 1 | the hand-curated teaching layer | several recordings per subtopic — 9700 |

**You normally do not pass it.** The default is adaptive: the script uses the
deepest layer the loaded syllabus actually has — `lesson` where there is a
lesson layer, `subtopic` where there is not — and logs which one it picked and
why, so the choice is visible rather than magic:

```
level: lesson (default) - this syllabus has a lesson layer - 91 lesson nodes
under 20 subtopics - and the lesson is what one recording covers
```

That is the only default that is right for both halves of the table above. A
fixed `subtopic` would be correct for 9701 and 0580 and would quietly discard
9700's lesson layer: every biology subtopic has between two and ten lessons
under it, so the six respiration recordings would come back as six files called
`... - Respiration.mp4`, differing only by date, inside a folder already called
`chap 12-Respiration`. A fixed `lesson` would be a default that is only ever
reached by a fallback on every other subject.

An explicit `--level` still wins, and `lesson` falls back to `subtopic` where a
syllabus has no lesson layer, so it is safe to pass against any subject.

### Never more than two subtopics

A recording that genuinely straddles two subtopics is named for both, joined by
` & ` — `Sep, 23 - Solubility in Solvents & Induced Dipoles.mp4`. Three is
never produced: the cap is enforced in configuration (`max_subtopics`, bounded
at 2), in the `Classification` model, and again in `chapters.py`.

A recording is also never split across two chapter folders. When a lecture
touches several chapters, the transcript is weighted per chapter and the
heaviest wins — one lesson, one folder. `--level chapter` and the per-file
console report both show the runner-up and the margin, so a close call is
visible rather than silent. When nothing was placed at all, there is no winner
and so no runner-up either; the line reads `chapter (none - not placed)
top candidates: ...`, which is the ranking rather than a second place to a
first place that does not exist.

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

   That label is also the one field with no taxonomy behind it, so the rules
   are applied to it twice. The structured-output schema makes every property
   required, which means the model emits a label even on the reply that says it
   could *not* place the lecture — so the label is dropped whenever there is no
   primary node, and the naming layer caps and shortens whatever survives. A
   free-text field that becomes a filename is exactly where "never three
   subtopics" gets quietly lost.

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

## What language the lecture is decoded as

These lectures are not in English. They are taught in mixed Urdu and English —
the explanation is Urdu, the technical vocabulary is English — and both teachers
in this batch code-switch constantly. Whisper hears that mostly as **Hindi** —
ten samples taken at fixed points through the two subjects came back Hindi at
0.69 to 0.99 — but not always: other thirty-second windows of the same two
lectures come back English. Which is the first thing this has to survive.

That leaves three ways to decode it, and only two of them work.

### Never: let the decoder detect it

`language=None` is how faster-whisper spells *decode in whatever you detect*,
and it is the one option that fails outright. Measured on a 3-minute clip of the
11 Aug Chemistry lecture, `small`, int8 on CPU:

| Decode | Speed | Output |
|---|---|---|
| auto-detect (`language=None`) | **0.3x realtime** | Devanagari |
| `language=en` | 5.2x realtime | English, terms intact |
| `task=translate` | 5.5x realtime | English, slightly clumsier |

Two independent failures, both fatal. The output is **unusable**: retrieval
scores English syllabus keywords, and a Devanagari transcript matches none of
them, so every recording lands in review and an ASR failure is
indistinguishable from a classifier one. And it is **seventeen times slower**,
because an uncertain language sends every window through Whisper's
temperature-fallback retries — up to six decodes where one would do. A
50-minute lecture goes from ten minutes to nearly three hours.

So `None` never reaches the decoder. `transcribe()` raises on it rather than
treating it as an unset default, and a test holds that shut.

### `--language auto` means *choose*, not *delegate*

Both remaining decodes produce usable English:

| Candidate | What it does |
|---|---|
| `transcribe` as `en` | keeps English terms verbatim, renders leftover Urdu phonetically in Latin script |
| `translate` from `hi` | turns the Urdu into English prose, sometimes at the cost of a technical term |

`--language auto` detects the language once per folder, then picks between those
two by running both on one 3-minute sample and scoring each on **how much of
that subject's syllabus vocabulary it recovers**
(`src/pipeline/asr/language.py`). The transcript exists to be matched against
the taxonomy, so the decode that recovers more of the taxonomy is better by
definition, and the metric cannot drift away from the goal it stands in for.
Translation has to *win* by a margin to be chosen; inside the margin the tie
goes to `transcribe`, which keeps technical terms verbatim.

Two details of the sampling are load-bearing, and both were wrong until they
were measured:

- **Detection votes across the whole 3-minute sample** rather than stopping at
  faster-whisper's default first 30-second window. One window does not
  describe a code-switched lecture: single windows of these two lectures
  returned `en` 0.40, `en` 0.65, `hi` 0.76, `hi` 0.87, `hi` 0.94 and `hi`
  0.99 — same teachers, same model, different thirty seconds. Both failure
  modes were live. `en` 0.40 is *below* the confidence floor, so the probe
  gave up and assumed English; `en` 0.65 is above it, so the probe concluded
  "already English" and never ran the bake-off at all. With all six windows
  classified and the majority taken, the same Maths folder detects `hi` at
  0.90 and the bake-off runs. Six encoder passes over 30 seconds cost
  fractions of a second, against a decision that commits the whole folder.
- **The sample is drawn from three points in the lecture**, not one, using the
  same total amount of audio and therefore costing nothing extra. See the
  scatter numbers below for why.

The decision, the detected language and both scores are logged, copied into
`rename_plan.json` beside every file, and remembered in
`.transcripts/language.json` so ordinary re-runs — a different `--scope`, a
different `--level` — stay instant off the transcript cache. `--reprobe` forces
a fresh probe.

```
language: auto - probing Aug, 17 - IG Mathematics.mp4 ...
  cannot decode it (Aug, 17 - IG Mathematics.mp4: InvalidDataError: ...); probing the next recording
language: auto - probing Aug, 12 - Indices I.mp4 ...
language: auto -> detected hi 0.90 -> transcribe as en (probe inconclusive (22% margin), defaulting to English)
  probe translate-hi       28.6 syllabus terms per 1000 words
  probe transcribe-en      22.4 syllabus terms per 1000 words
  decision cached in ...\.transcripts\language.json
```

The probe listens to the smallest recording in the folder, because reaching the
middle of a file means decoding all of it and that decode is the probe's only
real cost. It falls through to the next one when that file cannot be decoded,
which is not hypothetical: an interrupted upload produces a *short* file, so the
cheapest candidate is disproportionately likely to be the broken one — and in
the Maths folder it is exactly that.

### The default is `en`, and this is the measurement behind it

The old 10% margin was derived from a 2% difference on a single 3-minute clip,
which is not enough evidence to set a threshold with. It was re-measured
properly: **four 12-minute clips of real teaching, 48 minutes in total**,
sampled from the middle of two Chemistry and two Maths lectures, each decoded
both ways through the same slicing and the same syllabus priming the pipeline
uses.

| Clip | | syllabus terms /1000 | distinct terms | residual Urdu /1000 | speed |
|---|---|---:|---:|---:|---:|
| chem 11 Aug | transcribe `en` | **23.85** | 25 | 0.00 | 3.9x |
| | translate `hi` | 22.18 | 26 | 0.00 | 5.8x |
| chem 16 Aug | transcribe `en` | **15.92** | 18 | 3.65 | 3.5x |
| | translate `hi` | 15.03 | 17 | **0.00** | 5.2x |
| math 12 Aug | transcribe `en` | 22.55 | 26 | 0.00 | 5.5x |
| | translate `hi` | **25.00** | 29 | 0.00 | 6.1x |
| math 10 Aug | transcribe `en` | 22.94 | 27 | 0.00 | 2.5x |
| | translate `hi` | **23.81** | 27 | 0.00 | 2.8x |

Pooled per subject, which is the level the probe actually decides at:

| Subject | transcribe `en` | translate `hi` | |
|---|---:|---:|---|
| AS Chemistry | **19.7** | 18.7 | English by 5.7% |
| IGCSE Maths | 22.8 | **24.4** | translation by 7.3% |

**The two subjects give opposite answers.** That is a real result and not an
inconvenience — it is why the probe is memoised per subject rather than per
language, since the score counts *that subject's* vocabulary and a decision
reached for Chemistry is not evidence about Maths. If a future teacher produced
a decisive difference in one subject and not the other, the machinery would
already record that correctly.

Neither of these differences is decisive, though, for two independent reasons.

**One: both are smaller than the probe's own scatter.** Running the probe at
five points through one lecture, with the single-window sampling it used to
have:

| | 20% in | 35% | 50% | 65% | 80% |
|---|---|---|---|---|---|
| chem 11 Aug | translate 11.5% | translate 15.5% | **en 1.7%** | translate 3.0% | translate 5.2% |
| math 12 Aug | en 1.6% | en 1.8% | en 17.8% | **translate 11.4%** | en 0.8% |

The winner changes within a single lecture, and the margin ranges from 0.8% to
17.8%. Spreading the same 180 seconds over three points instead of one steadies
it — Chemistry then picks `transcribe` from all three sampling positions tried,
matching the 48-minute result — but does not remove the swing: the margins were
4.6%, 10.3% and 20.1% for Chemistry and 2.1%, 16.0% and 16.3% for Maths.

The live probe says the same thing from the other direction. Run for real on
the two folders, against the audio it actually samples, it reported **English
by 47%** for Chemistry and **translation by 22%** for Maths — against measured
48-minute differences of 5.7% and 7.3%. Each points the way its subject's long
measurement points, and each is three to eight times the size of it.

So the true effect (under 8%, either way) is entirely inside the range the
probe wobbles by (up to 47%). `MIN_RELATIVE_MARGIN` was raised from **0.10 to
0.25** accordingly: it now sits *above the probe's noise floor* rather than at
the size of a difference worth having. Its job is to catch a decode that is
dramatically better — a teacher whose Urdu defeats English transcription
outright — and to refuse to arbitrate anything smaller, because at this sample
size it genuinely cannot. At the old 0.10, the Maths folder would have been
switched to translation on that 22% reading — for a subject whose real
difference is 7.3% and whose technical vocabulary translation damages.

**Two: on the terms a human would check, transcription wins in both subjects.**
Counting subject-specific technical vocabulary across the same four clips —
`nucleon`, `isotope`, `relative atomic mass`, `surd`, `indices`, `denominator`
and so on:

| Subject | transcribe `en` | translate `hi` |
|---|---:|---:|
| AS Chemistry | **74** | 69 |
| IGCSE Maths | **88** | 79 |

Translation writes tidier sentences and loses content inside them.

### What translation buys, and what it costs

It buys the removal of untranslated Urdu. On the 16 Aug Chemistry clip,
transcribing as English left four romanised Urdu function words; translation
left none.

It costs technical content, in the same passage:

| | 16 Aug Chemistry, defining the unified atomic mass unit |
|---|---|
| transcribe `en` | "one u is basically **one twelfth** the mass of carbon 12 … one u kya hota you ka matlab kya u ki full form kya" |
| translate `hi` | "1u is basically **112th** the mass of carbon-12 … What is 1u? What is u? What is the full form of u?" |

Translation produced the cleaner sentence and dropped the fact. `one twelfth`
is a retrieval keyword for *Relative Atomic Mass*; `112th` is nothing. On the
same clip translation also took `carbon 12` from three mentions to zero
(keeping only the hyphenated form) and `abundance` from three to two. On the
10 Aug Maths clip it took `power` from 14 to 9, `fractional` from 4 to 1, and
`surd` from 6 to 5.

That is the trade in one line: **translation tidies the half of the transcript
nobody scores and damages the half that decides the filename.**

### How much untranslated Urdu is actually in there

Enough to notice, not enough to matter. Across all seven shipped transcripts —
**6.8 hours, 34,553 words, 4,173 segments** — a scan for romanised Urdu and
Hindi function words finds **23 tokens, 0.07% of words**, in **7 segments,
0.17%**, in three of the seven files. It is not spread thinly; it is four
sentences:

```
583011b70e79c4ab  04:12  bola bhai aise nahi ho na ki tum nahi number pakar ke basun ko yaha be rakh
78fca236111a7606  72:38  Karah kya hai hum nahi aaphe? Aapne sai likkha AB to the power m but what did you actually do?
97b8cc2742e875a6  40:05  Britta, power toh abhi maina chhiri nahi na power waise hi rakhi bhiya.
97b8cc2742e875a6  40:09  0.2 ko maina 2 over 10 likh sakta hoon?
```

None of them carries syllabus vocabulary, so none of them changes a
classification. And there is **no Devanagari or Arabic script anywhere in the
seven files** — zero non-Latin characters — which is the failure that would
actually break retrieval, and forcing English is what prevents it.

The more common damage is not untranslated Urdu at all but plain mishearing:
`"a port exam"` for *a board exam*, `"that last viking"` for *last week*. Those
are English words the acoustic model got wrong, not Urdu it failed to render,
so no choice of language or task addresses them — a bigger model or a better
priming prompt would be the lever, and [Choosing an ASR
model](#choosing-an-asr-model) reports what happened when both were tried.

### When to use `--language auto`

Pass it for **a teacher or a subject this has not been measured on**. It costs
one probe per folder — a model load, one recording decoded, two 3-minute
decodes, roughly two minutes — and then nothing, because the answer is cached
in `.transcripts/language.json` and reused until `--reprobe`.

`--language en`, the default, is right for these two teachers on this evidence:
translation recovers no more syllabus vocabulary than transcription does,
recovers *less* of the technical vocabulary, and costs a probe to discover it.
`--language ur` — or any ISO code — forces that language with no probe, for a
class that genuinely is taught in one.

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
