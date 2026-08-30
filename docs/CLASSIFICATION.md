# Classification

How a lecture transcript becomes a syllabus location, and how wrong answers get caught.

The requirement: **classify on content, not on title.** A Zoom meeting called "Chemistry Class — Aug 10" tells you nothing. The transcript tells you everything.

---

## Two designs, and why the second replaced the first

```
                          the default: classify.provider = openai
transcript ──┬─► the whole transcript, timestamped ──┐
             │                                       ├─► one ChatGPT call
whole syllabus subtree, chapter → subtopic → lesson ─┘   (ids enumerated in
             │                                            the reply schema)
             └─► chapter id + 1–3 subtopic ids ──► confidence gate
                                                   ──► file | review

                          the baseline: classify.provider = keyword
transcript
   │
   ├─► window into ~4-minute chunks with overlap
   │
   ├─► for each window:  BM25 score  ──┐
   │                     embedding score ─┴─► fused score per syllabus node
   │
   ├─► aggregate across windows ──► coverage profile + top-K candidates
   │
   ├─► take the top node (or, with provider `anthropic`, hand the shortlist
   │   to Claude to choose among and cite evidence for)
   │
   └─► confidence gate ──► file  |  review  |  leave alone
```

The retrieval design below is sound and its coverage profile is genuinely
useful, but it decides from **vocabulary overlap**, and vocabulary overlap
cannot distinguish two things that matter:

- **Taught from mentioned.** A teacher who explains glycolysis for fifty
  minutes without saying "glycolysis" loses to a chapter whose keywords happen
  to be said out loud.
- **Content from recap.** Every lecture after the first opens with five to
  fifteen minutes revising the previous one. Those windows are real syllabus
  content and score exactly like it, so the recap competes with the subject —
  and sometimes wins. See [RENAMING.md](RENAMING.md) for the 11 August
  measurement where it did.

Neither failure is visible in the filename it produces. So the default is now
the model reading the lecture, and **the candidate list is the whole syllabus
rather than a retrieval shortlist**: narrowing the options by keyword score
before anything has read the transcript puts both failures back, one step
earlier and harder to see.

Retrieval still runs on the model path. Its coverage profile is reported —
printed on every `rename_folder.py` line, stored in `rename_plan.json` and the
sidecar — as the keyword evidence, so the two can be seen to disagree. Nothing
is decided by it.

Stages 1–3 below describe the retrieval path. Stage 4 describes both decisions.

---

## Stage 1 — Windowing

The transcript is split into windows of roughly four minutes with about thirty seconds of overlap.

Four minutes is long enough to contain a coherent explanation and short enough that a window rarely spans two topics. The overlap stops a topic transition landing exactly on a boundary and being attributed to neither side.

Windows carry timestamps throughout, which is what makes the final evidence quotes clickable.

---

## Stage 2 — Retrieval

Each window is scored against every syllabus node by two independent methods.

### Lexical (BM25)

The node's `keywords`, `aliases`, and learning-outcome text form the document. The window is the query.

This is what catches decisive terms. "Halogenoalkane" appears in one chapter and nowhere else; when it appears fourteen times in a window, that is close to conclusive. Domain jargon carries enormous signal in a technical syllabus, and BM25's rarity weighting is exactly the right tool.

### Semantic (embeddings)

Learning-outcome text is embedded once at startup and cached. Each window is embedded and compared by cosine similarity.

This catches what BM25 misses: paraphrase. A teacher can spend ten minutes explaining that energy is needed to remove an electron from a gaseous atom without once saying "ionisation energy". Lexical matching sees nothing; semantic matching sees it clearly.

Default model is a multilingual sentence-transformer, which matters if lectures code-switch — an English-only embedding model degrades badly on mixed-language input.

### Fusion

Scores are normalised and combined with a configurable weight (default 0.5/0.5). Lexical is weighted a little higher for subjects with dense specialist vocabulary — chemistry is one — and the weight is per-subject config.

---

## Stage 3 — Aggregation and the coverage profile

Per-window scores aggregate into a per-node profile:

```
chem.2_1    Relative Atomic Mass     coverage 0.78   peak 0.91   windows 14/18
chem.1_2    Isotopes                 coverage 0.18   peak 0.44   windows  3/18
chem.2_2    Moles & Avogadro         coverage 0.04   peak 0.21   windows  1/18
```

Nodes here are **subtopics** (depth 1). Their **chapters** (depth 0) are what
the file is placed into, and coverage is rolled up per chapter to decide that —
see "One chapter, one to three subtopics" below.

`coverage` is the share of the lecture attributable to a node. This is the number that makes multi-topic lectures work:

- One node dominant → straightforward, file it there
- Two nodes with real coverage → both may name the file, if they are in the same chapter and the second clears both secondary bars
- Nothing above the floor → the lecture may be revision, an exam debrief, or admin. Route to review rather than forcing a match

It also detects **continuation lectures**: high coverage on a node that the previous lecture also covered, with the lecture opening mid-topic, indicates a part 2. That feeds the naming rules.

---

## Stage 4 — The decision

### The default: ChatGPT (`src/pipeline/classify/openai_classifier.py`)

One call per lecture. The model receives:

1. **The whole syllabus subtree** — every chapter, subtopic and lesson, nested,
   each with its id, verbatim title, filename label, aliases, keywords and
   learning outcomes. Grouping nodes are shown and marked *do not return this
   id*: the tree is what makes "one chapter" a decision the model can take, but
   a lecture named after the chapter it is already filed in says nothing twice.
2. **The whole transcript**, one timestamped line per 45 seconds. Not a window
   digest — what a lecture is about is a fact about all of it, and the
   timestamps are what let the model tell an opening recap from the subject.
3. **Weak signals** — date, teacher, Zoom meeting topic, original filename,
   labelled as weak and explicitly not to be classified from. These filenames
   are `class 4 (9 aug).mp4`; they are evidence of nothing.

It also gets told the audio is code-switched Urdu and English with English
technical terms, sometimes transcribed phonetically, and to read through the
transcription errors rather than looking for exact syllabus wording.

```json
{
  "chapter_id": "bio.topic_12",
  "subtopics": [{"node_id": "bio.12_2.glycolysis", "coverage": 0.7}],
  "confidence": 0.9,
  "evidence": [{"quote": "string", "start_s": 0.0}],
  "reasoning": "string",
  "abstain": false,
  "abstain_reason": "string"
}
```

**There is no field a filename could come from.** That is the design, not an
omission. `chapter_id` and every `node_id` are declared in the schema as a JSON
Schema `enum` of the actual syllabus ids and sent with `strict: true`, so a
conforming reply *cannot* name something the syllabus does not have; the name
is then rendered from the taxonomy's own `short_title` for those ids. A model
that invents a subtopic has no way to spell it.

Three checks sit behind that, because a schema constrains the reply and not the
world — the enum is dropped for a syllabus too large to enumerate, and a stale
schema is always possible:

- an id the taxonomy does not know is dropped;
- an id belonging to a different chapter than the one just chosen is dropped,
  because a name that disagrees with the folder misleads and only one of the
  two is visible in a listing;
- if nothing survives, the result is an **abstain** naming the ids that were
  rejected — never a fallback guess.

### The baseline, and the Claude path

`provider: keyword` takes the top node off the coverage profile.
`provider: anthropic` hands the top ~8 candidates, a window digest and the
coverage profile to Claude, which chooses among them — the design the OpenAI
path replaces. Its reply carries a `suggested_topic_label`, a free-text field
that becomes a filename, which is exactly where "never more than three
subtopics" gets quietly lost; `naming/template.py` caps and shortens it for
that reason.

## One chapter, one to three subtopics

Four rules, in `src/pipeline/classify/chapters.py` and enforced wherever a
`Classification` is built:

1. A recording belongs to **exactly one chapter**. Chapters are Drive folders,
   and a lecture is one lesson; it goes in one folder. On the model path the
   chapter is chosen first and the subtopics are constrained to it. On the
   keyword path each subtopic's coverage is attributed to its chapter, the
   totals are normalised, and the heaviest wins — so four subtopics at 15% beat
   one at 50%, which is the right answer when the lecture worked through most
   of a chapter.
2. Most recordings cover **one subtopic**, and the filename is that subtopic's
   `short_title`.
3. A recording that genuinely straddles two or three gets each of them, joined
   by `naming.subtopic_separator`. On the model path that judgement is the
   model's, made from having read the lecture. On the keyword path a second
   subtopic must clear an absolute coverage bar *and* a ratio of the first —
   the two bars fail differently, and one alone lets through either a
   four-minute aside or a weak runner-up to a weak winner — and a **third is
   never selected at all**, whatever the cap allows. Coverage counting can rank
   subtopics; it cannot say a third one was taught, and a name that asserts it
   was would be a guess dressed as a fact.
4. **Never four.** Capped in three places — `NamingSettings.max_subtopics`
   (`le=3`), `Classification`'s validator, and `HARD_SUBTOPIC_CAP` in
   `chapters.py`, which is imported from `models.MAX_SUBTOPICS_IN_NAME` so the
   cap cannot half-move — because by four the filename is a table of contents,
   and a name nobody reads is worse than a short name that is only mostly
   right.

The chapter's own node is never chosen as the second subtopic. Without that,
the recap minutes at the start of a lecture produced
`Aug, 10 - Relative Atomic Mass & Atomic Structure.mp4` filed inside *Atomic
Structure* — the folder name said twice.

Schema-validated at the API layer, so a malformed response is impossible rather than merely unlikely.

**`evidence` is required and non-empty unless abstaining.** This is the most important constraint in the whole design. Requiring the model to cite specific timestamped quotes:

- Makes every classification auditable — jump to 6:52 and check
- Makes failures diagnosable — wrong quote means retrieval failed; right quote with wrong conclusion means the taxonomy is ambiguous
- Empirically improves accuracy, because a model that must ground its answer commits less to a plausible-sounding wrong one

**`suggested_topic_label`** is a free-text name, and only the `anthropic` path
produces one. It exists because a lecture covering a specific slice of a node
can be named more precisely than the node — `Aug, 10 - Relative Atomic Mass.mp4`
rather than the blunter `Aug, 10 - Atomic Structure.mp4`.

The OpenAI path does not have this field and does not want it. It is the one
place a name reaches a filename with no taxonomy behind it, which is why
`naming/template.py` caps and shortens whatever arrives there: an abstaining
reply still carries a label, because the schema marks every property required,
and the unknown-node-id path clears the ids without clearing the label. Getting
the same precision from node ids alone is what the hand-curated lesson layer is
for — see [RENAMING.md](RENAMING.md).

**`abstain`** exists so the model has an honest way out. Without it, a model asked to choose from a list will choose from the list, even when the right answer is "this is an exam debrief and belongs in none of these".

---

## Stage 5 — The gate

```python
if result.abstain or result.primary.confidence < settings.review_threshold:
    → _Needs Review, notify, filename marked with "?"
elif result.primary.coverage < settings.coverage_floor:
    → _Needs Review  (confident about a topic that was barely covered — suspicious)
else:
    → file it
```

Two independent thresholds, because they catch different errors. High confidence with low coverage means the model is sure about something that occupied four minutes of a ninety-minute lecture — that's usually a lecture whose real subject isn't in the taxonomy at all.

### Calibrating the threshold

Do not guess it. Run shadow mode over lectures already filed correctly — your existing archive is a labelled dataset — and plot agreement against confidence. Pick the threshold where automatic filing is right at the rate you're comfortable with.

Start conservative (0.85). Every week of corrections is evidence for loosening it. A threshold that produces zero reviews in the first month is set too low, not working well.

---

## Failure modes and their fixes

| Symptom | Likely cause | Fix |
|---|---|---|
| Confidently wrong, evidence quotes are off-topic | Retrieval didn't surface the right candidate | Add keywords/aliases to the correct node |
| Confidently wrong, evidence quotes are on-topic | Two syllabus nodes genuinely overlap | Add distinguishing keywords to both; consider merging them |
| Everything routes to review | Threshold too high, or taxonomy too shallow | Lower the threshold; add subtopic nodes |
| Teacher shorthand never matches | Missing aliases | The correction loop proposes these automatically — approve them |
| Transcript mangles technical terms | ASR vocabulary priming isn't reaching the model | Check `initial_prompt`/`hotwords` are populated for the subject |
| Multi-topic lectures always land on the wrong one | Coverage floor too low | Raise `coverage_floor`; review the secondary-topic policy |

Every one of these is diagnosable *because* evidence quotes are mandatory. That is the payoff.

---

## The feedback loop

A human correction (moving the file in Drive) writes a record containing the transcript, the classifier's answer, and the human's answer. Three things use it:

**Immediately** — distinctive n-grams from the corrected transcript that don't appear in the correct node's keywords are proposed as new aliases, for approval. This is how "AOD" gets learned: it appears in a transcript, a human files it under Atomic Structure, and the system proposes adding it as an alias.

**Per-run** — the most recent corrections for a subject are included as few-shot examples in the classifier prompt, so recent human judgement directly shapes the next decision.

**Over time** — the correction set is an evaluation dataset. When you change a threshold, swap a model, or restructure the taxonomy, you can measure whether it helped instead of guessing. Without this, every tuning change is a vibe.

---

## Cost

Retrieval is local: BM25 is trivial, embeddings are one small model over a few dozen windows — under a second on CPU.

The Claude call sends candidates plus a window digest, not the full transcript, so it's a few thousand input tokens and a few hundred output. A few cents per lecture, against several minutes of GPU time already spent transcribing it. Classification is not the expensive part and should not be optimised as though it were — a cheaper model that misfiles lectures costs far more in human time than it saves.
