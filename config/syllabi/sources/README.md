# Syllabus sources

The raw Cambridge material every taxonomy in `config/syllabi/` is built from.
Committed rather than gitignored, for one blunt reason: **without these, four of
the five subjects cannot be rebuilt.** The two PDFs can be re-downloaded from
Cambridge; the two JSON files are custom extractions with no public URL, so if
they are lost the AS Chemistry and IGCSE Maths taxonomies are frozen at whatever
was last generated.

5 MB total. Cheap insurance.

## What's here

| File | Feeds | Ingester |
|---|---|---|
| `as-chemistry-9701.json` | `as-chemistry-9701.yaml` | `ingest_syllabus_json.py` |
| `igcse-maths-0580.json` | `igcse-maths-0580.yaml` | `ingest_syllabus_json.py` |
| `a2-chemistry-9701.pdf` | `a2-chemistry-9701.yaml` | `ingest_syllabus.py --section a-level` |
| `igcse-chemistry-0620.pdf` | `igcse-chemistry-0620.yaml` | `ingest_syllabus.py --section single` |

A2 Biology 9700 has no source here — its taxonomy was generated before this
folder existed. Re-download the 9700 PDF from Cambridge if it ever needs
regenerating.

`a2-chemistry-9701.pdf` is the **whole** 9701 syllabus, both years. Topics 1–22
are the AS half and 23–37 the A Level half; `--section` picks which. The AS
taxonomy happens to come from the JSON instead, which is why the same syllabus
appears twice in the table above under two names.

## The three-file model

Nobody edits a taxonomy by hand. Each one is generated from two inputs:

```
sources/<subject>.json|pdf     what Cambridge publishes
        +                      topics, subtopics, learning outcomes
overlays/<subject>.yaml        what only you know
        =                      Drive folder names, filename labels, keywords
config/syllabi/<subject>.yaml  what the pipeline reads   ← GENERATED
```

Cambridge does not know your folders are called `1. Atomic Structure (Chapter 1)`,
that a filename should read `Subatomic Particles` rather than the full sentence,
or that your teacher says "AOD". That is the overlay's job, and the overlay is
the only file a human writes.

Edit a generated taxonomy directly and your change disappears the next time
anyone regenerates. The header of each one says so.

## Regenerating

```bash
# From JSON
python scripts/ingest_syllabus_json.py \
    --json    config/syllabi/sources/igcse-maths-0580.json \
    --overlay config/syllabi/overlays/igcse-maths-0580.yaml \
    --out     config/syllabi/igcse-maths-0580.yaml

# From PDF — --section picks the half
python scripts/ingest_syllabus.py \
    --pdf     config/syllabi/sources/a2-chemistry-9701.pdf \
    --overlay config/syllabi/overlays/a2-chemistry-9701.yaml \
    --out     config/syllabi/a2-chemistry-9701.yaml --section a-level
```

Run the tests afterwards. They load every taxonomy, so a malformed one fails
loudly rather than at 3am against a real recording.

## Adding a subject

1. Put the syllabus PDF or JSON here.
2. Write an overlay in `overlays/` — copy the closest existing one; the header
   comments explain every field and why it is shaped that way.
3. Run the matching ingester.
4. Add the subject to `config/settings.yaml`.

The overlay is the work. Expect an hour or two per subject: the folder names
have to match Drive exactly, and the `short_title` for each subtopic is what
every filename in that subject will read.
