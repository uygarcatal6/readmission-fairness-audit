# Slides — 10-minute exam talk

A 10-minute talk. This deck has 10 talk slides plus one backup
slide (reproducibility), 16:9.

| File | What it is |
| :--- | :--- |
| `build_slides.py` | builds the deck with python-pptx; every number is read from `outputs/approved/*.json` |
| `readmission_fairness_talk.pptx` | the deck, with speaker notes on every slide |
| `readmission_fairness_talk.pdf` | PDF export (made by PowerPoint, see below) |
| `img/*.png` | the five figures of `readmission/figures.py`, adapted to the slides (see below) and rendered at 300 dpi |

## Rebuild

```bash
python slides/build_slides.py          # figures -> slides/img/, deck -> slides/readmission_fairness_talk.pptx
python slides/build_slides.py --pdf    # also the PDF, via PowerPoint (Windows + pywin32 only)
```

Needs python-pptx (tested with 1.0.2), matplotlib, Pillow and pytest (slide 11 counts the tests
pytest collects); `--pdf` needs PowerPoint and pywin32. No model is fitted and nothing under
`readmission/`, `outputs/` or `paper/` is written. The script prints the speaker-note word count
of each slide. The deck's document properties (author, title) are set by the script, not
inherited from the python-pptx template.

**Figures on the slides.** The paper figures are drawn for a 3.5-inch column. `build_slides.py`
adapts them for the slides only (`_for_slide`; `readmission/figures.py` and `paper/figures/` are
unchanged): each is resized to its box on the slide, text is enlarged ×1.35 and markers ×1.4
(legends about 14 pt, tick labels about 16 pt on the slide), numeric axes get at most three bins,
legends move outside the axes, and slide 7 shows only the right panel of the mitigation figure
(TPR by age band); the left panel's values are the bullets of that slide. The data drawn are the
same.

## Where the numbers come from

No number is typed by hand. `facts()` in `build_slides.py` reads each one from the five result
JSONs (the files behind the paper and the notebook) and formats it; small counts said in words
("one flag in four", "five approved models", "four mitigations") are computed too. `assert`s
stop the build if a sentence on a slide would no longer hold, e.g.: no approved model × rule
pair and no extension passes the gate, and no mitigation does; p is the minimum for the number
of shuffles; sex and race are within chance (p > 0.05); Kamiran–Calders widens the age gap (CI
above 0); balancing race cells cuts the age gap more than balancing age cells; neither
post-processor raises any age band's TPR against its twin, and ThresholdOptimizer lowers every
band; removing prior use moves the gap most and below the whole random-reference range;
ranking at the overall FPR is more than half of the age gap; the Hispanic glucose odds ratio
with the source indicator has a CI containing 1 and the African American one a CI below 1.
The source line at the bottom of each result slide names the report section that prints the
same numbers.

**† marks.** As in the paper (`\add`), additions beyond the approved proposal are marked † on
slides 4–8 (course extensions, permutation null, same-AUC-loss random reference, within-group
ranking, EO post-processing at capacity, the post hoc admission-type indicator), and the source
line of those slides explains the mark.

| Slide | Message | JSON | Report section |
| :---: | :--- | :--- | :--- |
| 1 | Title | — | — |
| 2 | Who gets the few follow-up slots: capacity rule and usability gate | `summary` | `APPROVED_PROPOSAL.md` §2; `approved_main.md` header |
| 3 | Data and CRISP-DM pipeline, validated patient by patient | `summary`, `fairness`, `sources`, `ordering` | `approved_main.md` header; `approved_fairness.md` §1 |
| 4 | Models rank better than chance, but none passes the gate (0 of 34 approved model × rule pairs) | `summary` | `approved_main.md` §1–§3 |
| 5 | Errors differ by age, not clearly by sex or race | `fairness` | `approved_fairness.md` §2, §3.2, §4 |
| 6 | The age gap comes mainly from ranking young patients better (TPR at the overall FPR) | `sources` | `approved_sources.md` §1–§5 |
| 7 | The mitigations that close the gap do so by levelling down | `mitigation` | `approved_mitigation.md` §1–§3, §5.1, §7 |
| 8 | Glucose race gaps are largely, not wholly, a recording pattern | `ordering` | `approved_ordering.md` §2, §4 |
| 9 | What a hospital should do with this model | `summary`, `fairness` | paper, Conclusions |
| 10 | Limitations and next steps, take-home message | — | paper, Conclusions |
| 11 | Backup: how to reproduce every number | test count from `pytest --collect-only tests/test_readmission_*.py` | `README.md` §Run |

Report files are `outputs/reports/approved_<name>.md`. The patient count (slide 3) is read from
`ordering.json` and the number of readmitted encounters from the group counts in
`fairness.json`; the counts of the raw file (before exclusions) are not in the JSONs and are
not on the slides.

## Speaker notes

Plain English, meant to be said aloud. Word counts as printed by the build: 1,188 words on
slides 1–10 (98–128 per slide) plus 131 for the backup slide. At an exam pace of 120–130 words
per minute that is 9:10–9:55, within the 10-minute limit.

The talk makes four points:
the usability gate fails (slides 1, 4), the two post-processors close the age gap only by
levelling down (slides 1, 7), the race differences in recorded glucose results are largely, not
wholly, a recording pattern (slides 1, 8), and the method changed, with one post hoc
sensitivity fit (slides 8, 10).
