# Paper — four-page IEEE conference paper (LaTeX)

`main.tex` is the paper, `references.bib` its bibliography, `figures/` the figures. It is a four-page
IEEE conference paper (LaTeX) structured along CRISP-DM.

## Compile on Overleaf

1. Overleaf → *New Project* → *Upload Project*, and upload the `paper/` folder as a zip
   (`main.tex`, `references.bib`, `figures/*.pdf`). The PNG files are not needed by the paper.
2. *Menu* → Compiler **pdfLaTeX**, main document `main.tex`. Overleaf runs BibTeX by itself.
3. `IEEEtran.cls` and `IEEEtran.bst` are part of TeX Live, so no template files need uploading.
   The other packages (`cite`, `amsmath`, `amssymb`, `graphicx`, `booktabs`, `url`) are standard.
4. Check that the PDF has four pages.

A compiled `main.pdf` is not included in this repository; compile `main.tex` as above.

## Figures

Every figure is drawn from the saved result JSONs; no model is refitted and no number is typed by
hand. `python scripts/make_figures.py` writes `figures/<name>.pdf` (vector, 3.5 in = one IEEE
column, used by the paper) and `figures/<name>.png` (200 dpi, for quick viewing). The drawing code
is in `readmission/figures.py`.

| Figure file | Function in `readmission/figures.py` | Reads (`outputs/approved/`) | Numbers in report | In the paper |
| :--- | :--- | :--- | :--- | :--- |
| `fairness.pdf` | `fig_fairness` | `fairness.json` (+ overall rates from `summary.json`) | `approved_fairness.md` §2–§3 | Fig. 1 |
| `sources.pdf` | `fig_sources` | `sources.json` | `approved_sources.md` §1–§2 | Fig. 2 |
| `mitigation.pdf` | `fig_mitigation` | `mitigation.json` | `approved_mitigation.md` §1–§3, §5.1 | Fig. 3 |
| `models.pdf` | `fig_models` | `summary.json` | `approved_main.md` §1–§2, §6 | not used (Table I gives the same numbers) |
| `ordering.pdf` | `fig_ordering` | `ordering.json` | `approved_ordering.md` §2, §4 | not used (numbers in §III-D text) |
| `complexity.pdf` | `fig_complexity` | `summary.json` | `approved_main.md` §4 | not used in the paper |

## The `% src:` convention

After every sentence, caption or table row that carries a result number, `main.tex` has a LaTeX
comment naming where the number comes from, for example

```latex
Age band leads: its EO difference is 0.257 [0.192, 0.320] against a null mean of 0.030
(permutation $p = 0.005$; Fig.~\ref{fig:fairness}). % src: approved_fairness.md §2
```

* `approved_<name>.md §n` means `outputs/reports/approved_<name>.md`, section n ("header" = the
  paragraph above §1). These reports and `outputs/approved/*.json` are the only sources of result
  numbers.
* Design facts (dataset size before exclusions, missingness, grids, tree count) are cited to
  `docs/APPROVED_PROPOSAL.md` or to a code line (`readmission/<module>.py:<line>`).
* The two literature numbers (ROC-AUC 0.669 and balanced accuracy 0.622 of a patient-grouped
  model, the latter matching its authors' estimated ceiling) are cited to the paper itself
  (arXiv:2609.01909, "Cohort Results" and Table 1).

The comments do not appear in the PDF. They are there for the oral exam and for checking: to
verify a number, open the named report section and search for it.

## Additions and wording rules the paper follows

* Additions beyond the approved text are marked with a dagger (†) in the text and in Table II.
* The post-processors that close the age gap are described as levelling down, never as a fix.
* Test ordering is written as "result recorded" (the dataset's proxy for "ordered"); the glucose
  race odds ratios are "largely, not wholly" a recording pattern.
* No person, hospital, city or institution names appear except the university, the course and the
  authors of cited works.

