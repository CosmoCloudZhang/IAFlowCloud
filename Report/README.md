# IAFlow science report

The manuscript source is `main.tex`, with sections under `sections/`,
AASTeX-style bibitems in `Reference.bib`, and supplied PDFs under `figures/`.
`main.pdf` is the compiled report. Contributions and assistance disclosures
are in `sections/Appendix.tex`.

## Build

From this directory:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error main.tex
```

The report builds using the supplied sources and figure PDFs. It needs no
analysis scripts, numerical result registers, training data or weights.

## Format and figure sources

The [current Physics submission guide](https://www.yau-science-awards.org/competitioncategory/physics-guide.html)
requests a cover followed by the title, author, abstract, keywords, contents
and body. The abstract page retains the title and author and has no date.

The four AE diagnostic figures are unmodified copies of the existing PDFs
under `Figure/NLA/AE/` in the full project. They show the August six-coordinate
checkpoint; Sections 5.3 and 5.4 identify it separately from the September
comparison results. `figures/README.md` records the source filenames,
published-image credits and assistance provenance.

## Review status

This is a review draft. The supervisor identity, actual division of work and
complete assistance disclosure still require factual completion by the
student and supervisor. The conclusions concern reconstruction of a sampled
synthetic response family using single-seed validation results. Final-test
evaluation and survey likelihood validation remain future work.

Bibliographic checks and development records remain outside this report
directory in `../Document/Report_Development_Archive/2026-09-14/` in the full
project. Three ADS retrieval attempts per reference failed during the earlier
audit, so canonical ADS bibcodes remain unverified. The records and actual
assistance history remain available for any required supporting disclosure;
removing development folders from the manuscript package does not change
authorship or the competition's
[disclosure requirements](https://yau-science-awards.org/bulletin/show-102055.html).
