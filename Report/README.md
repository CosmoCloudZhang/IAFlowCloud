# IAFlow science report

The manuscript source is `main.tex`, with sections under `sections/`,
references in `Reference.bib`, and included figures under `figures/`.
`main.pdf` is the compiled report.

Build from this directory with:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error main.tex
```

The existing figure PDFs are sufficient to build the manuscript. Figure
generators are in `scripts/`; run them only when their figures need updating.
The broad `sync_figures.sh` utility replaces selected report figures from
research outputs and is not required for an ordinary manuscript build.

Published-image sources, permissions, numerical provenance and schematic
assistance credits are retained in `figures/sources/README.md`.
Development plans, review guides, revision records and the pre-revision
manuscript are preserved outside the report directory in
`../Document/Report_Development_Archive/2026-09-13/`.
That archive records AI assistance with prose, figures and layout.

Sections 2--4 have been revised. Section 4 includes the PCA-assisted
autoencoder, with matching matrix-and-vector architecture diagrams for
both autoencoder methods. Regenerate these two diagrams with
`python scripts/build_section4_figures.py` from this directory.
The introduction, summary and cross-references follow the comparison at five
latent dimensions reported in Section 5.

The report's separate acknowledgements
and division-of-work placeholders still require factual completion by the
student and supervisor.
