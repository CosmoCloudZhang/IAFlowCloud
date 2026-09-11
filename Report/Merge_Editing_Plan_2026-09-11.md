# Report merge: scientific and editorial plan

Updated 11 September 2026 • Authorised merge plan and implementation record

**Recommended approach:** retain the project draft as the base, incorporate the useful explanatory additions from Downloads Sections 2.5–2.6, and integrate the six supplied figures with captions and display settings matched to their actual contents. Keep the project introduction, title structure, subtitle, headings, and bibliography format, with the terminology and notation updates below. Correct problems shared by both drafts rather than choosing between two inaccurate statements.

This is a selective merge of Chapters 1–2 and their supporting figures, with a report-wide check of equation numbering and terminology. Downloads also contains substantial changes to Chapters 3–7. Those form a separate, interdependent revision and remain outside this pass.

## 0. Additional instructions incorporated before implementation

| Requested change | Implementation decision |
|---|---|
| Survey references immediately after survey names | Attach each Stage III/IV survey's own citation to its name on introduction, and place release-specific citations beside the named analysis. Avoid a combined citation list after a multi-survey sentence |
| Consistent weak gravitational lensing terminology | Expand both “weak lensing” and adjectival “weak-lensing” throughout the report's authored prose, title, headings, and captions. Preserve published reference titles, filenames, and citation keys |
| A clearer noise symbol | Use `n_e` for measurement noise in ellipticity. This avoids both the model's redshift parameter eta and the common use of epsilon for ellipticity. Explain that random intrinsic shapes and measurement noise are distinct |
| Number all equations | Use numbered environments for every displayed equation, with continuous automatic numbering and descriptive labels. Existing numbered equations in later chapters retain their content and references. Inline expressions remain inline; no manual number resets |
| Upright correlation labels | Use `\mathrm{GG}`, `\mathrm{GI}`, `\mathrm{IG}`, and `\mathrm{II}` in equations, running text, captions, and schematic labels |
| Observed/intrinsic shape notation | Use `e^{\mathrm{O}}` and `e^{\mathrm{I}}` consistently; use the same notation in the pair correlation and the IA schematic |
| Define ellipticity | Add a short geometric definition before the observed-shape decomposition in Section 2.5, using image semiaxes and orientation. Explain the two components and 180-degree symmetry; identify this as one common convention |

For an approximately elliptical image, use the modulus `(a_img-b_img)/(a_img+b_img)` and components proportional to `cos(2 phi)` and `sin(2 phi)`. The image subscripts distinguish the semiaxes from the cosmological scale factor. A round image has zero ellipticity. Real measurements account for image blurring, pixel noise, and estimator response. The subsequent additive relation is a schematic separation of physical contributions, not the exact transformation law of an arbitrary galaxy image. This definition follows the epsilon-type ellipticity convention in [Bartelmann and Schneider, ellipticity definitions](https://arxiv.org/abs/astro-ph/9912508), denoted by e in this report; no new bibliography entry is needed.

The principal prose revision remains confined to Sections 1–2. Outside them, the authorised terminology and equation-numbering checks do not import the incoming methods, results, or appendices. The main title retains the approved wording with “Weak-Lensing” expanded to “Weak Gravitational Lensing”.

## 1. Pre-merge sources, inspection, and build status

The canonical source is `/Users/s2227120/IAFlowCloud/Report/`. The incoming source is `/Users/s2227120/Downloads/Report/`. In the decisions below, **project** and **Downloads** refer to those exact trees, not their previously compiled PDFs or an earlier Git revision.

I compared their LaTeX, bibliography, guide, scripts, and figure inventories; read the first two chapters and the incoming figure-generation script; inspected all six background PDFs separately and on the freshly compiled project pages; and checked the primary sources for the published panels. I also used the existing whole-project assessment and refreshed the model parameter list and surface configuration. This was not a new training run or a new evaluation of the test set.

| Audit item | Verified result | Consequence for the merge |
|---|---|---|
| Project build | Fresh temporary build succeeds: 23 PDF pages | The copied images are already found and included |
| Downloads build | Fresh temporary build succeeds: 26 PDF pages | Compilation success does not establish that its text or layout is preferable |
| Final build logs | Neither build reports LaTeX warnings, undefined references/citations, or overfull/underfull boxes | Remaining problems are scientific, editorial, and visual |
| Six background PDFs | Every project file has the same SHA-256 hash as its Downloads counterpart | No second copy or filename change is needed |
| Four existing result PDFs | The normally named files also match across both trees | No result-figure replacement is required |
| Four project files named `… copy.pdf` | Each is byte-identical to its normally named counterpart | They are redundant; leave them unused during this merge |
| Chapters 3–7 | All differ between the trees; Chapter 8 is identical | Do not copy the complete incoming `sections/` directory |
| Main document | Title/preamble differ; the diff contains no abstract change | Keep project `main.tex` as the base |
| Bibliography | Downloads restores visible ADS links and omits the project’s full paper titles | Preserve project `Reference.bib` |

Diagnostic copies and build logs are under `/private/tmp/iaflow-report-merge-kxtnrscy/`, with `project/`, `downloads/`, per-chapter diffs, rendered pages, and `inventory.json`. Neither original report tree was compiled in place. The inspection did not modify either manuscript tree. Pre-existing user edits and the deleted review PDF were preserved; a separate snapshot was taken before the authorised implementation.

The page counts and file comparisons above describe the pre-merge snapshots, not the completed revision. The final implementation record below gives the delivered build status. They include the title page. The larger Downloads count also reflects later-chapter changes; it cannot be attributed solely to Chapter 2.

## 2. Title, headings, and scientific scope to preserve

Keep the project title's structure, including its recently added first word, with the requested terminology expansion:

> Learning Compact Representations of Intrinsic-Alignment Responses for Weak Gravitational Lensing Cosmology

Keep the subtitle:

> A flexible scale-redshift model with linear and nonlinear reconstruction

| Location | Heading to retain |
|---|---|
| 1 | Introduction |
| 2 | Cosmological background and intrinsic alignment |
| 2.1 | Cosmic expansion and the standard cosmological model |
| 2.2 | Observational constraints on dark energy |
| 2.3 | Weak gravitational lensing and matter clustering |
| 2.4 | Cosmic-shear surveys and the S8 tension |
| 2.5 | Intrinsic alignment and galaxy-shape correlations |
| 2.6 | Intrinsic-alignment models and dimensionality reduction |

Retain the existing LaTeX math and PDF-bookmark handling for S8. Preserve all section and figure labels so cross-references continue to resolve. Keep Introduction and Background separate. Apply the notation policy in Section 0 to the examples and captions below.

The scientific object is the positive, synthetic response surface `A_Theta(k,z)`, with its logarithm used in the reconstruction experiment. It is not the full signed IA amplitude, the matter power spectrum, or a TATT emulator. The 13-parameter family is already a compact analytic generator of the 3,131 grid values; the experiment asks whether an even shorter approximate representation captures its variations accurately. Neither the title nor the background should imply that two latent coordinates are two fundamental physical parameters or that cosmological inference has already been validated.

The writing should use precise terminology in direct, readable sentences. Define unfamiliar terms through their physical meaning. Preserve “latent coordinates,” “response surface,” “parameter degeneracy,” and “reconstruction error”; avoid repeatedly replacing them with “numbers,” “patterns,” or “getting mixed up.” Scientific maturity comes from the reasoning and the limits of the evidence, not from journal-style density or conversational headings.

## 3. Chapter 1: retain the project text

**Decision: keep the current project introduction essentially intact.** Downloads offers an earlier, more conversational formulation rather than a substantive new contribution to this chapter.

| Part of the introduction | Decision and reason |
|---|---|
| Observable and purpose | Retain the direct weak gravitational lensing opening. It identifies what is measured and why it matters without a rhetorical question |
| Intrinsic alignment | Retain the explicit physical explanation and its consequence for inferred cosmology |
| Modelling difficulty | Keep the connection between flexibility, similar predictions, and the reconstruction question |
| Research object | Keep the positive synthetic family, 13 parameters, grid size, and definition of latent dimension |
| Comparison | Keep the convolutional autoencoder description and matched PCA/AE comparison on the same samples |
| Contribution | Keep the distinction between established techniques and the project’s response family, controlled dataset, and evaluation |
| Result preview | Keep the short qualitative preview and its limitation; do not import another numerical results paragraph |

The present division of labour remains sensible: the abstract gives a compact account including numerical outcomes; the introduction establishes the motivation, design, and significance of the comparison. Repeating the full score table or the complete background in Chapter 1 would weaken that division. This pass should not update the historical pilot metrics or assert that newer runs have replaced them.

## 4. Chapter 2: paragraph-level merge decisions

The comparisons and suggested replacements in Sections 4–6 describe the pre-merge snapshots and the agreed editing decisions. The manuscript and artwork have since been revised; Section 12 records the delivered state.

### 4.1 Section 2.1 — cosmology and the Planck map

Retain the project’s description of cosmic expansion, redshift, the concordance model, and the unknown nature of the dark components. These explain the ideas accurately without relying on prior astronomy knowledge.

**Adopt the incoming caption’s map-only interpretation.** The supplied image contains one all-sky CMB map. It contains neither a cosmic-composition chart nor an expansion/redshift strip. The project caption currently claims that a composition panel is present, and both drafts retain preparation notes for panels that were not delivered.

Replace the lead-in near project lines 75–76 with wording such as:

> Figure 1 shows the temperature variations measured by Planck. Their statistical properties help constrain the cosmological model used to describe the Universe’s contents and evolution.

Use a normal `Figure~\ref{fig:background-planck}` cross-reference in the manuscript. Keep the approximate present-day composition in the prose with the Planck cosmological-parameters citation. It does not have to appear in the map itself.

The immediate merge should accept the map as the subsection’s figure. A student-made composition/expansion strip can remain an optional later enhancement, clearly labelled as unfinished in the guide. It must not be described as an existing panel.

### 4.2 Section 2.2 — DESI and dark energy

Retain the project’s BAO explanation, definitions, equation-of-state progression, and carefully qualified motivation for complementary weak-lensing evidence. Keep the two supplied published contour panels.

The displayed horizontal axis is **H0 rd in units of 100 km s^-1**. It is not incorrectly labelled: the value near 102 is expressed in those scaled units. Explain this convention briefly in the caption or adjacent sentence; do not change only the axis label or multiply the plotted values. The source captions identify the model and dataset combinations. [DESI DR2 BAO paper, Figures 8 and 11](https://arxiv.org/html/2503.14738v3).

**Correct the sentence shared by both drafts about the 2026 hydrogen-absorption analysis.** Its joint constraints still favour evolving dark energy at 2.7 sigma with CMB and 3.2 sigma when supernovae are included. Calling the paper simply “consistent with LambdaCDM” misrepresents that result. [DESI DR2 Results IV, abstract](https://arxiv.org/abs/2607.27410v3).

Suggested replacement, without adding another list of significance values:

> A later DESI analysis adds information from absorption by intergalactic hydrogen. Its joint constraints also retain a preference for evolving dark energy, while showing that the strength of the evidence depends on the observations and modelling assumptions.

Attach the existing `2026arXiv260727410D` citation. Keep the distinction between an indication and a confirmed discovery, and between a complementary probe and an automatically independent combined analysis. The current BAO figure need not be replaced by a panel from the later paper.

### 4.3 Section 2.3 — lensing, density, and power spectra

Retain the project’s physical-to-statistical progression: light deflection; convergence and shear; the density field; density contrast; spatial wavelength and wavenumber; matter power spectrum; and the projected pair-correlation measurement.

This remains the right place to introduce `delta(x,z)`, `P_delta(k,z)`, and `k = 2 pi / lambda_spatial`. Moving those definitions to Section 2.1 would introduce formalism before its purpose is clear. Preserve the distinction between a spatial wavelength and the photon wavelengths used in the redshift equation. Keep the explicit comoving-length convention and the statement that a power spectrum describes two-point statistics rather than the full arrangement of matter.

The supplied three-part diagram broadly follows this structure, but needs targeted artwork corrections described in Section 5.3 below. Importing the incoming caption alone would not fix those issues. Preserve the project’s descriptive caption opening and its citation to `2001PhR...340..291B`.

Keep the broad measurement workflow here. The additional Section 2.5 paragraph should focus on why an apparent shape does not isolate the gravitational contribution, rather than repeat the entire calibration workflow.

### 4.4 Section 2.4 — survey evidence and the S8 tension

Retain the project’s definition of S8, explanation of sigma8, distinction between Stage III results and Stage IV prospects, and qualified description of reduced tension. Keep the explanation that alternative DES Y6 IA analyses use the same observations.

**Adopt and refine the Downloads caption describing the actual comparison.** The image shows KiDS-Legacy E_n, DES Y3 xi-plus/minus Hybrid, HSC Y3 xi-plus/minus, and Planck-Legacy. It contains no HSC Y1 contour, DES Y6 contour, earlier KiDS release, or Stage IV prospects panel. [Wright et al., Figure 14](https://arxiv.org/html/2503.19441v2).

Replace project lines 238–240, which claim the figure shows successive analyses, with a direct description of the published cross-survey comparison. The following sentence can state that the DES Y6 values discussed in the text are more recent than this figure. Retain the historical KiDS discussion and Stage IV prospects as prose.

The supplied figure therefore satisfies “one figure in this subsection,” but **does not fully satisfy the earlier request for a visual comparison across releases**. My recommendation is to use it in this merge, and record a release-history companion as a later figure task. Do not pretend that the existing panel already performs that comparison.

For a future companion, compare a specified earlier and later analysis within each survey using released chains or faithfully attributed panels. State which statistic, IA treatment, and other analysis choices changed. Do not create a DES Y6 contour from its quoted one-dimensional S8 error bar or relabel the DES Y3 contour as Y6. This additional data task should not silently expand the present text merge.

The citation to the published comparison should be Wright et al. The standalone HSC Y1/DES Y3 citations currently attached to the inaccurate figure sentence should only remain there if that sentence is rewritten to make a supported historical claim. Keeping their bibliography entries is harmless and avoids an unnecessary bibliography cleanup.

### 4.5 Section 2.5 — measured shapes and GG/GI/II

This is one of the strongest incoming additions. Its explanation that a telescope measures an image, and that the observable is a sum of correlated contributions, helps a reader understand why IA is a modelling problem.

Retain the project’s tidal-field introduction, annotated ellipticity relation, two-component explanation, and reduced-shear qualification. Adapt the incoming material as follows:

| Incoming addition | Recommended treatment |
|---|---|
| Pixelated, PSF-convolved images | Keep the physical point, but explain image blurring in ordinary language; avoid introducing an undefined PSF abbreviation |
| A single image “cannot provide a useful shear measurement” | Replace this absolute claim with “an individual galaxy shape is a noisy estimate of shear” |
| Expansion of `(G+I)_i(G+I)_j` | Explain the expansion in words, or explicitly define G and I before using this shorthand |
| Separate explanations of GG, II, and GI/IG | Adopt three compact parallel explanations following the equation |
| The survey measures their sum | Adopt; this makes the inverse problem clear |
| Random and correlated errors | Adopt the distinction: uncorrelated noise adds variance; coherent residuals can bias correlations |

A concise measurement bridge could read:

> Even after correcting image blurring and measurement bias, an individual galaxy shape is a noisy estimate of shear because its intrinsic ellipticity is usually larger than the lensing distortion. Correlations across many galaxy pairs reveal the combined signal, including any correlations of intrinsic shapes.

Before the four-term relation, state that the expression concerns distinct-galaxy signal correlations after allowing for uncorrelated noise. Avoid the unexplained statement “ignoring terms whose expectation is zero”: it hides the assumption that lets those terms be omitted.

Keep the full `GG + GI + IG + II` expression and explain that the two cross-orderings are often grouped as GI. Preserve the useful incoming explanation that physically separated sources can contribute to GI. Add one sentence after the term definitions:

> Observations constrain their sum; the different angular and redshift dependences help a model distinguish the contributions.

Use the existing weak-lensing and IA-review citations at the relevant explanations. Do not repeat a separate, lengthy disclaimer after every term.

### 4.6 Section 2.6 — models and the reconstruction question

The expanded Downloads explanation of LA/NLA and TATT is worth incorporating. It makes the mechanism behind the model hierarchy clearer than a simple “few parameters versus many parameters” contrast.

Use this sequence:

1. Introduce the leading tidal-alignment response and explain that NLA uses a nonlinear matter power spectrum in the correlation prediction. Retain the point that this is not a complete theory of nonlinear galaxy formation.
2. Explain that TATT adds further tidal responses, including quadratic terms and effects associated with how galaxies sample the density field. Keep “usually” when comparing parameter counts.
3. Explain suitability in terms of galaxy population, scales, precision, and adopted parameter probabilities. Flexibility has potential benefits and costs; neither model is universally best.
4. Describe the risk of cosmological bias from an inadequate model and the difficulty of constraining similar predictions from flexible models. Say computational cost **can** increase; more parameters do not by themselves establish a measured slowdown.
5. Retain the project’s separate explanation of line-of-sight projection and marginalization. These are distinct physical and statistical operations; changing coordinates does not automatically resolve either issue.
6. Connect these challenges to the specific positive response family, the matched PCA/AE reconstruction test, and what would still be needed for cosmological use.

Retain `2007NJPh....9..444B`, `2019PhRvD.100j3506B`, and the existing DES Y6 citation where they support the relevant claims.

A suitable closing bridge is:

> We therefore test whether the variations of a flexible, NLA-inspired response family can be reconstructed with a small number of latent coordinates. The 13 original parameters define the family; the compressed representation approximates its surfaces. Its usefulness for cosmology would require additional tests of errors in lensing predictions and of the probabilities assigned to the latent coordinates.

This can replace overlapping closing sentences rather than being appended to both versions. Keep the cross-reference to Section 3. Avoid an untested promise that the compression already reduces inference time or removes degeneracies. The current project scope and the incoming “future cosmological use” box are compatible when expressed this way.

## 5. Figure-by-figure integration and correction plan

All six filenames already match `\backgroundfigure` calls and `\graphicspath{{figures/}}`. No new image-discovery mechanism is required. The following are caption/artwork/layout tasks, not missing-file fixes.

### 5.1 Planck — `background_planck_cosmology.pdf`

**Keep the supplied map.** Its current display is approximately 15.36 × 7.76 cm and it renders clearly. It has no quantitative colour bar; describe the colours qualitatively rather than inventing a temperature scale.

Proposed caption:

> Planck’s all-sky map of CMB temperature anisotropies. The colours show small temperature variations in light that began travelling freely about 380,000 years after the Big Bang. This is an early-Universe observation, rather than a map of present-day dark matter. Image credit: ESA/Planck Collaboration.

Attach `\citep{2020A&A...641A...1P}`. Keep the ESA source and credit in the figure guide. The official image page identifies the map as based on the Planck Legacy release. [ESA image source](https://www.esa.int/ESA_Multimedia/Images/2018/07/Planck_s_view_of_the_cosmic_microwave_background).

Update the nearby source comments and fallback description to “Planck CMB temperature map.” Remove claims of existing composition/expansion panels from the caption and completion checklist.

### 5.2 DESI — `background_desi_dr2.pdf`

**Keep the measured contours and their legends.** The current display is approximately 15.36 × 7.53 cm. Tick labels are small, so preserve at least this width and review at printed size.

Retain the project caption with three refinements: identify the DR1/DR2/CMB comparison on the left; explain the scaled H0 rd units; and say that the dashed-line intersection on the right marks the cosmological-constant case. Keep the stated 68%/95% contours and the source panel numbers. The source figure has dashed reference lines rather than a separate point marker. [DESI source panels](https://arxiv.org/html/2503.14738v3).

No rescaling or contour reconstruction is needed. If later revising the assembly, retain the original SVG vectors where practical: the incoming generator rasterizes those panels before embedding them in the PDF. That is not a missing-data problem, but retaining vectors would preserve sharp labels when resized.

### 5.3 Lensing and density — `background_lensing_density.pdf`

**Keep the three-part concept; revise the drawing and typography before treating it as finished.** Its current display is approximately 14.0 × 8.4 cm because the shared height cap reduces its width. Extracted figure text includes sizes of about 5.0, 6.0, and 6.9 pt.

Required corrections:

- Redraw the top ray geometry. The current paths cross near the intervening mass and then turn again before reaching the observer; this is an ambiguous illustration of weak lensing. Show a weakly deflected bundle with a clear source–lens–observer ordering and no intermediate crossing/caustic. A simple thin-lens schematic is sufficient; label exaggerated deflection.
- Replace `kappa`, `gamma1`, `gamma2`, `P_delta(k,z)`, and `2 pi` with properly typeset symbols matching the text. Define the fixed axes if illustrating gamma1 and gamma2 separately. A single shear example is acceptable if the second adds clutter.
- Make the two density strips cover the same indicated distance interval. If arrows denote a wavelength, place their endpoints one complete cycle apart. The current arrows are illustrative lengths rather than unambiguous full periods.
- Reflow the power-spectrum description so it stays inside its box. Reduce redundant text and enlarge labels instead of shrinking the entire figure.

Retain the current caption’s description of geometry, image distortion, and spatial scale. After revision, ensure it identifies all three parts and states that the distortions and density waves are schematic. Keep the measurement/reduced-shear qualifications in the surrounding prose.

### 5.4 Survey comparison — `background_s8_surveys.pdf`

**Keep the supplied published panel; enlarge it and correct its interpretation.** The current shared cap produces a figure only about 7.66 cm wide, with a plotted square about 7.16 cm wide. A starting target of 0.70 times the text width gives approximately 11.2 × 12.3 cm for the complete image.

Proposed caption:

> Published constraints in the Omega_m–S8 plane from KiDS-Legacy, DES Year 3, HSC Year 3, and Planck-Legacy. The legend identifies the analysis statistic used for each survey. This comparison predates the DES Year 6 results discussed in the text. Adapted from the left panel of Figure 14 in Wright et al. (2025).

Use LaTeX symbols and `\citet{2025A&A...703A.158W}`. Preserve the source attribution and recorded CC BY 4.0 credit. Do not introduce precise probability levels beyond those established for the source panel. The small “Smoothing Kernel” inset is part of the published figure, not an additional survey measurement; explain it briefly in the guide if retained. [Source comparison](https://arxiv.org/html/2503.19441v2).

Keep the source’s distinct statistics visible. Do not describe this as a controlled comparison with identical modelling assumptions across all surveys. Put the deferred release-history task in the guide, separate from the delivered figure.

### 5.5 Intrinsic alignment — `background_intrinsic_alignment.pdf`

**Adopt the supplied three-mechanism layout with modest corrections.** The observed-shape relation and explicit GI/IG grouping are useful. At the current roughly 14.0 × 8.4 cm display, some explanatory labels are about 5.6 pt.

- Enlarge the complete figure to nearly the full text width and enlarge internal explanatory labels.
- Label foreground/background or increasing distance in the GI geometry, so the roles of I and G are immediately clear.
- Use anisotropic tidal-field arrows, or a clearly labelled environment symbol, in the II panel. Four similar inward arrows can look like isotropic compression and do not explain the preferred orientation.
- Consider replacing “GG — two lensed images” with “GG — correlated gravitational shears” to identify the statistical term more precisely.
- Keep the weak-distortion qualification and the explicit full sum. Panel areas must not imply relative measured amplitudes.

Retain the project’s formal caption opening, “Contributions to observed galaxy-shape correlations,” and its description of the mechanisms. Use the incoming provenance information as discussed in Section 7. There is no need to replace the formal opening with “Why observed shapes contain more than lensing.”

### 5.6 Model comparison — `background_model_tradeoff.pdf`

**Adopt the balanced model comparison and experiment/future-work separation; simplify and re-typeset the interior.** The figure already occupies approximately 15.36 × 8.34 cm. Removing the height cap will scarcely enlarge it. Its body text is around 6.2 pt, with some labels around 5.3 pt, so the artwork itself must change.

Recommended revisions:

- Replace the question “Can a flexible response family be described with fewer numbers at useful accuracy?” with a statement such as **“Compact representations of a flexible response family.”**
- Give each model branch at most three short points covering response flexibility, parameter constraints, and potential modelling/computational cost.
- Typeset `A_Theta(k,z)` consistently. The present literal `AΘ(k,z)` is small and difficult to distinguish.
- Make the workflow explicit: sampled parameters → positive surfaces → log-space representation and compression → two latent coordinates → reconstructed response → matched validation error. This can use two rows to allow readable labels.
- Retain a visually separate box for future lensing-error, latent-prior, and inference tests. It should remain clear that those are required next steps, not reported results.

Keep the project caption’s formal opening. Do not adopt the Downloads `\clearpage`/float-page-offset block: it puts the same small figure on a largely empty page without improving its internal text size. Prefer better artwork and controlled normal placement.

## 6. LaTeX display and placement changes

The current macro imposes both `width=0.96\linewidth` and `height=0.34\textheight`. This is the main reason the portrait survey plot and taller schematics shrink. Give the Section 2 macro an optional set of graphics settings while keeping its four content arguments and existing labels.

The selected macro interface is:

```latex
\newcommand{\backgroundfigure}[5][width=0.96\linewidth]{%
  \begin{figure}[!htbp]
    \centering
    \IfFileExists{figures/#2}{%
      \includegraphics[#1]{#2}%
    }{%
      \fbox{\parbox[c][3.0cm][c]{0.90\linewidth}{%
        \centering\small\textbf{Figure to be prepared}\par
        \medskip #3\par\medskip
        \texttt{\detokenize{#2}}}}
    }
    \caption{#4}
    \label{#5}
  \end{figure}%
}
```

This pattern illustrates the implemented macro interface; the current source also scopes float-placement settings within Section 2. Existing four-argument calls still work; the survey figure can use `[width=0.70\linewidth]`. Providing width alone preserves aspect ratio. If both dimensions are supplied for a particular image, include `keepaspectratio`.

| Figure | Starting setting for the revised build | Additional requirement |
|---|---|---|
| Planck | `width=0.96\linewidth` | Map-only caption |
| DESI | `width=0.96\linewidth` | Preserve complete legends and units |
| Lensing/density | `width=0.96\linewidth` | Correct geometry and enlarge internal text |
| Survey comparison | `width=0.70\linewidth` | Accept a taller figure; do not squeeze to 34% page height |
| Intrinsic alignment | `width=0.96\linewidth` | Enlarge explanatory labels |
| Model comparison | `width=0.96\linewidth` | Redesign text density; a larger page slot alone is insufficient |

Place figure calls close to their first substantive discussion and inspect the rendered reading order. The implementation combines subsection float barriers with relaxed placement fractions inside a local Section 2 group. Barriers alone produced nearly empty figure pages; adjusting the fractions let figures share pages with prose while keeping each figure before the next subsection. The final barrier also prevents background figures from drifting into Section 3. Later-section float settings remain unchanged.

Preserve `placeins`, `xurl`, the existing final Section 2 `\FloatBarrier`, and the project’s title wrapping. Keep the separate `\reportfigure` macro for later chapters unchanged. Do not make all report figures obey the new background sizing rule.

For original schematics, aim for approximately 8–10 pt body labels at final print size. At 15.36 cm width, that generally requires at least 24 px text in a 1,200 px-wide design and about 27 px text in a 1,400 px-wide design. These are starting values; verify the PDF output rather than relying solely on design-canvas font sizes. Published-panel labels should be reviewed independently because their source proportions differ.

## 7. Source files, credits, bibliography, and figure guide

The following source assets were selected for import. They are now present in the project, with the executable schematic renderer separated from the archived incoming assembly script:

| Incoming relative path | Purpose and implemented treatment |
|---|---|
| `figures/sources/planck_cmb_2018.jpg` | Retain the credited source map |
| `figures/sources/desi_dr2_figure8_left.svg` | Retain the exact published left panel |
| `figures/sources/desi_dr2_figure11.svg` | Retain the exact published dark-energy panel |
| `figures/sources/kids_legacy_figure14_omega_m_s8.svg` | Retain the published comparison panel |
| `figures/sources/README.md` | Keep source URLs, versions, panel identities, credits, and reuse information; add a record of assembly changes |
| `scripts/build_sourced_background_figures.py` | Archive the incoming Pillow/CairoSVG script as `figures/sources/Incoming_Background_Builder.py.txt`; supply a NumPy/Matplotlib renderer with per-figure selection for the three revised schematics |

The archived incoming script rebuilds all six PDFs. The current executable rebuilds only the three explanatory diagrams, selected by `--figures lensing ia models`, and writes their PDFs under `figures/` and editable SVGs under `figures/sources/schematics/`. It uses paths relative to itself. The three published map/contour PDFs remain byte-identical to the incoming files. The existing `sync_figures.sh` handles four result plots, not the six background figures, and need not be run for this merge.

The incoming schematics and their captions explicitly record GPT-5.6 Sol assistance. The source available here is a Python/SVG drawing script. Preserve that provenance accurately; do not relabel the figures as independently drawn by the student. A concise description such as “Schematic prepared with AI assistance; see figure-source record” could replace repeated large tool labels only if an equivalent, accurate source record is retained. The record should distinguish published observations, code-generated explanatory artwork, and the student’s actual revisions. Do not claim a student validation step until it has happened. This is a source-attribution recommendation, not a new determination of contest policy.

**Keep the current project bibliography intact.** Its AASTeX-style `\bibitem` entries are deliberately included as LaTeX, despite the `.bib` extension. The project already supplies author–year citations, full paper titles, journal details, DOI links, and nonprinting ADS comments. The incoming file would undo that agreed presentation. No conversion to BibTeX/Biber, replacement with abbreviated entries, or restoration of visible `[ADS]` links is needed.

Update `Section2_Figure_Guide.md` to reflect the delivered state:

- Replace the statement that all six slots are still visible drafting boxes.
- Record each actual image, source, caption status, and pending visual corrections.
- Update the Planck and survey layouts to match the files present.
- Replace the global height-cap advice with per-figure guidance.
- Keep unmade composition/history/prospects panels under a clearly separate optional-work heading.
- Record the actual source/generator locations and appropriate attribution.
- Retain the current citation workflow and project subsection titles.

Keep detailed drawing instructions in the Markdown guide. Short LaTeX comments should identify the figure and point to that guide. Captions should explain the science visible in the finished image, not instruct the student how to prepare it.

## 8. Ordered implementation plan

| Step | Files / action | Completion condition |
|---|---|---|
| 1. Preserve the baseline | Snapshot current manuscript and figures; retain all pre-existing edits | Comparison can distinguish this merge from earlier work |
| 2. Lock the retained material | Keep project title, headings, Section 1, preamble, bibliography, and Chapters 3–8 | No accidental reversion from copying whole files |
| 3. Merge Chapter 2 prose | Adapt 2.5/2.6 additions; correct the DESI statement; update Planck/S8 lead-ins | One coherent narrative with no unsupported panel descriptions |
| 4. Bring in editable assets | Selectively copy the source panels, source record, and generator | Each figure has a traceable editable/source route |
| 5. Correct original schematics | Fix the lensing geometry, II illustration, notation, internal labels, and model flow | Diagrams explain the intended mechanisms at readable size |
| 6. Configure figure display | Add per-figure sizing; retain normal float handling and final barrier | All six images display without distortion or clipping |
| 7. Align guide and captions | Use actual panel contents and credits; separate deferred extensions | Guide, LaTeX comments, and printed captions agree |
| 8. Build and inspect | Fresh complete build; review first two chapters and their transition to Section 3 | No missing images/citations, confusing placement, or tiny labels |
| 9. Review the final diff | Compare against the saved pre-merge tree | Only the intended text, figure support, and background artwork changed |

The scientific/caption corrections and readable schematic labels are required for this merge. Additional historical contour panels, new composition diagrams, and a broader report redesign are separate enhancements. This ordering keeps the incoming work usable without turning every earlier figure idea into a prerequisite.

The text and LaTeX changes can be prepared locally. Put the detailed artwork corrections in the student’s guide so she can revise the editable diagrams and explain the finished figures herself. The current supplied schematics can be included in a working merged draft while those corrections remain explicitly pending; they should not be labelled submission-ready before that review.

## 9. Acceptance checks for the merged draft

1. **Title and structure:** preserve the approved title and heading structure, with the requested weak gravitational lensing terminology. Introduction and Background remain separate.
2. **Scope:** preserve methods/results/appendix content and numerical values, allowing the requested report-wide terminology and automatic equation-numbering changes. Preserve the four existing result figures and bibliography formatting. Pagination may change.
3. **Scientific meaning:** the dark-energy discussion reflects dataset dependence; Stage III measurements and Stage IV prospects remain distinct; the response surface remains distinct from the matter power spectrum and full IA physics.
4. **Figures:** each subsection has its intended PDF; filenames and labels match; captions mention only panels present; provenance is retained.
5. **Geometry and notation:** the lensing ray diagram has a clear physical interpretation; tidal alignment is not represented as unexplained isotropic compression; k and wavelength annotations are consistent.
6. **Readability:** review the rendered pages at intended print size, especially the S8 comparison and the labels in Figures 3, 5, and 6. No diagram text extends beyond a box or relies on extreme zoom.
7. **Narrative flow:** the new measurement/model explanations replace overlapping material; they do not make the introduction or background repeat the same claim several times.
8. **Build:** run `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` from a fresh complete report copy. Resolve undefined citations/references, missing assets, duplicate labels, and layout warnings before replacing the review PDF.
9. **Student understanding:** use the revised guide to check that the student can explain the observable, each figure’s source, the GG/GI/II distinction, and the reconstruction experiment’s actual contribution. Revise unfamiliar wording through explanation rather than deleting the scientific terminology.
10. **Notation and citations:** each survey introduction has its own adjacent reference; every displayed equation is numbered; all correlation labels are upright; observed and intrinsic ellipticity use O and I; measurement noise uses n_e. The ellipticity convention and the limits of the schematic decomposition are explicit.

## 10. Incoming later-chapter changes to preserve for a separate review

These differences are real and should not be lost, but are not approved into this Chapters 1–2 merge:

| Incoming material | Why it requires a coordinated later pass |
|---|---|
| Section 3 | Rewrites model/sampling explanations and moves detailed equations and bounds to an appendix |
| Section 4 | Reworks normalization/training/diagnostics discussion and redirects implementation details to appendix material |
| Sections 5–6 | Expands interpretation, limitations, and next steps; these claims need to remain tied to the reported run |
| Section 7 | Adds a substantial equations/configuration appendix and changes reference targets |
| Manual equation numbering | Incoming Section 3 uses `\setcounter{equation}{7}` and the appendix refers literally to “Equation (8)”; these should eventually become label-based references |

Moving only some of these files would break the intended equation/table/reference structure. A later pass should compare them as a package and reconcile the existing review’s pilot-run provenance and historical test-exposure findings before strengthening any results claims. The present plan deliberately avoids presenting those incoming statements as newly verified scientific evidence.

## 11. Agreed file scope

**Primary edit:** `Report/sections/Section2.tex`.

**Supporting edits/additions:** `Report/Section2_Figure_Guide.md`; the selected `Report/figures/sources/` assets and source record; `Report/scripts/build_sourced_background_figures.py`; and the three original background schematic PDFs if their drawing corrections are implemented. The Planck and published contour PDFs can remain byte-identical unless their assembly or print quality requires adjustment.

**Small consistency edits:** `Report/main.tex` and, if needed, other authored prose for the weak gravitational lensing terminology. Existing numbered equations in later chapters will renumber automatically when background equations are numbered.

**Expected to retain their scientific content:** `Report/sections/Section1.tex`, `Report/Reference.bib`, Chapters 3–8, the four result figures, and `Report/scripts/sync_figures.sh`. The bibliography and later-chapter source files need not change if the consistency audit finds no additional occurrences.

The plan above was updated before manuscript implementation. The final record below distinguishes completed edits from optional future work.


## 12. Completed merge and validation record

The authorised merge has been implemented. The delivered report is `Report/main.pdf`, compiled from the canonical source through a fresh, complete temporary copy. This record supersedes the pre-merge status in Section 1; it does not mark the optional future figure extensions or the later-chapter merge as completed.

| Item | Completed result |
|---|---|
| Title and introduction | Preserved the project title structure and subtitle; expanded “Weak-Lensing” in the title. Section 1 is byte-identical to the pre-edit snapshot because its terminology and scientific framing already met the agreed requirements |
| Background | Revised all six subsections for the progression from cosmology and expansion to observations, matter clustering, intrinsic alignment, and the reconstruction question. Incorporated useful incoming explanations with corrected physical claims |
| Survey citations | Individual Stage III and IV references follow their survey names; release-specific references accompany the corresponding named analyses |
| Shape notation | Added a stated image-ellipticity convention and orientation components. Used observed e^O, intrinsic e^I, measurement noise n_e, and upright GG/GI/IG/II throughout the relevant text and diagrams |
| Equation numbering | Verified PDF destinations numbered continuously from 1 to 21. The nine background equations have descriptive labels. Existing later equations retain their mathematical content and renumber automatically |
| Figure integration | All six background PDFs display. Captions describe the actual panels. The three published map/contour PDFs are unchanged; the three original schematics have revised geometry, typography, notation, and editable SVG sources |
| Layout | Revised figure sizing and placement. Local Section 2 float settings plus subsection barriers keep the figures before the following subsection without isolated, mostly empty figure pages. Later figure settings are unchanged |
| Bibliography | Preserved the existing 21 AASTeX-style entries and author–year citation configuration, including full titles, DOI links, and nonprinting ADS records. No new entry was necessary for the ellipticity definition |
| Preserved scientific material | Hash checks confirm that Section 1, Sections 3–8, Reference.bib, the four result figures, the three published background PDFs, and the result-figure synchronisation script match the pre-edit snapshot |
| Fresh PDF build | latexmk succeeds; 25 PDF pages including the cover. The final LaTeX log has no warnings, undefined references/citations, or overfull/underfull boxes |
| Source and diagram checks | No unnumbered display environments, suppressed equation numbers, manual equation resets, or abbreviated weak-lensing terminology remain in authored LaTeX. The revised diagram script passes the project Python-style audit and Ruff |

The fresh final build and visual-check files are retained under `/private/tmp/iaflow-report-edit-44olfp7x/final-build/` and its parent directory. Report manuscript edits were made against the existing working tree, not against a clean Git revision. The pre-existing deleted review PDF, redundant figure copies, and unrelated project edits were not reverted. No commit or model-training run was made.

### Student review still to complete

The separate `Section2_Figure_Guide.md` is the working handoff for the student. It records the exact published panels, their credits, the provenance of AI-assisted schematics, and questions she should be able to answer about each figure. It also provides the editable source locations and selective rebuild commands. Her scientific review and authorship contribution have not been assumed.

The optional composition/redshift strip and release-history companion plot remain future additions. The delivered S8 panel compares Stage III surveys and does not include DES Year 6 or Stage IV forecasts. Current release information and Stage IV prospects are explained in the prose. Later-chapter Downloads changes and the earlier review's evidence/provenance questions remain a separate revision, as agreed.
