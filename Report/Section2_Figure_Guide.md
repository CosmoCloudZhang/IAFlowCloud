# Section 2: student figure and reference guide

Prepared 10 September 2026 for the revision of Sections 1–2.

The six figures below should help a reader outside astronomy follow the argument. The student should assemble the figures, choose the final annotations, and be able to explain every panel. The Planck map and survey contours must come from the cited measurements; the explanatory diagrams can be her own drawings.

## How this guide fits into the report

Keep the detailed instructions in this Markdown file. Short comments beside each figure in `sections/Section2.tex` identify the corresponding brief. This keeps the report source readable while preserving the reasoning behind each figure.

There is currently one visible drafting box per subsection, with a caption and cross-reference. These boxes are **not finished figures**. Export the student's artwork as the following PDFs into `Report/figures/`; the LaTeX will automatically use each file when it exists.

| Subsection | Required filename | Figure reference label | Main job |
|---|---|---|---|
| 2.1 | `background_planck_cosmology.pdf` | `fig:background-planck` | Connect early-Universe evidence, expansion, and today's cosmic contents |
| 2.2 | `background_desi_dr2.pdf` | `fig:background-desi` | Explain the DESI distance constraints and the dark-energy question |
| 2.3 | `background_lensing_density.pdf` | `fig:background-lensing` | Connect lensing geometry, image distortions, and matter scales |
| 2.4 | `background_s8_surveys.pdf` | `fig:background-surveys` | Compare successive Stage III analyses and explain Stage IV prospects |
| 2.5 | `background_intrinsic_alignment.pdf` | `fig:ia-clue` | Show where GG, GI, and II come from |
| 2.6 | `background_model_tradeoff.pdf` | `fig:background-model-tradeoff` | Connect the modelling tradeoff to the experiment actually performed |

The report currently limits these images to 96% of the text width and 34% of the text height, preserving their aspect ratios. Aim for approximately 15 cm wide by 6–8 cm high. Keep editable originals alongside the working artwork. A completed image may be taller than its drafting box, so rebuild and check pagination after inserting it.

### Shared design choices

- Use a white background, a small consistent colour palette, and readable labels. Aim for labels of at least 9–10 pt at their final printed size. Use line styles as well as colour to distinguish contours.
- Put short explanations beside the feature they describe. Define symbols in the text or caption; avoid paragraphs inside figures.
- Use vector shapes and text for diagrams and plots. Preserve sufficient resolution for the Planck image. Check that PDF export has not clipped labels or substituted mathematical symbols.
- Distinguish **observations**, **illustrations**, and **future possibilities** in the panel labels. Mark exaggerated distortions and schematic curves explicitly.
- Cite the source of any image, data, or reproduced panel in the final caption. Include the specific analysis and figure number where relevant. Use the source's stated credit and reuse conditions. A citation alone does not describe whether a panel was reproduced or replotted.
- Never draw plausible-looking survey contours by hand. One-dimensional central values and error bars do not determine a two-dimensional contour or its parameter correlation.
- Revise the draft captions to describe the figure actually delivered. In particular, remove instructions such as “must retain” once the plotting choice is settled.

## Figure for 2.1 — the expanding Universe and its contents

**Reader's takeaway:** ancient light supports a model of an expanding Universe, but much of that model's present-day contents remains unexplained.

### Suggested layout

Use two panels across the top and a narrow strip below them.

1. **Top left, about two-thirds of the width:** the Planck all-sky CMB temperature map. Label it “Light released about 380,000 years after the Big Bang.” Add a short note: “Colours show tiny temperature differences.” Explain that the oval represents directions across the sky, rather than the physical outline of the Universe.
2. **Top right:** a simple stacked bar or three clearly labelled regions showing approximately 5% ordinary matter, 27% dark matter, and 68% dark energy. Title this panel “Today's energy budget, assuming ΛCDM.” Use a different visual treatment from the CMB colour scale so the two are not confused.
3. **Bottom strip:** two small drawings of the same expanding coordinate grid, labelled “Earlier: a < 1” and “Today: a = 1.” Show an emitted light wave with a shorter wavelength and the received wave with a longer wavelength. Place `1 + z = λobserved / λemitted = 1/a` beneath it. Galaxies can be dots carried farther apart on large scales; show the same dots in both drawings.

### Sources

- [ESA: Planck's view of the cosmic microwave background](https://www.esa.int/ESA_Multimedia/Images/2018/07/Planck_s_view_of_the_cosmic_microwave_background). The page identifies the 2018 map and the credit **ESA/Planck Collaboration**. Retain the source credit. If the chosen map includes a colour bar, preserve its values and units; do not invent a numerical scale for an image that lacks one.
- [Planck 2018 results I: overview and legacy — ADS](https://ui.adsabs.harvard.edu/abs/2020A%26A...641A...1P/abstract), citation key `2020A&A...641A...1P`.
- [Planck 2018 results VI: cosmological parameters — ADS](https://ui.adsabs.harvard.edu/abs/2020A%26A...641A...6P/abstract), key `2020A&A...641A...6P`.

### Scientific checks

The CMB panel is a map of temperature variations in radiation, not a picture of today's dark matter. The 5/27/68 fractions apply approximately **today**, not at the time the CMB was released. Radiation is negligible in this rounded present-day illustration but mattered greatly in the early Universe. Do not depict the Big Bang as an explosion at a central point into pre-existing empty space, or imply that bound objects such as atoms expand with the grid. The wave is an illustration of cosmological redshift, not an actual Planck measurement of an individual travelling wave.

**Student explanation check:** “What did Planck measure directly, and which part of this figure is inferred using a cosmological model?”

## Figure for 2.2 — DESI DR2 and dark energy

**Reader's takeaway:** the measurements constrain combinations of parameters, and evidence for changing dark energy depends on the data combination and model.

### Suggested layout

Make two similarly sized contour panels, labelled (a) and (b). Preserve axis units, probability levels, and dataset labels.

**Panel (a): matter density and the BAO ruler.** Start from **Figure 8, left panel**, in the DESI DR2 BAO paper. Show the DESI DR1 and DR2 comparison and the identified CMB comparison within ΛCDM. Explain the axes as “Matter fraction, Ωm” and “Expansion rate × ruler length.” The requested quantity is `H0 rd`.

Check the original axis carefully before redrawing: authors may express this combination as `h rd` in Mpc, with `h = H0/(100 km s⁻¹ Mpc⁻¹)`. In that convention, `H0 rd = 100 (h rd / Mpc) km s⁻¹`. Either retain the original quantity and explain the conversion, or transform both plotted values and units consistently. Do not change only the axis title. Match the caption to the choice. Identify precisely which Planck/ACT lensing combination is used.

**Panel (b): constant or changing dark energy.** Start from **Figure 11** of the same paper. Plot `w0` horizontally and `wa` vertically, with the DESI+CMB and supernova combinations identified. Mark `(-1, 0)` with a black cross labelled “Cosmological constant.” A small annotation can say “w0: value today; wa: change with expansion.” Retain the distinction between Pantheon+, Union3, and DES-SN5YR rather than labelling them all simply “DESI.”

Use light outer fills and stronger inner outlines for the 95% and 68% regions. Keep one shared explanatory key: “Allowed parameter combinations under the stated model.” The caption can give the more precise term, posterior probability.

### Sources and preparation route

- [DESI DR2 results II: BAO measurements and cosmological constraints — ADS](https://ui.adsabs.harvard.edu/abs/2025PhRvD.112h3515A/abstract), key `2025PhRvD.112h3515A`.
- [Paper with Figures 8 and 11, arXiv version 3](https://arxiv.org/html/2503.14738v3).
- [Authors' supplementary data on Zenodo](https://zenodo.org/records/16644577). Inspect the supplied README and figure/data products before selecting files. If suitable released products are available, replot them with documented selections. Otherwise, use clearly attributed published panels with legible labels and the applicable reuse permission. Do not infer a posterior from a screenshot or manually approximate its shape.
- The text also cites the later [DESI DR2 intergalactic-hydrogen analysis](https://ui.adsabs.harvard.edu/abs/2026arXiv260727410D/abstract), key `2026arXiv260727410D`. This is context for the evolving evidence; do not silently replace the requested BAO panels with that different analysis.

### Scientific checks

The two panels use different model assumptions. Label ΛCDM on panel (a) and the evolving `w0–wa` model on panel (b). BAO principally constrains a ruler–expansion combination; it does not measure `H0` alone without further information. The reported 2.8–4.2σ preferences belong to specific combinations, not to a DESI-only discovery. Contour overlap is not a substitute for the paper's statistical calculation. DR1 and DR2 overlap in observations and are not independent experiments.

**Student explanation check:** “Why does combining DESI with different supernova samples give different contours?”

## Figure for 2.3 — lensing, density, and spatial scale

**Reader's takeaway:** weak lensing changes images, while correlations across many images reveal the distribution of matter over a range of scales and distances.

### Suggested layout

Use an original illustration with three compact rows.

1. **Light paths:** place a distant galaxy on the left, intervening matter in the middle, and the observer on the right. Draw two nearby light rays deflected by the intervening gravitational field, with arrows indicating the direction of travel. A faint straight reference path may show where the source would appear without lensing. Label “Source galaxy,” “Intervening matter,” and “Observer.” Keep deflections modest enough that the drawing does not suggest multiple strong-lensing images. State “Deflections exaggerated.”
2. **Image distortions:** begin with the same small circular source in each example. Show the unlensed circle; positive convergence κ as a larger circle; one shear component γ1 as a horizontal ellipse; and γ2 as an ellipse rotated by 45°. Use dashed original outlines behind the transformed images. A note can explain that the two shear components describe orientation as well as elongation. Convergence changes size; shear changes shape. Area preservation for a pure shear is only a first-order statement, so avoid claiming it is exact for a visibly exaggerated transformation.
3. **Density and scale inset:** draw a broad smooth density fluctuation and a more rapidly varying one over the same distance interval. Mark the spatial wavelengths with double-headed arrows. Label them “Large scale → small k” and “Small scale → large k,” followed by `k = 2π/λspatial`. A small box can state “Pδ(k,z): strength of density fluctuations at each scale and time.” This is enough; a realistic cosmological power-spectrum curve is not necessary.

Section 2.3 is the main home of the density-field explanation. It introduces `δ = (ρ − mean density)/(mean density)`, followed by `Pδ(k,z)` and `k`, immediately before explaining how galaxy-shape correlations are measured. The figure should support that sequence rather than add a second independent derivation.

### Source

[Bartelmann & Schneider: Weak gravitational lensing — ADS](https://ui.adsabs.harvard.edu/abs/2001PhR...340..291B/abstract), key `2001PhR...340..291B`. Use this for the physical definitions; the student can design the geometry and shapes herself.

### Scientific checks

Label the density inset “Schematic.” It is not a simulation snapshot or observed map. The symbol λ here means a **spatial density wavelength**; in 2.1 it meant the wavelength of light. Keep the subscript or a clear label. Comoving distance removes the common expansion of the coordinate grid. A large value of k means a small spatial scale, not a large redshift. The matter power spectrum is different from the CMB angular power spectrum. Avoid implying that one measured galaxy shape determines the density at one point: cosmic shear is a statistical signal integrated along the line of sight.

**Student explanation check:** “Why must we measure many galaxies, and why does a small spatial structure correspond to a large k?”

## Figure for 2.4 — survey comparisons and future observations

**Reader's takeaway:** more data and improved treatment of measurement errors can change the inferred cosmology; the same observations can also give different answers under different alignment models.

### Suggested layout

Use three contour panels across the top, one each for KiDS, DES, and HSC. Use `Ωm` horizontally and `S8` vertically, with common axis ranges where practical. Within each panel, show the previous analysis as an outline and the latest selected analysis with filled contours. Retain 68% and 95% levels. Use a light grey Planck reference with the exact likelihood/model combination named. If the references differ across papers, say so rather than presenting them as one identical dataset.

Below the measurements, add a narrow strip of three simple cards:

- **Euclid:** wide surveys with sharp space-based imaging and complementary distance information.
- **Rubin:** repeated, wide-field imaging in several optical bands.
- **Roman:** sharp near-infrared imaging for cosmology and complementary surveys.

Title the top row “Stage III: measured cosmic-shear constraints” and the bottom strip “Stage IV: prospects.” These cards should explain what each programme adds; they do not need launch dates, fixed survey areas, or numerical forecast ellipses.

### Which analyses to compare

| Survey | Earlier comparison | Latest comparison used in the text | Important choice |
|---|---|---|---|
| KiDS | KiDS-1000 reanalysis, Li et al. 2023a; optionally add the original Asgari et al. 2021 result for history | KiDS-Legacy, Wright et al. 2025 | Label the original and reanalysed KiDS-1000 results separately; calibration changes matter |
| DES | Year 3 cosmic shear, Amon et al. 2022 | Year 6 cosmic shear, DES Collaboration et al. 2026 | State NLA or TATT for each contour; Y6's two IA choices use the same data |
| HSC | Year 1 correlation-function analysis, Hamana et al. 2020 | Year 3 correlation-function analysis, Li et al. 2023b | Compare the same observable family; do not substitute a power-spectrum or 3×2pt result without relabelling |

This is a comparison of published analyses, not a newly harmonized joint reanalysis. Changes in priors, calibration, scale cuts, or nuisance models can accompany a larger sample. Explain those choices in a short caption or note. If displaying both DES Y6 IA models makes the panel crowded, use one clearly identified model for the release comparison and a small inset for the alternative.

### Data and figure sources

- [KiDS public science data](https://kids.strw.leidenuniv.nl/sciencedata.php) lists the Legacy and KiDS-1000 products. Select the chains associated with the cited analysis and read their documentation. Original KiDS-1000 chains are not automatically the Li et al. 2023 reanalysis.
- [KiDS-Legacy paper](https://arxiv.org/html/2503.19441v2): Figure 9 includes the Ωm–S8 comparison with Planck; Appendix I and Figure 24 help explain the steps from KiDS-1000 to Legacy. Consult these when explaining why the constraints changed.
- [KiDS-1000 improved-shape analysis](https://www.aanda.org/articles/aa/pdf/2023/11/aa47236-23.pdf): its headline `0.776` uses a maximum-posterior estimate and a projected joint interval. Do not present it as a marginal mean beside another paper's marginal summary. For contours, use the actual two-dimensional posterior. If using one-dimensional summaries in a supplementary table, identify the estimator and interval convention consistently.
- [DES Year 6 papers and data links](https://www.darkenergysurvey.org/des-y6-cosmology-results-papers/) and [DES Y6A2 public release](https://des.ncsa.illinois.edu/releases/y6a2). Use the **cosmic-shear** paper, [arXiv:2602.10065](https://arxiv.org/abs/2602.10065), and its associated products. The `S8 = 0.798` and `0.783` values in the text refer to the NLA and TATT alternatives, respectively.
- [DES Year 3 cosmic-shear paper](https://arxiv.org/abs/2105.13543).
- [HSC Year 1 correlation functions](https://arxiv.org/abs/1906.06041) and [HSC Year 3 correlation functions](https://arxiv.org/abs/2304.00702). Follow the papers' data-availability information for the matching chains or use attributed published contours. Do not fabricate missing joint information from the quoted S8 interval.
- For the prospects strip: [Euclid overview](https://ui.adsabs.harvard.edu/abs/2025A%26A...697A...1E/abstract), [Rubin/LSST design](https://ui.adsabs.harvard.edu/abs/2019ApJ...873..111I/abstract), and [Roman cosmology study](https://ui.adsabs.harvard.edu/abs/2021MNRAS.507.1746E/abstract). These explain scientific capabilities. For current Roman programme definitions, consult the [official High-Latitude Wide-Area Survey page](https://roman-docs.stsci.edu/roman-community-defined-surveys/high-latitude-wide-area-survey), rather than treating older forecast assumptions as the current observing plan.

### Plotting steps and checks

1. Record the paper version, chain/product name, model, observable, parameter definitions, weights, and any documented burn-in treatment.
2. Use the supplied S8 column if its definition matches the report. Otherwise calculate it from each posterior sample's σ8 and Ωm. Preserve sample weights; do not discard weighted samples as though they were interchangeable draws.
3. Use a documented posterior-contour method and check that its regions correspond to 68% and 95% integrated probability. Record any smoothing and check against the published panel. Do not independently rescale contours to make agreement look better.
4. Label original versus updated analyses and the selected Planck reference. If a common plotting pipeline is not practical, assemble attributed published panels and state that their analysis choices differ. Update the caption accordingly.
5. Explain the limited claim: KiDS-Legacy is consistent with Planck, while some DES/HSC comparisons still differ at roughly 2σ. A one-dimensional S8 tension and a multidimensional tension are different summaries. Do not label the whole S8 question “solved.”

**Student explanation check:** “Could these contours move even if the sky had not changed? Which changes come from more observations, and which from calibration or modelling?”

## Figure for 2.5 — intrinsic alignment and the observed shape

**Reader's takeaway:** correlated galaxy shapes can arise from lensing, the galaxies' own orientations, or a cross-correlation of the two.

### Suggested layout

At the top, show a small original galaxy ellipse, a shear arrow, and the resulting observed ellipse, beside the schematic relationship

`measured ellipticity ≈ intrinsic ellipticity + gravitational shear + measurement noise`.

Label the diagram “Weak-distortion illustration; not literal addition of images.” The text explains the shape-estimator response and the more precise reduced shear. Do not attempt to make the drawn ellipticities obey an unspecified exact addition law.

Below, use three panels with a consistent observer position and increasing distance/redshift upwards:

- **GG — two lensed images:** two distant galaxies have their light deflected by correlated foreground matter. Use one colour for the matter and curved arrows for light deflection. The shared influence is the lensing field; the source galaxies need not be neighbours.
- **II — related intrinsic shapes:** two galaxies at similar distances share a tidal environment and have correlated intrinsic orientations. Show a faint common environment or tidal arrows around them. Do not draw a foreground lens as the cause of their intrinsic shapes.
- **GI — intrinsic shape and lensing:** a foreground galaxy's intrinsic orientation is correlated with the surrounding matter. That matter also lenses a more distant galaxy. Label the foreground member “I” and the background contribution “G.” Draw the relevant line of sight so it is clear that this is not the same geometry as II.

Place `GG + GI + IG + II` below the panels, with a brace grouping GI and IG into the familiar shorthand `GG + GI + II`. Explain that the shorthand GI includes both orderings when relevant. Differentiate tidal influence and light deflection with line styles as well as colour.

### Sources

- [Chisari's intrinsic-alignment review — ADS](https://ui.adsabs.harvard.edu/abs/2025A%26ARv..33....5C/abstract), key `2025A&ARv..33....5C`.
- [Bartelmann & Schneider — ADS](https://ui.adsabs.harvard.edu/abs/2001PhR...340..291B/abstract), for shear and reduced shear.

### Scientific checks

The intrinsic shape includes a large random component as well as any coherent alignment. II and GI refer to correlations, not the entire intrinsic ellipticity of one galaxy. Random shapes and measurement noise contribute uncertainty and are treated separately in the signal decomposition. GI can connect different source redshifts, so separating nearby galaxy pairs does not remove every IA contribution. Do not label all tidal alignments universally radial or tangential: the response depends on the galaxy population and model. Panel sizes and arrow lengths should not imply measured GG/GI/II amplitudes.

**Student explanation check:** “How can a foreground galaxy's intrinsic shape correlate with the lensing of a different, more distant galaxy?”

## Figure for 2.6 — the modelling tradeoff and this project

**Reader's takeaway:** the project asks a specific numerical question motivated by an astrophysical problem; a successful reconstruction would still need further tests before use in cosmology.

### Suggested layout

Draw a compact flow chart with balanced branches and a clearly bounded experiment.

**Top box:** “Galaxy alignments are uncertain across spatial scale and time.”

**Two branches underneath:**

| Restricted response, illustrated by NLA | More flexible response, illustrated by TATT |
|---|---|
| Fewer adjustable parameters | Additional tidal-response terms |
| Usually easier to constrain and explore | Can describe a wider range of responses |
| May miss relevant behaviour | More directions can be weakly constrained |
| Risk of shifting inferred cosmology if inadequate | Extra computation and parameter degeneracies |

Keep the boxes the same size. Avoid a green “good” versus red “bad” treatment. TATT's suitability depends on the data and scales, and NLA can be adequate for some analyses. The chosen number of free parameters depends on the prescription and dataset, so use qualitative labels here.

**Join the branches with the question:** “Can a flexible response family be described with fewer numbers at useful accuracy?”

**Bottom experiment strip:**

`13 input parameters → response on a 31 × 101 grid → PCA or autoencoder → two coordinates → reconstructed response → compare errors`.

If space is tight, place PCA and autoencoder as two small parallel boxes around the middle of the strip. Label the input `positive response AΘ(k,z)` and the test `same validation samples`. Identify the source model as “NLA-inspired synthetic family.” Use a small surface thumbnail drawn schematically, not an uncited new result.

**Dashed future box:** “Still needed: errors in lensing predictions; probabilities for the new coordinates; tests in cosmological inference.” Connect it after the reconstruction assessment.

### Sources and scientific checks

- [Bridle & King 2007 — ADS](https://ui.adsabs.harvard.edu/abs/2007NJPh....9..444B/abstract), key `2007NJPh....9..444B`, for the NLA prescription.
- [Blazek et al. 2019 — ADS](https://ui.adsabs.harvard.edu/abs/2019PhRvD.100j3506B/abstract), key `2019PhRvD.100j3506B`, for the extended tidal model.
- The experiment strip comes from this report's existing model and two-coordinate pilot. It must stay consistent with Sections 3–6.

The 3,131 grid values are already generated by 13 parameters; do not claim 3,131 independent physical degrees of freedom. The compressed response is positive, while the full signed alignment prescription contains additional factors. It is not the matter power spectrum, a complete TATT emulator, or a learned prior. The diagram must not imply that PCA or the autoencoder automatically removes parameter degeneracies, speeds up a full cosmological analysis, or establishes unbiased cosmology. Do not introduce a new preferred dimension or result from a different run while the later sections still describe the two-coordinate pilot.

**Student explanation check:** “What has this experiment tested, what would count as a failure, and what remains to be tested before applying it to real observations?”

## Reference file and citation workflow

### What has been configured

`Reference.bib` contains the **AASTeX entries exported from ADS**, using `\bibitem` syntax. It is deliberately not a conventional BibTeX database of `@article` records. The filename follows the requested project convention. `main.tex` reads it using `\input{Reference.bib}` inside `thebibliography`, with `natbib` providing author–year citations. No BibTeX or Biber run is needed.

The exported bibliographic details are retained. DOI hyperlinks and an ADS link have been added to each item. The two abbreviated “Li et al. (2023)” citations and the two Planck (2020) citations have a/b labels to distinguish them. The entries are in alphabetical order; full journal names are supplied by macros in the preamble. Do not switch to an AASTeX journal document class just to use these reference entries.

Use, for example:

```latex
\citet{2025PhRvD.112h3515A} measured ...
... from the DESI observations \citep{2025PhRvD.112h3515A}.
... in successive analyses \citep{2021A&A...645A.104A,2025A&A...703A.158W}.
```

For an additional reference: open its ADS record, confirm the title and version, choose **Export → AASTeX**, copy the exported `\bibitem` into the alphabetically appropriate position, and add the ADS source link. Check for duplicate author–year labels. Keep any needed new journal macro in `main.tex`. Do not paste a BibTeX record into this particular file without changing the bibliography workflow.

### ADS source inventory

The descriptions below identify what each citation supports; the linked ADS record is the authority for its bibliographic details. There are 21 entries, all cited in Sections 1–2.

| Reference | Use in the revised text | ADS record / citation key |
|---|---|---|
| Abdul Karim et al. 2025 | DESI DR2 BAO and evolving-dark-energy constraints | [2025PhRvD.112h3515A](https://ui.adsabs.harvard.edu/abs/2025PhRvD.112h3515A/abstract) |
| Amon et al. 2022 | DES Year 3 cosmic shear | [2022PhRvD.105b3514A](https://ui.adsabs.harvard.edu/abs/2022PhRvD.105b3514A/abstract) |
| Asgari et al. 2021 | Original KiDS-1000 cosmic shear | [2021A&A...645A.104A](https://ui.adsabs.harvard.edu/abs/2021A%26A...645A.104A/abstract) |
| Bartelmann & Schneider 2001 | Weak lensing, shear, density statistics | [2001PhR...340..291B](https://ui.adsabs.harvard.edu/abs/2001PhR...340..291B/abstract) |
| Blazek et al. 2019 | Extended tidal-alignment model, TATT | [2019PhRvD.100j3506B](https://ui.adsabs.harvard.edu/abs/2019PhRvD.100j3506B/abstract) |
| Bridle & King 2007 | Nonlinear-alignment prescription | [2007NJPh....9..444B](https://ui.adsabs.harvard.edu/abs/2007NJPh....9..444B/abstract) |
| Chisari 2025 | IA mechanisms, evidence, and modelling limitations | [2025A&ARv..33....5C](https://ui.adsabs.harvard.edu/abs/2025A%26ARv..33....5C/abstract) |
| DES Collaboration et al. 2026 | DES Year 6 cosmic shear; NLA/TATT comparison | [2026arXiv260210065D](https://ui.adsabs.harvard.edu/abs/2026arXiv260210065D/abstract) |
| DESI Collaboration et al. 2026 | Later intergalactic-hydrogen analysis | [2026arXiv260727410D](https://ui.adsabs.harvard.edu/abs/2026arXiv260727410D/abstract) |
| Eifler et al. 2021 | Roman cosmological survey prospects | [2021MNRAS.507.1746E](https://ui.adsabs.harvard.edu/abs/2021MNRAS.507.1746E/abstract) |
| Euclid Collaboration et al. 2025 | Euclid mission and scientific capabilities | [2025A&A...697A...1E](https://ui.adsabs.harvard.edu/abs/2025A%26A...697A...1E/abstract) |
| Hamana et al. 2020 | HSC Year 1 correlation functions | [2020PASJ...72...16H](https://ui.adsabs.harvard.edu/abs/2020PASJ...72...16H/abstract) |
| Hinton & Salakhutdinov 2006 | Autoencoder dimensionality reduction | [2006Sci...313..504H](https://ui.adsabs.harvard.edu/abs/2006Sci...313..504H/abstract) |
| Ivezić et al. 2019 | Rubin/LSST scientific design | [2019ApJ...873..111I](https://ui.adsabs.harvard.edu/abs/2019ApJ...873..111I/abstract) |
| Jolliffe & Cadima 2016 | PCA and its interpretation | [2016RSPTA.37450202J](https://ui.adsabs.harvard.edu/abs/2016RSPTA.37450202J/abstract) |
| Li, S.-S., et al. 2023a | Updated KiDS-1000 analysis | [2023A&A...679A.133L](https://ui.adsabs.harvard.edu/abs/2023A%26A...679A.133L/abstract) |
| Li, X., et al. 2023b | HSC Year 3 correlation functions | [2023PhRvD.108l3518L](https://ui.adsabs.harvard.edu/abs/2023PhRvD.108l3518L/abstract) |
| Planck Collaboration et al. 2020a | Planck overview, CMB map context | [2020A&A...641A...1P](https://ui.adsabs.harvard.edu/abs/2020A%26A...641A...1P/abstract) |
| Planck Collaboration et al. 2020b | ΛCDM parameters and cosmic contents | [2020A&A...641A...6P](https://ui.adsabs.harvard.edu/abs/2020A%26A...641A...6P/abstract) |
| Spurio Mancini et al. 2022 | Existing neural-network applications in cosmology | [2022MNRAS.511.1771S](https://ui.adsabs.harvard.edu/abs/2022MNRAS.511.1771S/abstract) |
| Wright et al. 2025 | KiDS-Legacy cosmology and calibration changes | [2025A&A...703A.158W](https://ui.adsabs.harvard.edu/abs/2025A%26A...703A.158W/abstract) |

## Handoff and completion checks

- Produce the three original explanatory diagrams first (2.3, 2.5, 2.6), then assemble the sourced map and contours (2.1, 2.2, 2.4). The contour comparison needs the most checking.
- Read each subsection aloud, then explain its figure without reading the caption. Replace wording that the student cannot yet explain with language she understands, preserving the scientific meaning.
- Insert the six PDFs under the exact filenames, update the captions with final source credits and analysis choices, and compile `main.tex` with `latexmk -pdf`. Check every figure reference and citation, then inspect the rendered pages for placement and label size.
- The main text should lead into each figure before moving to the next topic. If a completed figure floats too far from its subsection, adjust placement locally and recheck the surrounding pages.
- The current revision changes the title, Sections 1–2, and reference support. The abstract and Sections 3 onward have been preserved. The earlier review's outstanding issues in those sections, including historical-run provenance and the statement that the test split was never read, still require a separate revision before final submission.

The current draft is a starting point for the student's figure work and final wording choices. Its placeholders and provisional captions should not appear in the submitted report.
