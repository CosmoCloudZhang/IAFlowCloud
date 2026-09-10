# IAFlow research report: assessment and editing plan

Prepared 10 September 2026 | S.-T. Yau High School Science Award, North America, 2026

This is an editorial and scientific assessment of the current report, informed by the project implementation, saved experiment artifacts, teaching materials, a fresh LaTeX build, and current primary literature. It is a plan for revision; proposed experiments and example paragraphs are not completed research. The student's report sources and figures have been preserved.

## 1. Main recommendation

The report has a sound foundation: a focused computational question, a documented synthetic model family, a linear baseline, explicit data partitions, and unusually candid limitations. Its main weakness is that it presents an early experiment after the underlying project has moved substantially further. More polished background alone will not resolve that mismatch.

The strongest defensible research question is:

> How compactly can a flexible family of intrinsic-alignment response surfaces be represented, how does nonlinear compression compare with PCA at the same dimension, and which physical features remain difficult to reconstruct?

Keep the two-coordinate experiment as the starting point. Make the accuracy-versus-dimension study, together with its failure cases, the main result. The existing artifacts support a more interesting conclusion than the current report: two coordinates improve on two-component PCA but are inadequate under the project's chosen average-error criterion; six-coordinate models meet that criterion, while localized errors remain appreciable. That is a useful result with an honest boundary.

The key priorities, in order, are:

1. Replace the untraceable historical headline with one coherent, verified set of experiments that the student can explain and accurately attribute.
2. Explain the physical purpose of the compressed quantity and the benefit sought beyond the existing 13-parameter analytic representation.
3. Establish the contribution relative to existing IA modelling and scientific machine learning; add citations and a bibliography.
4. Reorganize the introduction and background around a short chain of reasoning from observations to the specific experiment.
5. Add reconstruction and failure-case figures, then complete authorship information and improve layout.

My assessment is that the current PDF is a readable research draft, but not yet a strong final competition submission. Its limitations are repairable, and much of the necessary evidence already exists. I cannot estimate an award probability from the report or the published criteria.

## 2. Competition fit and timing

### Recommended category: Physics

Astronomy and cosmology fit the Physics Award. The official North America Physics page includes computational investigations in physics. Here the scientific subject is an astrophysical uncertainty in gravitational-lensing cosmology, with machine learning serving the investigation. A Computer Award entry would require a stronger contribution to computing methodology itself and a broader computing benchmark. There is no need to search for a generic STEM category. [Physics submission guide](https://www.yau-science-awards.org/competitioncategory/physics-guide.html).

The published Physics criteria emphasize physical relevance, originality, creative methodology, rigorous reasoning, potential contribution, and clear writing. They also distinguish established background from the team's contribution. My editorial interpretation is that an advanced topic or a large network will be less persuasive than a well-controlled experiment whose decisions and limitations the student understands. [Physics assessment criteria](https://www.yau-science-awards.org/competitioncategory/physics-assessment.html).

The dated 2026 announcement gives **July 1-September 15** for registration and submission of application materials. Use that announcement in preference to generic assessment-page dates. As of this review, five calendar days remain; the public announcement does not specify the closing time or time zone. Check those details in the submission portal. The immediate plan below is therefore a short submission revision, with larger research extensions separated from it. [2026 North America schedule](https://www.yau-science-awards.org/bulletin/show-102053.html).

The published guide calls for a PDF with the required front matter, a bibliography starting on a separate page, and an account of contributions and external assistance. The current report has no citations or bibliography and still contains contribution prompts. Those are substantive completion tasks. Do not assume the hidden note in Section8 establishes a US-specific requirement of 500-1,500 words; that length was not confirmed by the public North America pages reviewed here. [Submission guide](https://www.yau-science-awards.org/competitioncategory/physics-guide.html).

One eligibility point needs checking against the actual arrangements: the published guide restricts instructor affiliation with commercial training organizations. The local reference material is branded as a research-training programme. That alone does not establish the nature of this student's supervision or eligibility, but it makes clarification with the organizer sensible before submission. Describe assistance truthfully; do not try to resolve an eligibility question through wording changes. [Instructor requirements](https://www.yau-science-awards.org/competitioncategory/physics-guide.html).

### Where the current draft stands

| Dimension | Assessment of current draft | Most useful improvement |
|---|---|---|
| Physical relevance | Good topic; connection from compressed amplitude to observed shear is incomplete | Explain the IA spectra and where they enter lensing measurements |
| Research question | Clear but artificially restricted to two coordinates | Study the accuracy-dimension tradeoff and its failure modes |
| Originality | Not established by a literature comparison | Identify the specific experiment and conclusions beyond established PCA/AE methods |
| Methodological rigor | Good split and normalization principles; reported run is not reproducible locally | Use exact saved configurations, checkpoints, and matching metrics |
| Results | Too little evidence for the sophistication of the methods section | Add current dimension study, paired reconstructions, and tail diagnostics |
| Student contribution | Cannot be assessed from placeholders or repository size | Complete a factual account of decisions, analysis, code, and assistance |
| Writing and presentation | Generally accessible; uneven emphasis and several layout issues | Shorten the abstract, strengthen motivation, improve figure balance |

## 3. Compilation and document inspection

### Verified build

A fresh copy containing the current TeX sources and the four report figures was built in an isolated directory, without copying old auxiliary files. `latexmk` completed successfully with pdfLaTeX from TeX Live 2023. The final document has **16 physical PDF pages**, comprising an unnumbered cover and printed pages 1-15. The final converged log contains no LaTeX warnings, undefined references, overfull boxes, or underfull boxes. Initial cross-reference messages resolved during the normal subsequent passes.

The validation command was:

```text
latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error \
  -cd /private/tmp/iaflow-report-review-20260910/source_snapshot/main.tex
```

All 16 pages were rendered and visually reviewed. Compilation success should be distinguished from scientific readiness and final typesetting quality.

### Visual findings

| PDF page | Finding | Editing action |
|---|---|---|
| 1 | Supervisor is shown as a dash; date remains August 2026 | Complete the verified name and intended submission date |
| 2 | Abstract fits but is dense and carries too many procedural details | Aim for roughly 180-250 words; keep the main result and its limit |
| 3 | Contents are clear and correctly distinguish Sections 1-6, Appendices A/B, and acknowledgements | Preserve that distinction when renaming source files |
| 7-8 | The 13-parameter table leaves its last row alone on the next page | Keep the complete table together, or move the detailed version to an appendix |
| 9 | Sampling figure occupies a largely empty float page; legend text is small | Integrate it with the dataset discussion and enlarge labels |
| 11 | Architecture table is legible but long; preceding paragraph is interrupted by the float | Reduce the main-text table to the actual reported architecture |
| 12 | Both results figures show PCA; no neural reconstruction is visible | Give at least one main figure to the method named in the title |
| 14 | Short summary is isolated on a page with substantial unused space | Reflow after substantive editing; avoid unnecessary forced breaks |
| 16 | Acknowledgements are still instructions and placeholder entries | Replace with factual prose and a concise completed contribution table |

The figures' apparent minus-sign extraction problems are not evidence of visibly missing glyphs: the rendered plots display their mathematical labels. This is one reason visual inspection was necessary.

In `main.tex`, `\draftnote` currently expands to nothing regardless of `\reportdrafttrue` or `\reportdraftfalse`. The README still describes visible blue drafting notes. Restore a functioning draft mechanism if wanted, or update the README; merely toggling the flag will not reveal the notes. Do not hide unfinished contribution text and treat the report as complete.

## 4. Project context and evidence audit

### What the project actually studies

The active compression target is the positive, dimensionless **A_Theta(k,z)** response surface. The signed full amplitude is assembled as **A_IA = -A0 A_Omega A_Theta**. The analytic model separates cosmological growth scaling and the overall signed multiplier from a family of scale/redshift responses. The compression experiments do not infer dark energy, fit a galaxy survey, or learn the complete signed IA power spectra.

The source dataset contains 100,000 surfaces, each on a 31-by-101 grid. The sampled family has 13 shape parameters. The documented grid spans redshift 0-3 and wavenumber 0.01-10 Mpc^-1. The stored split is 80,000 training, 10,000 validation, and 10,000 test samples. The active data contract is a pointwise base-10 logarithm, a training mean surface, and a single training global RMS scale.

The project now includes PCA, direct Conv1D and Conv2D autoencoders, an independent PCA-plus-MLP autoencoder, latent dimensions 2/4/6/8/10, several capacity tiers, common validation metrics, saved histories and diagnostic plots. The current report describes only an earlier Conv1D trial. The training materials additionally discuss VAEs and conditional generation; those are educational plans and must not be presented as completed IAFlow results.

### Evidence used for this review

| Project material | Role in the assessment |
|---|---|
| `README.md`, `pyproject.toml`, environment specifications | Active scope, package boundaries, reproducibility workflow |
| `Code/ia_models/nla/` and `Code/ia_models/utilities/` | Amplitude equations, parameter sampling, grids, schema and splits |
| `Code/iaflow/core/`, `autoencoder/`, `pca_autoencoder/` | Preprocessing, metrics, architecture, losses, saved-run contracts |
| `Code/iaflow/comparison.py`, `validation_visuals.py` | Matching PCA, candidate eligibility, diagnostics and model-selection rules |
| `Config/NLA/`, `Scripts/NLA/` | Architecture tiers, optimization policy and sweep behaviour |
| Seven `Notebooks/NLA/` notebooks | Model interpretation, recorded analysis, plots and selection workflow |
| `Data/NLA/Comparison/`, `Data/NLA/PCA/`, selected run artifacts | Quantitative evidence and provenance |
| `Reference/` lectures, teaching notebooks, homework, schedule and training plan | Educational scope and intended student progression |
| All report sources and the freshly rendered PDF | Actual claims, structure and visual presentation |

Paths in tables are repository-relative identifiers. The repository root is `/Users/s2227120/IAFlowCloud`. This was a source/artifact review, not a new training run or a full numerical re-execution of the project. No new test-set evaluation was performed.

### Current claims that need reconciliation

| Location | Current statement or implication | Required correction |
|---|---|---|
| Abstract; Sections 1, 4-6 | Two coordinates and a 25-epoch trial are the main experiment | Use a reproducible current study or explicitly restrict the report to a historical pilot |
| Section5, lines 33-41 | Only scalar summaries of the reported trial are available | An unavailable checkpoint cannot support a reproducible headline; do not silently substitute a different run |
| Section4, lines 75-115 | Historical architecture is paired with current optimization settings | Describe the exact resolved architecture and training history for every reported row |
| Section4, lines 79-88 | Dense encoder ends at 64 then latent; 2,039,969 weights | Revised Depth03 includes an additional width-16 layer; current L=2 has 2,041,857 learned parameters |
| Section5, lines 102-111; Section6, lines 23-25 | Dimension study is entirely future work | Relevant studies already exist; distinguish their availability from student ownership and comprehension |
| Section7, lines 7-8 | A single old Git commit identifies the work | Record source revision, uncommitted report state, dataset/PCA hashes and concrete run directories separately |
| Section7, lines 17-29 | Rank-25 PCA and shared seed describe provenance | Active PCA front end is rank 30; recover seeds from actual artifacts, not a later template |
| Several sections | Entire test split has never been read | Historical PCA notebooks contain test diagnostics; correct the claim and distinguish this from training leakage |

The two PCA PDFs in `Report/figures/` differ from the corresponding active project PDFs. This is more than a byte-level difference: the report plots stop at rank 25, while the active plots reach rank 30. The scale-redshift and prior-ensemble PDFs match their project counterparts. Do not run the synchronization script until choosing a coherent results version; it would overwrite the student's edited figure copies.

There is a second PCA distinction: the notebook's rank-curve calculation uses **256 validation surfaces**, while the scalar benchmark table uses **all 10,000**. Separate reconstruction checks use 64 rows per split. Each caption must state its own sample size and provenance. The report's stated rank-25 training variance is consistent with the first 25 modes of the current basis; the problem is the mixed basis/subset descriptions, not that every older number is wrong. Saved notebook cells also contain slightly different rank-30 training totals, so they should not be treated as one freshly executed analysis.

Historical `Code/PCA.ipynb` at Git commit `ca1bed7` explicitly includes test reconstruction diagnostics and an analysis of 10,000 test coefficient vectors, with saved outputs. This confirms historical test exposure. It does **not** establish that the PCA fit used test data or that neural hyperparameters were selected using those diagnostics. Current cache preparation also transforms all source rows, while fitting normalization on training rows only. These distinctions should replace the blanket claim that the test split was never read.

Suggested factual wording: **The reported model comparisons use validation data. Historical PCA diagnostics accessed the test split; no final evaluation of a frozen neural selection is reported here.** If an independent final assessment is needed, define a new holdout protocol before generating or inspecting its outcomes. Do not claim that the existing data have become unseen again.

### A verified saved comparison, not a final selection

The saved comparison manifest is dated **2 September 2026**, not the date of this review. Its result-table and audit-table hashes match the current files. The PCA metadata and metrics hashes also match. All ten recorded artifact hashes for each of the two six-coordinate diagnostic runs match, including checkpoints, histories, resolved configurations and validation diagnostics.

The saved manifest records 29 eligible cells out of 45 expected: 14 Conv1D, 15 PCA-AE and no eligible revised Conv2D cells. Its status is provisional, its selection is not frozen, and test evaluation is not allowed. Those are the coverage and policy of that saved snapshot; a training label in it does not establish that a process is running now.

The read-only current-artifact collector now finds **31 eligible cells out of 45**: 15 Conv1D, one Conv2D and 15 PCA-AE. There are 35 eligible run records because four Conv1D Depth03 cells have a second eligible run with the same seed, 42. The current input fingerprint differs from the saved manifest, and the selection notebook rejects duplicate eligible seeds. A simple rerun of model selection therefore needs a deliberate policy for these repeats first. Do not delete a run, choose the better repeat, or count them as independent seeds merely to make the notebook pass.

This affects the strongest-accuracy claim. Conv1D Depth05 L=10 now has log-RMSE 0.00151618, about 3.26% below the old Depth03 L=10 result, with approximately 4.53 times as many learned parameters. The September 9 Depth03 L=2 repeat records 98.6483% recovered variance and 0.0318952 log-RMSE; its L=6 repeat records 99.9200% and 0.00776147. These are additional completed validation runs, not an automatically updated final selection. Deterministic execution is disabled, and identical seed labels alone do not guarantee identical training trajectories.

The numerical tables below intentionally preserve the authenticated September 2 choices so their provenance remains unambiguous. For the revised report, either use that explicitly dated study with its scope stated, or update all figures and tables under a declared repeat-handling rule. Do not mix the best values from different snapshots.

The following values are transcribed from authenticated saved artifacts. Percentages are converted from the stored fractions; all rows use the 10,000-surface validation set. Neural rows below are Depth03, seed 42.

| Method | Coordinates | Recovered log variance (%) | Log10 RMSE | Mean relative error (%) |
|---|---:|---:|---:|---:|
| PCA | 2 | 96.2287 | 0.053275 | 8.6637 |
| Direct Conv1D AE | 2 | 98.3620 | 0.035110 | 5.9522 |
| PCA | 6 | 99.7614 | 0.013401 | 2.0611 |
| Direct Conv1D AE | 6 | 99.9161 | 0.007946 | 1.2461 |
| PCA-AE | 6 | 99.9194 | 0.007786 | 1.2151 |
| PCA | 10 | 99.9379 | 0.006834 | 1.0437 |
| Direct Conv1D AE | 10 | 99.9967 | 0.001567 | 0.2444 |

| Method | Coordinates | p95 surface relative RMS (%) | p99 surface maximum error (%) | Global maximum error (%) |
|---|---:|---:|---:|---:|
| PCA | 2 | 21.4722 | 92.2376 | 178.1312 |
| Direct Conv1D AE | 2 | 12.7828 | 62.2362 | 143.9268 |
| PCA | 6 | 5.5303 | 33.3298 | 55.6502 |
| Direct Conv1D AE | 6 | 3.2137 | 17.1725 | 25.5727 |
| PCA-AE | 6 | 3.1441 | 16.7881 | 48.0646 |
| PCA | 10 | 2.8900 | 15.7336 | 29.0353 |
| Direct Conv1D AE | 10 | 0.7058 | 3.7894 | 7.4082 |

The six-coordinate direct AE improves on six-component PCA, but ten-component PCA remains slightly better on average than that six-coordinate AE. At ten coordinates, the direct AE is much more accurate than ten-component PCA. Keep both equal-dimension and different-dimension comparisons explicit.

The PCA-AE six-coordinate result has slightly better average errors and a smaller saved model representation than the direct AE, but its worst error is larger. That is a useful tradeoff to discuss, not a reason to declare an unconditional winner. Its neural parameter count is 51,076; including the stored PCA transform and normalization increases the codec-state count to 151,329 scalars. The direct six-coordinate AE has 2,041,989 learned parameters and 2,045,121 codec-state scalars. PCA at ranks 2, 6 and 10 needs 9,393, 21,917 and 34,441 scalars for its basis and mean. These counts exclude the per-surface code, optimizer state and serialization overhead; they are not measured runtime speedups or complete checkpoint file sizes.

Exact run identities for the table are Conv1D Depth03 L=2/6/10 under `20260823-103003`, and PCA-AE Depth03 L=6 under `20260822-192700`. The newer runs discussed separately are Conv1D Depth05 L=10 under `20260830-115056` and Conv1D Depth03 L=2/6 under `20260909-003908`.

### The central conceptual distinction

The original 13 analytic parameters already provide a compact way to regenerate each synthetic surface. Therefore, “3,131 values are too many to carry through a calculation” is insufficient as the main motivation. The report should ask whether the **observable variation of this chosen family** admits an approximate representation with fewer effective coordinates, and whether nonlinear methods do so more efficiently than a linear basis at a stated error tolerance.

For a smooth map from 13 parameters to 3,131 values, the local dimension cannot exceed 13. A curved family can nevertheless require many linear PCA components for accurate approximation. Finding approximate compression is expected in principle; measuring the dimension needed at an explicit accuracy and diagnosing its limits is the substantive experiment. Do not call 99.9% recovered variance “99.9% of cosmological information.” No likelihood or Fisher-information preservation has been demonstrated.

## 5. Chapter 1: motivation, contribution and result preview

### Keep Introduction and Background separate

Keep Section 1 short, approximately 450-650 words, and Section 2 as a selective explanation, approximately 900-1,200 words. These are editorial targets, not competition limits. Section 1 answers **why this question, what this project contributes, and what was learned**. Section 2 gives readers the concepts needed to understand the experiment. Merging them would make the main question harder to find as the background expands.

If by “sections 1 and 2” you mean the first two subsections inside the background, keep their logical distinction but combine very short paragraphs into a single smooth subsection where helpful. Do not create a separate heading for every definition.

### Mention results in Chapter 1

Yes. Include one short result-preview paragraph near the end. The abstract must stand alone; the introduction should build the argument and then reveal its outcome. Some repeated scientific facts are appropriate. Repetition becomes distracting when the same workflow, numbers and caveats appear in the same sequence.

| Part | Its job | Include | Leave elsewhere |
|---|---|---|---|
| Abstract | Complete miniature of the study | Problem, method, central quantitative result, scope limit | Formula details, architecture widths, full split discussion |
| Introduction | Explain why the experiment is worth doing | Physical motivation, specific gap, question, contribution, one preview | Survey-by-survey history, full result table |
| Background | Equip the reader | Lensing observables, IA, NLA/TATT context, modelling tradeoff | Project results and training settings |
| Results | Present the evidence | Matched comparisons, plots, uncertainties and failures | Repeated general cosmology motivation |
| Conclusion | Answer the question | Supported answer, limitation, next decisive step | Repeating the entire abstract |

### Recommended paragraph sequence

1. **What is measured:** tiny correlations in galaxy shapes reveal how matter is distributed and how structure grows.
2. **Why interpretation is difficult:** galaxy formation also creates correlations, so accurate measurements need adequate astrophysical models.
3. **The modelling tradeoff:** flexible responses describe more possibilities but make it harder to represent and characterize uncertainty economically.
4. **The specific question:** within one documented synthetic response family, how many coordinates are needed at a stated reconstruction tolerance?
5. **What this study contributes:** a controlled family, a matched linear/nonlinear comparison, and a diagnosis of where compression succeeds or fails.
6. **Result preview and boundary:** two coordinates are insufficient under the stated criterion; six-coordinate candidates pass the average criterion, but difficult cases and survey-level implications remain open.

### Example opening and transition

The following is suggested wording for the student to adapt and verify. It assumes the current reproducible dimension study is selected for the report and its division of work is accurately stated elsewhere.

> The shapes of distant galaxies provide a way to study matter that we cannot see directly. Gravity slightly distorts these images as their light travels through the Universe. By measuring patterns across many galaxies, astronomers can investigate how matter clusters and how cosmic structure changes with time. The interpretation is difficult because galaxies can also acquire related orientations from the environments in which they form. These intrinsic alignments contribute to the same shape correlations used to measure weak gravitational lensing.
>
> A useful model must allow uncertainty in how alignment changes with distance and physical scale. However, a flexible family of possible responses can contain variations that are strongly related to one another. This raises a practical question: can those responses be described using fewer effective coordinates while retaining the features that distinguish them?
>
> This project investigates that question in a controlled synthetic model family. Thirteen parameters generate positive alignment-response surfaces, each evaluated at 3,131 grid points. We compare a linear method, principal component analysis, with nonlinear autoencoders on the same validation samples. The aim is to measure how reconstruction accuracy changes with the number of retained coordinates and to identify the surfaces that remain difficult to reconstruct.
>
> The saved validation experiments show that two learned coordinates improve on two-component PCA but do not reach our chosen 99.9% recovered-log-variance benchmark. Six-coordinate models reach that average benchmark, although localized errors remain substantially larger. This establishes a useful compression result for the selected model family and identifies the further checks needed before using such a representation in cosmological analysis.

### State innovation at the right level

Do not claim that using an autoencoder, PCA preprocessing, or neural networks in cosmology is itself new. CosmoPower already demonstrates neural emulation of cosmological spectra, including PCA-based representations; it addresses a different mapping and validates inference performance. This project encodes existing response surfaces into a bottleneck and tests their reconstruction. [CosmoPower](https://arxiv.org/abs/2106.03846).

Likewise, likelihood-aware compression can target parameter information directly. This report currently optimizes reconstruction of model surfaces; that is a different objective. [Alsing and Wandelt](https://arxiv.org/abs/1712.00012).

Potentially defensible contributions are the design and justification of this response family, the controlled comparison at fixed dimension, the measured accuracy-cost tradeoff, and an explanation of its difficult regimes. Establish which are original to the project and which were supplied by the supervisor. A focused literature search did not establish a first-ever claim for IA compression, and the report should not make one.

### Title and abstract

Recommended title: **Learning Compact Representations of Intrinsic-Alignment Response Surfaces**.

Possible subtitle: **A controlled comparison of principal components and autoencoders**.

If the student retains a direct-AE-only study, the current main title remains reasonable, but replace the two-number subtitle. “How many numbers are needed to reconstruct an alignment response?” poses the question without implying that two must succeed.

Write the abstract last. Use approximately two sentences for motivation, two for the experiment, two for the result, and one for the limitation. Include one matched comparison, such as six-coordinate AE versus rank-six PCA, and one tail metric. Keep the original amplitude equation, all 13 parameter ranges, the detailed split counts, and the VAE/flow discussion out of it.

## 6. Chapter 2: a selective and current scientific background

### 2.1 Cosmology, dark matter and dark energy

Begin with the questions cosmology asks: what the Universe contains, how it expands, and how structures form. Introduce the standard Lambda-CDM framework, where cold dark matter supports structure formation and a cosmological constant describes accelerated expansion. Separate the empirical success of this framework from the unknown microscopic identity of dark matter and physical origin of dark energy. A lensing map measures gravitating matter; it does not directly identify a dark-matter particle.

Explain that redshift is a measure of wavelength stretching and is used to relate observations to cosmic epoch. Define wavenumber as inverse spatial scale: larger k corresponds to smaller structures. These definitions are needed before the model equations. There is no need for a long history of the Big Bang or a catalogue of particle candidates. The existing Lecture1 and Lecture2 material provides an appropriate explanatory level; replace teaching-slide shorthand with precise connected prose.

### 2.2 DESI as motivation, with the 2026 update

Explain the dark-energy equation-of-state parameter as pressure divided by energy density, using a consistent convention. A cosmological constant has w = -1. The familiar evolving model is w(a) = w0 + wa(1-a). One equation and a short explanation are sufficient; no detailed dark-energy model taxonomy is needed.

The March 2025 DESI DR2 BAO analysis reported a preference for evolving dark energy of 2.8-4.2 sigma when combined with CMB and alternative supernova samples. It is a result of specified data combinations and model assumptions, not a measurement by DESI alone or an established discovery. [DESI DR2 BAO paper](https://arxiv.org/abs/2503.14738).

For a September 2026 report, add a brief update: the July 2026 DESI Ly-alpha full-shape results are compatible with the standard model and complicate the earlier picture. The June 2026 bispectrum analysis also finds that the inferred preference changes with the information and datasets included. Avoid portraying the field as a steady march toward a confirmed detection. [DESI DR2 Ly-alpha paper](https://arxiv.org/abs/2607.27410), [DESI full-shape/bispectrum analysis](https://arxiv.org/abs/2606.23936).

Suggested transition:

> Recent measurements have renewed interest in whether dark energy changes with time, but the interpretation depends on the observations and modelling assumptions used. Weak gravitational lensing supplies complementary information through the distribution and growth of matter. Making that comparison reliable requires careful treatment of effects that also influence measured galaxy shapes.

Describe lensing as a **complementary probe with different systematics**. It can provide an independent observational check when appropriately analysed, but adding lensing to a CMB/BAO combination does not make every resulting constraint statistically independent. The present compression experiment is preparation for better modelling, not a confirmation or rejection of the DESI interpretation.

### 2.3 From gravitational lensing to cosmological measurements

Use one simple diagram: background galaxies, foreground matter, observer, and distorted images. Explain the distinction between strong and weak lensing, then spend more space on what is measured than on the dramatic strong-lensing examples.

Describe the measurement sequence in plain language: detect galaxies; estimate their shapes; correct for the telescope/atmosphere and measurement response; estimate their redshift distributions; correlate shapes at different separations and in different redshift bins; compare those statistics with model predictions. “Averaging galaxy shapes” is a useful intuition, but the analysis needs calibration and correlations, not an unqualified average. [Weak gravitational lensing review](https://arxiv.org/abs/astro-ph/9912508).

Introduce shear as the coherent stretching component and convergence as a change related to projected mass. Explain the meaning of a two-point correlation before writing its notation. A schematic weak-shear relation, observed ellipticity approximately equals intrinsic ellipticity plus shear plus measurement noise, is enough if explicitly labelled schematic and calibration-dependent.

Then connect the observables to both geometry and growth. Redshift bins provide some distance information, but each bin receives lensing contributions from matter along a broad line of sight. This explains why several physical changes can produce similar observed correlations.

### 2.4 KiDS, DES, HSC and the status of S8

Define **S8 = sigma8 sqrt(Omega_m/0.3)**, with sigma8 describing the RMS matter-density contrast smoothed on 8 h^-1 Mpc and Omega_m the matter-density fraction. Explain that cosmic shear constrains this combination particularly well. The tension concerns comparisons of late-time measurements with CMB-based predictions under a cosmological model; it is not direct proof of new physics.

Use a small contextual table, not a catalogue of every survey result:

| Analysis | Reported S8 and 68% interval | Why it belongs in this background |
|---|---|---|
| KiDS-Legacy cosmic shear | 0.815 (+0.016, -0.021) | Agrees with Planck at 0.73 sigma; demonstrates the impact of calibration improvements |
| DES Y6 cosmic shear, NLA | 0.798 (+0.014, -0.015) | Current example of precision lensing and its modelling dependence |
| DES Y6 cosmic shear, TATT | 0.783 (+0.019, -0.015) | Same dataset with a different IA model; not an independent measurement |
| HSC Y3 cosmic shear, correlation functions | 0.769 (+0.031, -0.034) | Complementary deep imaging result with roughly 2-sigma difference from Planck |

Sources: [KiDS-Legacy](https://arxiv.org/abs/2503.19441), [DES Y6 cosmic shear](https://arxiv.org/abs/2602.10065), [HSC Y3 correlation functions](https://arxiv.org/abs/2304.00702). These intervals come from different analyses and should not be combined as independent Gaussian measurements. In a final table, preserve each paper's parameter-summary convention and model choice.

For KiDS-Legacy, the authors attribute the shift mainly to redshift-distribution estimation/calibration, together with new area and improved image reduction. Do not say that IA corrections alone resolved the tension. DES Y6 still reports about 2.0/2.3 sigma differences in S8 for NLA/TATT against its CMB comparison, even though multidimensional consistency is better. Thus, “the tension has been reduced in some recent analyses” is more accurate than “the tension has disappeared.” The exact degree depends on dataset, model and comparison statistic. The sources above support these distinctions.

The main lesson for this report is that modelling and calibration matter as measurement errors shrink. Keep dark-energy evolution and the S8 discussion as distinct motivations; the project's compression results solve neither observational question directly.

### 2.5 Intrinsic alignment: physical origin and observational status

Replace the heading “a false lensing pattern” with **Intrinsic alignment: an astrophysical contribution to galaxy-shape correlations**. IA is a real physical effect; it acts as contamination when the goal is to infer lensing alone.

Explain tidal fields as differences in gravitational pull across a galaxy and its environment. Introduce tidal alignment and tidal torquing as useful physical mechanisms, without assigning every elliptical galaxy perfectly to one and every spiral perfectly to the other. Alignment depends on galaxy population, mass/luminosity, environment and scale. Observations and simulations constrain parts of this behaviour, but a universal accurate prescription is not established. [Chisari's IA review](https://arxiv.org/abs/2510.15738).

Retain GG, GI and II, but show why GI need not involve two nearby source galaxies: a foreground galaxy can align with a tidal environment that also lenses a more distant one. In tomographic notation, the schematic signal contains GG + GI + IG + II; if the two cross terms are grouped as GI, say so explicitly. Include measurement noise where appropriate rather than treating it as part of IA.

### 2.6 NLA, TATT and the modelling tradeoff

| Model | Accessible explanation | Strength | Limitation |
|---|---|---|---|
| NLA | Relates alignment to the tidal field and uses a nonlinear matter spectrum in the two-point prescription | Simple, economical, widely used | Restricted response structure; not a complete nonlinear galaxy-formation theory |
| TATT | Adds tidal and density-weighting terms through a perturbative expansion | Captures additional types of alignment response | More modelling terms and nuisance freedom; validity still depends on scales and population |
| This project's family | A positive scalar response with adjustable scale and redshift factors | Controlled, interpretable compression benchmark | Phenomenological; not a replacement for the full TATT model |

For the underlying models cite [Bridle and King](https://arxiv.org/abs/0705.0166) and [Blazek and collaborators](https://arxiv.org/abs/1708.09247). A common NLA implementation varies an amplitude and redshift slope; commonly used TATT implementations allow additional amplitudes, evolution and weighting parameters. Counts depend on the analysis, so do not state a universal two-versus-five rule without identifying the exact implementation.

Avoid “TATT is always better.” More flexibility can reduce bias from a wrong restricted model, but may increase uncertainties or introduce poorly constrained directions. The value of that flexibility must be tested. The DES Y6 comparison already provides a concrete observational reason to take the tradeoff seriously.

Distinguish two meanings of projection: **line-of-sight projection**, which mixes physical scales/redshifts into observables, and **posterior marginalization**, where integrating over nuisance directions can change one-dimensional summaries through prior-volume effects. Neither effect is eliminated merely by relabelling parameters as latent coordinates.

End the background with the bounded question: can the variation of a documented flexible response family be represented compactly while retaining reconstruction accuracy? This leads directly to Section 3.

## 7. Chapters 3 and 4: strengthen the physical and methodological argument

### Chapter 3: model and dataset

**Add the missing bridge to physical spectra.** The existing `Power.ipynb` demonstrates the model's use of P_deltaI = A_IA P_nl and P_II = A_IA squared times P_nl. Include these relations with clear definitions and the adopted sign convention. Explain that this family links the spectra through the same scalar response; it cannot represent every independent IA contribution or all of TATT. No B-mode or general stochastic-alignment model is learned by the positive scalar surface.

**Explain each factor before the formula.** Introduce broad redshift evolution, a second smooth redshift transition, and a scale transition whose shape may evolve. “Luminosity-like” needs qualification: R_L is currently a redshift-dependent phenomenological factor, not a function of an observed luminosity catalogue. Use a name such as “additional redshift-transition factor,” or explain precisely the population effect it is intended to mimic.

**Show useful limiting cases.** These can provide a small, student-understandable analytical contribution:

| Limit or condition | Consequence in the adopted equations | Interpretation |
|---|---|---|
| q = 0 | S = 1 | No explicit scale response; its scale-shape parameters become irrelevant |
| k much smaller than k_t | S approaches 1 for positive transition exponent | Recovery of the large-scale baseline |
| k = k_t | S = 1 + q/2 | The correction at the transition is independent of n, alpha and m |
| z = z_star | R_z = R_L = 1 | Redshift factors share a normalization pivot, but A_Theta is still S(k,z_star) |
| alpha = 0 | Tail factor is unity | m and its evolution no longer affect that factor |

These are deductions from the current equations, not new numerical findings. The high-k asymptote also needs care: alpha describes the correction's tail; because S contains an added 1, the logarithmic slope of the full S is not always alpha. For negative alpha, the correction decays and S tends back toward 1.

**Justify the parameter ranges.** Group them by function and explain whether a range reflects mathematical stability, desired shape diversity, a literature scale, or a practical training choice. Finite and positive surfaces establish numerical validity, not agreement with nature. The 8,192 random draws and 8,192 corners are useful checks but do not prove all interior behaviour is realistic or bounded in the same way. Explain k_t's units and log-uniform distribution. Because q is nonnegative, this sampled scale factor satisfies S >= 1; it does not allow arbitrary suppression relative to its scale-independent baseline. Other factors can still make the full positive response smaller than unity.

The Sampling notebook records a failed initial prior check followed by a revised range choice. If that investigation belongs to the student's work, it can supply a concrete scientific decision: state what failed, which evidence motivated the change, and how that change restricted the model family. Avoid rewriting a supplied prior as a student discovery.

**Show data coverage and preserve the split contract.** An independent random split tests interpolation within the chosen sampling distribution. It does not establish extrapolation to unseen physical models or to survey-calibrated priors. Small k/z errors can be important despite being infrequent under the sampling distribution. If a sensitivity or boundary analysis is added, identify it separately from model selection. The saved generation, split and prior-validation seeds are 2027, 2028 and 2026; the shared training normalization scale is 0.27552554548. Record these actual provenance values rather than inferring them from filenames.

### Chapter 4: compare methods on equal terms

Rename this section **Compression methods and evaluation** if both PCA and AE are retained. Put PCA before the autoencoder so the baseline is established before the more flexible model.

Give a short reconstruction equation or diagram for PCA, and explain what its coefficients represent. Explain the distinction between the training eigenvalue spectrum and validation reconstruction variance. Use a common symbol such as **u** for latent coordinates, avoiding confusion with cosmological redshift **z**.

Preserve the explanation of Conv1D: redshift values form channels, while the kernel moves along log-wavenumber. This mixes redshifts through channel weights but does not exploit local redshift neighbourhoods in the way a 2D convolution would. Treat its usefulness as a design choice evaluated empirically, not proof that it is the physically optimal architecture.

Use the architecture and optimizer from each run's `ResolvedConfig.json`, the actual best epoch from its history/summary, and the actual stopping condition. Do not describe a 1,000-epoch maximum as 1,000 completed epochs. Capacity tiers change both convolutional and dense layers, so a depth comparison is also a capacity comparison.

Explain the metrics explicitly:

| Metric | Definition and meaning |
|---|---|
| Recovered log variance | 1 minus squared reconstruction error divided by squared distance from the **training mean surface**, summed over the validation grid and samples |
| Log10 RMSE | RMS difference between reconstructed and original log amplitudes |
| Physical relative error | `abs(10**(Y_hat - Y) - 1)`, where Y is the original log10 amplitude |
| p95 surface relative RMS | First compute relative RMS over each surface; then take the 95th percentile across surfaces |
| p99 surface maximum | First find each surface's largest relative error; then take the 99th percentile across surfaces |
| Global maximum | Largest relative error over every evaluated point in every validation surface |

The 99.9% recovered-variance threshold is the project's numerical benchmark. It is not a Euclid/DES/LSST requirement. Explain why it is useful for organizing this comparison, then retain tail diagnostics without retroactively inventing a pass condition that favours a selected result.

Training-only estimation prevents normalization leakage. The use of a global scale additionally defines the relative weighting of features; a per-feature scale estimated on training data would not itself be leakage. Correct the current wording that blends those two points.

If PCA-AE becomes a main method, add a compact diagram: surface -> 30 PCA coefficients -> short nonlinear code -> reconstructed coefficients -> surface. Its rank-30 projection residual is part of the final error and must be included. The two neural families have different epoch budgets, so equal latent dimension is not equal training compute. Show model-state cost separately and measure runtime before making speed claims.

## 8. Chapters 5 and 6: make the evidence the centre of the report

### Recommended scope of the results chapter

Use PCA plus one reproducible Conv1D capacity tier as the minimum coherent main study. It supplies all five latent dimensions and is straightforward to explain. Add PCA-AE if the student can explain the frozen basis, coefficient loss and model-state tradeoff. A method should enter the report because it answers the question and belongs to the documented work, not merely because it exists in the supervisor's repository.

Place incomplete Conv2D coverage and broader capacity sweeps in supplementary material or an explicit coverage table. Do not claim that Conv1D outperforms Conv2D when the eligible comparison is incomplete. A smaller prespecified scope is acceptable; describe the broader exploration honestly if it influenced the chosen scope.

Recommended sequence:

1. Establish how PCA accuracy varies with rank.
2. Compare PCA and the chosen neural method at dimensions 2, 4, 6, 8 and 10.
3. Identify the smallest **evaluated** neural dimension satisfying the stated numerical criterion; do not call it the exact intrinsic dimension or a proven minimum over all architectures.
4. Show paired reconstructions and locate the remaining errors.
5. Discuss model size, robustness and limitations without turning them into a catalogue of future methods.

### Figure plan

| Figure | Scientific purpose | Existing starting point |
|---|---|---|
| 1. Lensing and IA geometry | Explain what is observed and why IA enters | Replace the current text-arrow diagram with a clearer schematic |
| 2. Model factors and representative surfaces | Relate parameter changes to recognizable shapes | Formula and Sampling notebooks; reduce redundant panels |
| 3. Accuracy versus retained dimension | Answer the central research question | `Figure/NLA/ModelSelection/Overall_Validation_Performance.pdf` |
| 4. Original, reconstruction and residual | Let readers see what “accurate” means | Selected validation diagnostics and worst-reconstruction figures |
| 5. Distribution or map of difficult cases | Explain the limits of the compact representation | Tail and validation-diagnostic figures |
| Optional 6. Accuracy versus model-state size | Explain why PCA-AE might be useful | `Validation_Accuracy_Complexity.pdf` |

Use the same validation surfaces for paired method comparisons. Select a typical case by a stated rule and a difficult case by a stated error criterion; do not choose only attractive reconstructions. Include a signed residual or log residual beside the absolute relative error when it clarifies the direction of failure. Report each displayed sample's dataset index and its selection rule in a caption or appendix.

Do not simply shrink a large notebook figure into the report. Simplify panels, enlarge type, use accessible colours, identify train versus validation, and express percentages consistently. A rank plot of maximum relative error need not decrease monotonically even when PCA's squared log-space reconstruction error does; the metrics optimize different properties.

### Discuss the limits with specificity

The main limitations are the selected phenomenological model family and sampling distribution; numerical reconstruction rather than likelihood accuracy; single-seed evidence; incomplete explored coverage; and localized errors. These are more informative than repeated generic statements that “more work is needed.”

Good latent-parameter correlations are descriptive, not proof that a latent coordinate equals a physical parameter or that parameter information has been retained. Latent coordinates can rotate or be reparameterized. Use reconstruction along meaningful input-parameter changes to support interpretation; arbitrary decoder traversals can leave the distribution of valid encoded surfaces.

### Chapter 6 should answer, not restart

Aim for approximately 250-350 words: the research question, the supported result, its most important limitation, and the next decisive test. Move the detailed future-work list out of Section 5 so it is not repeated here. Prefer “Six-coordinate candidates passed our validation benchmark within the sampled family” to “The model needs only six parameters.”

## 9. Additional work ranked by scientific value and feasibility

### A. Required now: evidence reconciliation and reproducible figures

**Question resolved:** Are all reported numbers attributable to the same experiment and evaluation contract?

Create a small report-result ledger linking each table row and figure to its run, checkpoint, data/PCA version and split. Replace all historical headline values together. Reuse existing diagnostics where possible. This is primarily analysis and editorial work, not a new training programme. Acceptance: every reported number has a traceable artifact, and abstract, methods, results and appendix agree.

### B. High value: student-led investigation of difficult surfaces

**Question resolved:** What features make some surfaces harder to compress?

Use saved validation diagnostics to compare errors with physically meaningful features: transition location, sharpness, redshift evolution, dynamic range and curvature. Start with a few hypotheses, such as whether moving transitions or sharper changes are difficult. Show the actual result even if the hypothesis fails. Account for correlated parameters and distinguish exploratory patterns from causal explanations.

Deliverable: one error map or carefully selected comparison and a paragraph explaining what it reveals. This builds naturally on the homework about bottleneck capacity and weighted losses, and is more valuable than adding another fashionable architecture.

### C. Valuable if time and compute permit: a small seed study

**Question resolved:** Does the headline advantage depend strongly on one optimization seed?

Choose the scope using validation only, then repeat a small prespecified set of promising configurations with the same data, objective and budgets. At least three total seeds gives a first view of training variability, not a precise population uncertainty. Retain per-seed metrics and summarize their spread. A bootstrap over validation surfaces can quantify sampling variation conditional on one trained network, but it is not a substitute for retraining across seeds.

Gate: do not delay the submission revision or open the test set simply to make the study look final. If replicates cannot finish, label the single-seed conclusion and remove claims of statistically robust superiority between close neural candidates.

### D. Strong physics extension: propagate amplitude errors to IA spectra

**Question resolved:** How does reconstruction error affect the quantities to which the amplitude is applied?

For the adopted factorization, hold A0, A_Omega and P_nl fixed, and define the signed fractional response error as epsilon = reconstructed A_Theta / A_Theta - 1. Where the relevant spectra are nonzero, the fractional P_deltaI error is epsilon and the fractional P_II error is 2 epsilon + epsilon squared. This follows directly from the linear and quadratic amplitude dependence. The absolute-error version must be handled carefully near zero full amplitude.

Deliverable: a short derivation and, if added to the analysis, an example using saved validation reconstructions. This is a direct physical connection available without building a full cosmological inference pipeline. It still does not determine the error in a projected shear correlation or in cosmological parameters.

### E. Optional after submission: observable projection and inference validation

Specify source redshift distributions, cosmology, angular scales and the adopted IA prescription. Project true and reconstructed spectra into the same shear data vector. Report absolute residuals and, if a defensible covariance is available, a covariance-weighted discrepancy. A toy example must be labelled a toy example; do not invent a survey error budget or claim cosmological-bias control from an arbitrary weight function.

Only after this step would an inference study establish whether compression biases parameter constraints or accelerates a full analysis. A freely sampled latent box also needs a justified prior: the distribution induced by the original 13-parameter sampling is not generally uniform or Gaussian. Dimensionality reduction does not by itself remove degeneracies or make marginalization valid.

### F. Optional analytical extension: why is the family compressible?

Explore the singular values of a local finite-difference Jacobian of the log surface with respect to the 13 parameters, scaled consistently with their ranges. Compare local parameter sensitivity with global PCA requirements. Check finite-difference step stability and representative interior/boundary points. This can distinguish weak parameter sensitivity from curvature of the global family, but is not necessary before September 15.

A simpler alternative is a prespecified ablation: fix the redshift-evolution parameters and repeat a limited compression comparison. Changing the family also changes its difficulty, so evaluate each family consistently and do not treat better performance on a simpler family as an improvement on the original benchmark.

## 10. Source organization, bibliography and contributions

### Rename Section7 and Section8

Yes. They are source containers for appendices and acknowledgements, not numbered Sections 7 and 8. The files currently use the `.tex` extension.

| Current file | Recommended name | Contents |
|---|---|---|
| `sections/Section7.tex` | `sections/Appendices.tex` | Reproducibility checklist and implementation notes, currently Appendices A and B |
| `sections/Section8.tex` | `sections/Acknowledgements.tex` | Unnumbered acknowledgements and contributions |

Update the two `\input` statements in `main.tex` and any README references in the same change. Keep `\appendix` at one clear boundary. A later optional cleanup can rename Sections 1-6 descriptively, but it is not needed for scientific improvement. No renames were performed during this assessment.

### Restore a small, purposeful bibliography

Use a single `references.bib` or another consistent bibliography mechanism. Start the bibliography on its own page and cite borrowed methods and factual claims where they are introduced. Avoid padding it with papers the student has not read. Suggested reading responsibilities are:

| Topic | Starting reference | What the student should extract |
|---|---|---|
| Lensing measurement | [Bartelmann and Schneider](https://arxiv.org/abs/astro-ph/9912508) | Observable, shear, redshift bins and statistical measurement |
| Current IA understanding | [Chisari](https://arxiv.org/abs/2510.15738) | Mechanisms, observational limits and model scope |
| NLA | [Bridle and King](https://arxiv.org/abs/0705.0166) | What the approximation represents |
| TATT | [Blazek et al.](https://arxiv.org/abs/1708.09247) | Additional physical terms and why flexibility matters |
| Current shear cosmology | [KiDS-Legacy](https://arxiv.org/abs/2503.19441), [DES Y6](https://arxiv.org/abs/2602.10065), [HSC Y3](https://arxiv.org/abs/2304.00702) | Dataset/model-specific status rather than a universal tension claim |
| Dark-energy motivation | [DESI DR2 BAO](https://arxiv.org/abs/2503.14738), [2026 Ly-alpha result](https://arxiv.org/abs/2607.27410) | Evidence, assumptions and why confirmation remains needed |
| Related scientific ML | [CosmoPower](https://arxiv.org/abs/2106.03846) | Prior work and the distinction between emulation and this encoding experiment |
| Reconstruction versus information | [Alsing and Wandelt](https://arxiv.org/abs/1712.00012) | Why low reconstruction error is not a proof of retained cosmological information |

Add primary references for PCA and autoencoders and cite the numerical packages actually used, after checking their appropriate citation entries. The table above is a targeted reading queue, not a completed exhaustive novelty review. Some advanced papers should be read in selected sections with supervision; the student need not reproduce their full formalism in this report.

### Complete contributions without inference from file ownership

For each research stage, record what the student actually decided, implemented, ran, checked, interpreted or wrote, and what was supplied or corrected by others. A repository commit author, notebook output or polished paragraph alone cannot establish that division of work.

Use concrete examples: a hypothesis about failed reconstructions, a comparison the student designed, a mistaken interpretation corrected through evidence, and a result the student can reproduce. Include unsuccessful attempts when they explain a real scientific decision. Do not invent a personal research story from the course outline, and do not attribute the entire professional codebase to the student.

The report should sound like a capable student who understands the work. Explain unfamiliar terms and use direct sentences; do not remove scientific rigor or insert advanced terminology solely to impress judges. Ask the student to explain every main equation, figure and choice aloud before treating that material as ready.

## 11. Editing sequence and completion gates

### Submission-focused schedule

The schedule below is a recommendation, not an estimate of training completion time.

| Date | Work | Concrete completion condition |
|---|---|---|
| September 10 | Confirm category/portal details; choose report scope; reconcile artifacts and contribution boundaries | A fixed list of reported runs and claims, with remaining unknowns recorded |
| September 11 | Update Methods and Results together; prepare central comparison and reconstruction figures | Every numerical statement matches its artifacts and captions |
| September 12 | Rewrite Introduction and Background; add citations and the physical bridge | A reader outside astronomy can follow observation -> IA -> representation question |
| September 13 | Complete failure-case interpretation, limitations, contributions and bibliography; write abstract last | No historical/current experiment mixing or unsupported claims remain |
| September 14 | Fresh build, full-page visual inspection, citation/table check, student oral walkthrough | Complete readable PDF with no placeholders and a traceable result package |
| September 15 | Submission buffer, after verifying portal cutoff | Upload checked files and verify receipt through the user's submission workflow |

This review does not submit anything or contact the organizer. Submission is a separate action. If time becomes tight, prioritize traceability, the main dimension study, one reconstruction figure, citations and truthful contribution information. A new VAE, flow or broad Conv2D sweep is not on the critical path.

### Gate 1: evidence and scope

Record dataset identity, coordinate order, target, transformation, split identity, PCA version, each reported run, checkpoint, seed and actual completed epoch. Separate the code version that produced a run from the report's current working-tree revision. Document the confirmed historical test diagnostics. If an independent final assessment is claimed, reserve an independently generated final holdout under a fixed protocol; do not relabel already inspected samples as untouched.

Also correct reproducibility instructions: the current result notebooks require ModelSelection before AE and PCA_AE, while the README lists the reverse order. That selection step currently requires resolution of same-seed duplicate runs. Environment YAML files are intended specifications, not proof of the versions that produced existing artifacts; for example, saved PCA output lists NumPy 2.3.5 while Appendix A claims 2.0. Report verified run-time versions or state that historical environment details are unavailable.

### Gate 2: scientific consistency

The abstract, title, question, architecture table, results, figure captions and conclusion must describe the same study. Every superiority claim specifies the metric and comparison dimension. Every adequacy claim identifies its numerical criterion and sampled domain. Claims about speed, likelihoods, survey requirements, physical information or parameter recovery require separate evidence.

### Gate 3: student understanding and authorship

The student can explain why the model was chosen; what its 13 parameters do; why compress when those parameters already exist; why PCA is a fair baseline; how the splits are used; what recovered variance measures; why a large local error can coexist with a good average; and what the project has not established. A truthful account of assistance accompanies that understanding.

### Gate 4: presentation and submission

Build from a complete fresh source snapshot. Inspect every page at readable size. Check figure provenance, axis units, caption definitions, table rounding, cross-references, bibliography, names and date. Remove every visible placeholder and unresolved drafting instruction. Preserve a compact reproducibility package with result tables and instructions, subject to the actual submission rules.

## 12. Decisions to carry into the revision

Keep Chapters 1 and 2 separate; give Chapter 1 a short result preview. Make the background selective and current, including the developments after the initial DESI DR2 announcement. Present IA as a real astrophysical signal whose modelling affects lensing inference. Explain NLA and TATT fairly, without claiming that the current scalar response implements the full latter model.

Most importantly, move from an isolated two-coordinate demonstration to a reproducible investigation of accuracy, dimension and difficult cases. Preserve the distinction between compact reconstruction of the chosen synthetic family and successful cosmological inference. Rename the appendix and acknowledgement files for clarity, restore references, and make the student's actual contribution visible. These changes would make the report more convincing because its question, evidence and claimed contribution would finally match.

## Evidence index for implementing the plan

These links identify the main local sources; the review has not rewritten them.

- [Report assembly and abstract](/Users/s2227120/IAFlowCloud/Report/main.tex:85), [historical results](/Users/s2227120/IAFlowCloud/Report/sections/Section5.tex:33), [architecture description](/Users/s2227120/IAFlowCloud/Report/sections/Section4.tex:62), [reproducibility appendix](/Users/s2227120/IAFlowCloud/Report/sections/Section7.tex:3), [contribution placeholders](/Users/s2227120/IAFlowCloud/Report/sections/Section8.tex:9).
- [Model implementation](/Users/s2227120/IAFlowCloud/Code/ia_models/nla/model.py:495), [sampling implementation](/Users/s2227120/IAFlowCloud/Code/ia_models/nla/sampling.py:73), [reconstruction metric definition](/Users/s2227120/IAFlowCloud/Code/iaflow/core/metrics.py:211), [data preparation](/Users/s2227120/IAFlowCloud/Code/iaflow/core/data.py:333).
- [Saved September 2 selection](/Users/s2227120/IAFlowCloud/Data/NLA/Comparison/ValidationSelectionLive.json), [saved comparison table](/Users/s2227120/IAFlowCloud/Data/NLA/Comparison/ValidationResults.csv), [PCA benchmark](/Users/s2227120/IAFlowCloud/Data/NLA/PCA/PCAValidationMetrics.json), [PCA metadata](/Users/s2227120/IAFlowCloud/Data/NLA/PCA/PCATransformMetadata.json).
- [Six-coordinate direct-AE metrics](/Users/s2227120/IAFlowCloud/Runs/NLA/AE/Conv1D/Depth03/Latent06/20260823-103003/ValidationMetrics.json), [six-coordinate PCA-AE metrics](/Users/s2227120/IAFlowCloud/Runs/NLA/PCA_AE/Depth03/Latent06/20260822-192700/ValidationMetrics.json).
- [Newer Depth05 L=10 metrics](/Users/s2227120/IAFlowCloud/Runs/NLA/AE/Conv1D/Depth05/Latent10/20260830-115056/ValidationMetrics.json), [September 9 L=2 repeat](/Users/s2227120/IAFlowCloud/Runs/NLA/AE/Conv1D/Depth03/Latent02/20260909-003908/ValidationMetrics.json), [September 9 L=6 repeat](/Users/s2227120/IAFlowCloud/Runs/NLA/AE/Conv1D/Depth03/Latent06/20260909-003908/ValidationMetrics.json).
- [Selection notebook and duplicate-seed policy](/Users/s2227120/IAFlowCloud/Notebooks/NLA/ModelSelection.ipynb:756), [PCA rank-curve source](/Users/s2227120/IAFlowCloud/Notebooks/NLA/PCA.ipynb:998), [physical spectrum connection](/Users/s2227120/IAFlowCloud/Notebooks/NLA/Power.ipynb:589), [prior investigation](/Users/s2227120/IAFlowCloud/Notebooks/NLA/Sampling.ipynb:460).
- Historical test evidence is reproducible with `git show ca1bed7:Code/PCA.ipynb`; inspect saved cells 15 and 20-22. This reads the historical notebook, not the test dataset.
