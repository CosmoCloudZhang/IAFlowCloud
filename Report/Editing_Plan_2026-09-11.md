# IAFlow report: revised editing plan

Prepared 11 September 2026 | Title, headings, and scientific tone

This plan incorporates the agreed direction: preserve the accessible progression of ideas, use precise scientific terminology, and remove conversational framing that understates the investigation. It proposes the next revision of the title and Sections 1-2. The report itself has not been edited in this planning pass.

## 1. Editorial decision and scope

Write for a scientifically literate reader who may be unfamiliar with cosmology. The report should communicate a substantive research investigation through direct explanations, appropriate terminology, and carefully bounded conclusions. The student's level of understanding should be evident in the reasoning and interpretation. There is no need to make the language deliberately elementary.

The Physics assessment criteria emphasize originality, methodological creativity, rigorous reasoning, physical contribution, and clear scholarly writing. They distinguish established background from original contributions. The recommendation here is an editorial interpretation of those criteria, not a claim that the judges require one particular title style. [Official Physics assessment criteria](https://www.yau-science-awards.org/competitioncategory/physics-assessment.html)

The next manuscript pass should cover:

- The main title and subtitle in `main.tex`.
- The Section 1 introduction: its opening, statement of the problem, contribution, and result preview.
- Section 2 headings, transitions, and selected sentences that need a more precise register.
- Corresponding Section 2 captions and the names used in the student figure guide.

Preserve the abstract, Sections 3-8, numerical results, existing figure assets, and the established scientific scope during that pass. Later-section recommendations are recorded separately below. This plan does not authorize a new experiment, promote a different saved run to the headline result, or resolve the outstanding scientific issues through wording alone.

## 2. Recommended title and subtitle

**Main title:**

> Compact Representations of Intrinsic-Alignment Responses for Weak-Lensing Cosmology

**Preferred subtitle:**

> A flexible scale-redshift model with linear and nonlinear reconstruction

The main title makes one small change to the user's preferred wording: replace **Low-Dimensional** with **Compact**. This makes the opening less dense while retaining the physical quantity and cosmological motivation. Explain in the introduction that compactness refers to the number of retained coordinates. It does not establish a small network, fast cosmological inference, or reduced uncertainty in cosmological parameters.

Retain **responses**. The experiment reconstructs the model's scale- and redshift-dependent response, rather than measuring galaxy orientations directly or demonstrating a general replacement for every intrinsic-alignment model. Retain **for weak-lensing cosmology** to make the scientific purpose visible, while stating in the abstract and introduction that the current tests concern synthetic-response reconstruction.

The preferred subtitle gives prominence to the flexible model as well as the reconstruction experiment. It is shorter than a list of model construction, dataset generation, network architecture, and diagnostics. The first methods paragraph should identify the linear method as principal component analysis and the nonlinear method as a convolutional autoencoder. Using both method names in the subtitle is optional; their exact names must appear early in the introduction.

If explicitly naming the algorithms on the cover is preferred, use this alternative subtitle:

> A flexible response model with principal component analysis and convolutional autoencoder reconstruction

Use only one subtitle in the report. The shorter version is the recommendation because it reduces the combined title's density without reverting to phrases such as “fewer numbers” or presenting the work as only an algorithm comparison.

### What the title should and should not imply

The combination of title and subtitle should communicate three points: an astrophysical modelling problem motivates the work; a flexible response family supplies the controlled setting; and linear and nonlinear representations are assessed through reconstruction.

Do not add “novel,” “universal,” “physics-informed,” “unbiased,” or “efficient cosmological inference” without evidence for the specific claim. The existing physical prescription does not, by itself, make the autoencoder a physics-informed learning method. Good reconstruction does not establish posterior accuracy or preservation of all cosmological information.

### Cover layout

Use title case for the main title and sentence case for the subtitle. Keep normal scientific hyphenation; there is no need to remove it from established modifiers. Allow the main title to wrap naturally over two or three balanced lines and the subtitle over one or two lines. Avoid a final line containing only “Cosmology.” Check the repeated title above the abstract as well as the cover. Do not reduce the font until the text becomes secondary to surrounding metadata merely to force a single line.

## 3. Updated heading structure

Use concise descriptive headings. The main sections do not need additional subtitles. Keep LaTeX labels unchanged so existing cross-references continue to work.

| Location | Current heading | Recommended heading |
|---|---|---|
| 1 | Introduction | Introduction |
| 2 | Scientific background | Cosmological background and intrinsic alignment |
| 2.1 | An expanding Universe and its contents | Cosmic expansion and the standard cosmological model |
| 2.2 | Is dark energy constant? | Observational constraints on dark energy |
| 2.3 | Weak lensing and the distribution of matter | Weak gravitational lensing and matter clustering |
| 2.4 | What galaxy surveys tell us, and what comes next | Cosmic-shear surveys and the S8 tension |
| 2.5 | Intrinsic alignment: how environments affect galaxy shapes | Intrinsic alignment and galaxy-shape correlations |
| 2.6 | The modelling tradeoff and our compression question | Intrinsic-alignment models and dimensionality reduction |

In LaTeX, write S8 as `$S_8$`. These titles identify the subject without requiring the reader to infer it from a rhetorical question. A question can still appear in the introduction if it states a specific, testable research problem. The change concerns the accumulated tutorial tone, not a prohibition on interrogative sentences.

“Scientific background,” the current 2.1 heading, and the current 2.3 heading are not scientifically incorrect. Their replacements improve consistency and specificity. The strongest need for revision is in 2.4 and 2.6, whose current wording does not identify the scientific content as directly.

## 4. Language and terminology policy

Use familiar sentence structures while retaining the terms needed to describe the work accurately. Define a specialist term when it becomes necessary, then use it consistently. Keep the explanation “small k corresponds to large spatial scales,” for example: it is clear and scientifically useful.

| Prefer in the report | Explain on first use | Avoid as repeated scientific shorthand |
|---|---|---|
| Intrinsic alignment | Correlations in galaxy orientations associated with their formation and environment | Related directions; a false lensing pattern |
| Response surface | A model response tabulated across scale and redshift | Shape or surface without identifying the quantity |
| Dimensionality reduction | Representing the response with fewer retained coordinates | Fewer numbers as the main methodological description |
| Latent coordinates; latent dimension | The values in the compressed representation and their number | Two learned numbers throughout the results discussion |
| Reconstruction error | A difference between the original and reconstructed responses, measured by a stated metric | How well the method works |
| Parameter degeneracy | Different parameter choices producing similar predictions | Parameters getting mixed up |
| Complementary observational probe | A measurement with different sensitivity and sources of uncertainty | An independent confirmation without qualifications |

Use “we” for concrete actions taken in the reported study. Active constructions such as “We generate,” “We compare,” and “We evaluate” are appropriate. Removing first-person language or adding long passive sentences would not make the work more rigorous.

Distinguish the positive response A_Theta(k,z), its logarithm used for compression, the signed full alignment amplitude, and the matter power spectrum. The title can remain readable; the definitions in the body must establish these distinctions.

Limitations should identify the measurement or claim that remains untested. State the scope clearly at the end of the introduction and at the transition from background to methods. Avoid repeating the same general disclaimer in every paragraph, but retain qualifications beside quantitative or potentially misleading claims.

## 5. Section 1: paragraph-level editing plan

Keep Introduction and Background separate. The introduction should establish the reason for the investigation and its specific question before the reader enters the longer cosmology explanation. An approximate length of 450-650 words remains sensible; this is an editorial guide, not a competition requirement.

### Paragraph 1: identify the observable and scientific purpose

Replace the opening rhetorical question with a direct statement about weak lensing. Retain the explanation that individual galaxy shapes are noisy and that correlations across many galaxies reveal statistical information about matter.

Example of the intended register:

> Weak gravitational lensing probes the distribution of matter through small distortions in the observed shapes of distant galaxies. Correlations measured across many galaxies provide information about matter clustering and the growth of cosmic structure.

This is an illustrative sentence revision, not a replacement for the student's complete introduction. Keep the existing weak-lensing citation attached to the physical explanation.

### Paragraph 2: introduce the modelling problem

Replace “There is a complication” with an explicit description of the additional contribution. Explain that intrinsic alignment is a physical signal that must be modelled when interpreting galaxy-shape correlations for lensing cosmology.

Example:

> Intrinsic galaxy alignments contribute additional correlations to the observed shape signal. Their scale and redshift dependence must therefore be accounted for when using weak lensing to infer cosmological parameters.

The paragraph should establish why the problem matters without suggesting that this project has already corrected a survey measurement.

### Paragraph 3: motivate dimensionality reduction

Retain the current reasoning: model flexibility allows more possible responses; some variations may be closely related; a shorter representation may approximate those variations at useful accuracy. Name dimensionality reduction and distinguish its goal from parameter inference.

Make the key motivation explicit: the original 13 parameters already generate the full grid. The experiment tests whether the response family admits an even shorter approximate representation. It does not start from 3,131 independent physical degrees of freedom. Do not claim that changing coordinates automatically removes nuisance-parameter degeneracies or reduces computation in a full cosmological analysis.

### Paragraph 4: define the experiment and its question

Identify the synthetic response family, its scale-redshift grid, and the relationship between model parameters and retained coordinates. Replace repeated “two numbers” wording with a defined two-dimensional representation.

For the currently reported pilot, the question can be written as:

> We test whether a two-dimensional autoencoder representation reconstructs the chosen intrinsic-alignment responses more accurately than two-component principal component analysis on the same validation samples.

This preserves the present report's experiment. Do not silently replace it with a completed dimension-selection study while Sections 3-6 still describe the historical two-coordinate pilot. A broader question about accuracy versus dimension is appropriate only when the methods, results, figures, and abstract are revised together.

### Paragraph 5: describe the contribution at the right level

Present the investigation as a connected sequence: specification of the flexible response family; generation and checking of synthetic surfaces; comparison of linear and nonlinear representations under matched conditions; and assessment of reconstruction accuracy and difficult cases.

Explain what is being investigated without claiming that PCA or autoencoders are new. Retain the methodological and related-work citations. Distinguish the report's scope from the student's individual contributions; authorship must follow the actual division of work, not what the title sounds like.

### Paragraph 6: preview the result and state its limit

Keep a short result preview. Describe the direction of the reported comparison and the accuracy limitation without repeating the abstract's full list of numerical values.

Suitable wording for the current scope:

> In the reported preliminary comparison, the autoencoder achieves lower reconstruction errors than PCA at the same latent dimension. The remaining errors show that this improvement alone does not establish sufficient accuracy for cosmological applications.

Retain “reported preliminary comparison” until the historical result's provenance is reconciled. A tone revision should not turn this into a newly verified result or a general claim of nonlinear superiority.

## 6. Section 2: retain the scientific sequence, refine the delivery

Keep the six-subsection structure and the explanation of the density field in 2.3. The earlier review's 900-1,200-word background target predates the request for six substantive subsections and six figures; it should no longer govern the revision. A working target of roughly 1,600-2,200 words, excluding captions, is more realistic, with completeness and clarity taking priority over an arbitrary cutoff.

### 2.1 Cosmic expansion and the standard cosmological model

Retain the hot Big Bang, cosmic expansion, the scale factor, cosmological redshift, the CMB, and the distinction between ordinary matter, dark matter, and dark energy. State what is observed and what is inferred within Lambda-CDM. Keep the physical meaning of each symbol alongside the redshift equation.

Use direct scientific statements rather than an invitation to “explore the mysteries” of the Universe. The current explanations of the CMB and expansion are largely suitable; this subsection needs selective polishing. Preserve the recent clarification that CDM means cold dark matter and that the scale factor tracks cosmic expansion. Do not replace these improvements with wording from an older draft.

The Planck figure should connect the observed temperature map, the approximate present-day composition, and expansion/redshift. Its caption must distinguish ancient radiation from a present-day matter map.

### 2.2 Observational constraints on dark energy

Retain the sequence from distance measurements to BAO, the product H0 rd, and the evolving equation-of-state parameterization. Explain the physical purpose of the parameters before discussing contours or significance. Keep the stated data-combination and model qualifications attached to the DESI result.

Remove broad phrases such as “the wider picture is still developing” where a precise statement of dataset dependence can do the work. The tone pass should preserve the existing primary references and scientific qualifications rather than add new numerical claims.

The final paragraph should explain why weak lensing is useful as a complementary probe. It should not imply that this report independently confirms dark-energy evolution. The two DESI panels remain the figure plan; preserve their actual axis definitions and analysis assumptions.

### 2.3 Weak gravitational lensing and matter clustering

Keep the sequence: light deflection; convergence and shear; density contrast; matter power spectrum and wavenumber; statistical measurement of galaxy-shape correlations; connection to geometry and structure growth.

Use physical explanations before formal definitions. Retain the distinction between a spatial density wavelength and the wavelength of light introduced in 2.1. Explain that the matter power spectrum summarizes two-point statistics, while lensing combines contributions over a range of distances and scales.

There is no need to add a Fourier-transform derivation or the full lensing integral solely to make the section look more advanced. The density and scale inset belongs with the lensing diagram, so the reader can connect k to the later response surfaces.

### 2.4 Cosmic-shear surveys and the S8 tension

Organize the discussion around the observable and what successive analyses teach about measurement and modelling. Define S8 and explain why it is useful before listing survey results. Retain the current distinction between Stage III measurements and Stage IV prospects.

Keep KiDS, DES, and HSC comparisons tied to their analysis choices. Different IA models applied to the same observations are alternative analyses, not independent measurements. Changes in calibration or priors should not be presented as effects of more data alone.

Replace “what comes next” with a concise account of how Euclid, Rubin, and Roman motivate tighter control of systematic uncertainties. Do not add numerical forecasts or launch schedules during this language pass. The figure should separate measured contours from qualitative prospects.

### 2.5 Intrinsic alignment and galaxy-shape correlations

Retain tidal fields, the influence of environment on galaxy shapes, and the schematic decomposition of measured ellipticity. Keep the reduced-shear and shape-response qualifications concise and adjacent to the expression they qualify.

Introduce GG, II, and the two GI/IG cross terms through their physical connections. Continue to explain why a foreground intrinsic orientation can correlate with the shear of a more distant source. This is essential background for the modelling problem, not optional technical decoration.

Prefer “additional astrophysical contribution” to “false lensing pattern.” Preserve the separation between the correlated signal and random shape or measurement noise. Use the existing three-geometry student diagram rather than adding more equations to demonstrate sophistication.

### 2.6 Intrinsic-alignment models and dimensionality reduction

Present NLA and TATT as model choices with different capabilities and limitations. Discuss flexibility, computation, parameter degeneracies, and the role of uncertain redshifts. Identify line-of-sight projection separately from marginalization over nuisance parameters.

End with the motivation for this particular response family and the reconstruction question. Replace vague phrases such as “this first step” with the specific experiment being introduced. Keep the distinction between the positive NLA-inspired response and a complete physical TATT prescription.

The flow chart should give balanced model tradeoffs and show the chain from response-family construction to compact reconstruction. Retain a clearly separated future requirement for observable and inference validation. The figure should not imply that the autoencoder has already resolved the modelling tradeoff.

## 7. Examples for a controlled sentence-level pass

These examples specify the intended register. Adapt them to the surrounding paragraph and retain the citations supporting the claims.

| Current wording | Proposed direction |
|---|---|
| “How can we study matter that we cannot see?” | Begin with the weak-lensing observable and its scientific purpose. |
| “There is a complication” | Identify intrinsic alignment as an additional contribution to measured shape correlations. |
| “Two learned numbers can improve on two PCA coefficients” | State which reconstruction metric improves at equal latent dimension in the reported comparison. |
| “The matter doing the lensing is not spread uniformly” | “The matter distribution is inhomogeneous, with clusters, filaments, and underdense regions.” |
| “The wider picture is still developing” | State how the interpretation depends on the observations and model assumptions. |
| “This makes the limits of compression part of the result” | “The reconstruction errors also identify limitations of the two-dimensional representation.” |
| “Contours must retain their stated model and probability levels” inside a caption | Describe the actual plotted datasets, model, interval levels, and source; move instructions to the figure guide. |

Do not perform a mechanical word substitution across the report. For example, “compact” is suitable for the title, but “low-dimensional” is more precise when discussing a numerical latent dimension. Similarly, “surface” is useful after its quantity and axes have been defined. Retain short explanatory sentences and remove ambiguity, rather than replacing every ordinary word.

## 8. Captions, references, and the student figure guide

Keep the detailed figure instructions in the separate [Section 2 figure guide](/Users/s2227120/IAFlowCloud/Report/Section2_Figure_Guide.md). Its direct instructional tone is useful for the student and does not need to match the research report's prose. Synchronize its subsection names when the manuscript headings change; retain its filenames, source links, and technical plotting checks.

There are six background figure slots in the current source. Their captions are drafts. For each completed figure, the final caption should identify the subject, explain panels and symbols, state the data/model or schematic status, and give the source credit. Remove instructions such as “Contours must retain...” from the submitted caption after the final plotting choices are known. Keep them in the guide until the figure has been checked.

Preserve the six existing background filenames and LaTeX labels. In particular, retain `fig:ia-clue` unless a coordinated reference update is explicitly needed. Do not replace the student's existing result figures as part of a tone revision or run the figure synchronization script automatically.

The earlier task to restore the bibliography is already addressed in the current source: `Reference.bib` contains 21 ADS-derived AASTeX entries, and `main.tex` configures author-year citations through natbib. Preserve that workflow. This file contains LaTeX `bibitem` entries read with `input`, not a conventional BibTeX database. Renaming headings does not require re-exporting the references.

When revising a scientific sentence, check that its existing citation supports the new wording. A stronger-sounding verb can change the claim: “supports,” “constrains,” and “demonstrates” are not interchangeable. If a genuinely new source is needed, follow the established ADS-to-AASTeX procedure rather than inserting an unverified reference.

## 9. Relationship to the earlier scientific review

This plan supersedes the earlier advice on the title, heading register, conversational opening, and background length. It also updates the work list to acknowledge the six-subsection background, citations, and student figure briefs already present. The earlier proposal to merge short background topics is no longer appropriate for the agreed six-subsection structure.

The [10 September assessment](/Users/s2227120/IAFlowCloud/Report/Review_and_Editing_Plan_2026-09-10.md) remains the record of the broader source and artifact audit. Its numerical tables, run coverage, and compilation observations refer to that dated inspection. They have not been refreshed by this editorial pass and should not be presented as new validation.

The following later-section issues remain outside the next title/Sections 1-2 pass:

| Outstanding issue | Later action |
|---|---|
| Historical pilot versus newer experiment artifacts | Select a coherent reported study and reconcile its architecture, checkpoints, results, and figures together. |
| Blanket claim that the test split was never read | Reconcile the wording with the historical PCA diagnostics documented in the earlier review. |
| Interpretation of reconstruction success | Distinguish average accuracy, difficult cases, and any untested implications for cosmological inference. |
| Abstract and conclusion | Revise after the reported experiment is settled; use one coherent quantitative result and its scope limit. |
| Appendices and acknowledgements | Resolve provenance and contribution details before treating them as final; defer source-file renaming. |

If the later sections are subsequently revised, suitable section titles are **Model formulation and dataset construction**, **Autoencoder architecture and training**, **Results and discussion**, and **Conclusions and future work**. Possible subsection refinements include **Two-dimensional latent representation**, **Motivation for one-dimensional convolutions**, and **Reconstruction-error diagnostics**. These are recorded recommendations, not changes to Sections 3 onward.

The title proposed here can accommodate a later accuracy-versus-dimension study without another conceptual rebranding. It should not be used as evidence that such a study has already been incorporated into the present report.

## 10. Implementation sequence and acceptance criteria

### Pass 1: title and headings

Apply the recommended main title, choose one subtitle, and update the Section 2 headings. Keep Section 1 as Introduction. Preserve labels and source filenames. Inspect both title appearances and the table of contents for balanced line breaks and readable hierarchy.

**Completion condition:** a reader scanning the title and contents can identify the astrophysical problem, the reconstruction investigation, and the scientific progression of the background.

### Pass 2: introduction and transitions

Revise the opening, IA transition, compression question, contribution paragraph, and result preview. Use the currently reported pilot as the scope constraint. Follow the physical motivation through to the mathematical representation problem without reproducing the abstract's numerical sequence.

**Completion condition:** the introduction explains why compress a family already defined by 13 parameters, identifies the comparison being made, and states the significance and limit of its reported result.

### Pass 3: background prose and terminology

Make targeted edits within the six existing subsections. Define terms before relying on them, replace vague conversational connectors with explicit scientific relationships, and keep the physical explanation before each equation. Preserve numerical values and cited data/model distinctions during the tone pass.

**Completion condition:** each subsection has a clear scientific purpose, at least one planned figure, and a transition that prepares the reader for the next topic. The reader can explain k, redshift, shear, intrinsic alignment, and the reconstruction target before reaching Section 3.

### Pass 4: captions and guide consistency

Synchronize the figure-guide headings. Check captions against the student's actual artwork as figures become available. Preserve source provenance and the distinction between observations, illustrations, and prospects. Do not treat a drafting box as a completed scientific figure.

**Completion condition:** figure references, filenames, and subsection numbers agree; caption language describes the delivered figure and the guide contains the remaining construction instructions.

### Pass 5: verification and student review

Build the revised manuscript from a complete source snapshot containing `main.tex`, `Reference.bib`, all section files, and the selected figures. Use `latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error`, check the converged log, and visually inspect every page. Confirm that the abstract, Sections 3-8, and protected figure assets remain unchanged unless separately revised.

The student should explain the title and subtitle in her own words, justify the model and comparison, and identify what the evidence does and does not establish. Resolve any wording that she cannot explain through discussion and revision. Scientific terminology should follow understanding.

**Completion condition:** no unresolved citations, broken references, clipped titles, or unreadable captions; no accidental change in experiment scope; and a coherent explanation that the student can defend. Manuscript compilation after these edits is a future verification step, not a result of producing this plan.

## 11. File map and delivery status

| File | Role in the next manuscript pass |
|---|---|
| `Report/main.tex` | Title/subtitle only; preserve abstract and reference setup unless a specific technical fix is needed. |
| `Report/sections/Section1.tex` | Motivation, research question, contribution, and result preview. |
| `Report/sections/Section2.tex` | One section heading, six subsection headings, prose, and six captions. |
| `Report/Section2_Figure_Guide.md` | Synchronize subsection titles and maintain detailed student instructions. |
| `Report/Reference.bib` | Preserve current ADS entries; edit only if a claim requires a verified new or corrected reference. |
| `Report/sections/Section3.tex` through `Section8.tex` | Protected during the next title/Sections 1-2 pass. |

This delivery consists of an editable Markdown plan and a matching PDF reading copy. The original 10 September audit and all manuscript sources remain intact. The new title/subtitle and sentence examples are recommendations for the next revision, not edits already applied to the report.
