# Figure sources and credits

The supplied PDFs in this directory are the report's build inputs. This file
retains their credits and provenance. Unused source assets and historical
drafts have been moved out of the report to
`Document/Report_Development_Archive/2026-09-14/Unused_Figure_Sources/`
in the full project. The manuscript requires no plotting scripts, numerical
result folders or archived source assets.

## Original notebook exports

These six report PDFs are byte-for-byte copies of existing exports in the
full project's `Figure/NLA/` directory. No plot was redrawn for this update.

| Report PDF | Existing export, relative to `Figure/NLA/` |
|---|---|
| `ae_validation_diagnostics.pdf` | `AE/AE_Selected_Validation_Diagnostics.pdf` |
| `ae_worst_reconstructions.pdf` | `AE/AE_Selected_Worst_Reconstructions.pdf` |
| `ae_latent_geometry.pdf` | `AE/AE_Selected_Latent_Geometry.pdf` |
| `ae_latent_parameter_correlations.pdf` | `AE/AE_Selected_Latent_Parameter_Correlations.pdf` |
| `pca_explained_variance.pdf` | `PCA/Explained_Variance.pdf` |
| `pca_reconstruction_error_by_rank.pdf` | `PCA/Reconstruction_Error_by_Rank.pdf` |

The four AE exports were saved by `Notebooks/NLA/AE.ipynb` on 2 September
2026. They describe `Runs/NLA/AE/Conv1D/Depth03/Latent06/20260823-103003`,
checkpoint epoch 561, on all 10,000 validation surfaces. That checkpoint
has 1.25% mean relative error, 25.57% maximum error, 3.21% surface-RMS
95th percentile and 17.17% surface-maximum 99th percentile. It is a different
fit from the September run used in the main comparison. The manuscript
identifies that distinction explicitly. The PCA spectrum is the original
rank-30 training-basis export from `Notebooks/NLA/PCA.ipynb`. The PCA rank-error
plot evaluates ranks 1--30 on a fixed subset of 256 validation surfaces;
its maximum relative errors are fractions, not percentages. Both PCA files
were renamed to match the manuscript references without changing their contents.

## Other numerical figures

The main comparison and tail-comparison PDFs retain the preceding report
plots from saved validation artifacts, with the three-layer Conv1D series
`20260909-003908` at every width. Its six-coordinate checkpoint is epoch
538 and has 1.21% mean relative error and 26.36% maximum error. The two
comparison plots cover 30 neural configurations and matched PCA ranks
2, 4, 6, 8 and 10, using the same 10,000 validation surfaces.

The complete-validation PCA scores discussed in the text come from stored
evaluations at ranks 2, 4, 6, 8, 10 and 25--30. They are distinct from the
256-surface notebook diagnostic now shown in the report. The two comparison
PDFs above are unchanged.

Section 3 plots originate from the project's analytic-model and sampling
workflows. Earlier report edits reordered the luminosity-factor panels and
corrected labels; their numerical curves were preserved. The explanatory
architecture diagrams depict the reported three-layer configurations with
exact printed dimensions and schematic geometric proportions.

## Published background panels

The original asset filenames below refer to the development archive above.

- **Planck CMB:** `planck_cmb_2018.jpg`, from
  [ESA, Planck's view of the cosmic microwave background](https://www.esa.int/ESA_Multimedia/Images/2018/07/Planck_s_view_of_the_cosmic_microwave_background).
  Credit: ESA/Planck Collaboration. ESA permits educational and informational
  reuse of publicly released images with credit.
- **DESI DR2:** `desi_dr2_figure8_left.svg` and `desi_dr2_figure11.svg`, from
  [DESI DR2 Results II, arXiv:2503.14738v3](https://arxiv.org/abs/2503.14738v3),
  Figure 8 left and Figure 11. Attribution: DESI Collaboration. CC BY 4.0.
- **Stage III comparison:** `kids_legacy_figure14_omega_m_s8.svg`, from
  [Wright et al., KiDS-Legacy, arXiv:2503.19441v2](https://arxiv.org/abs/2503.19441v2),
  the Omega_m--S_8 panel of Figure 14. Attribution: Wright et al. (2025).
  CC BY 4.0. This historical panel does not include subsequent HSC updates,
  DES Y6 results or Stage IV forecasts.

The archived `before_section2_finalisation/` inputs record the earlier
published-panel assembly and label corrections. The archived HSC comparison
under `schematics/` is a retired draft and is not included in the report.
Full credits also appear in the manuscript captions.

## Assistance and development history

Incoming explanatory schematics credited GPT-5.6 Sol assistance. Subsequent
Codex assistance covered schematic and architecture drawing, plotting-code
and label changes, the preceding report-generated result plots,
scientific interpretation, captions, language, provenance checks and PDF
verification. The four AE diagnostic PDFs were copied unchanged in this
update; their original notebook provenance is recorded above. Copying them
does not establish independent student authorship or completed verification.

Earlier sources, retired plot details and assistance records are preserved
in `Document/Report_Development_Archive/2026-09-13/` and `2026-09-14/` in the
full project. The immediate incoming report, source README, copy hashes and
verified diagnostic metadata are retained in
`2026-09-14/Original_Notebook_PDFs/`. The student and supervisor must supply
their actual contributions and complete assistance history before submission.
