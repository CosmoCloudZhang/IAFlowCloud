# IAFlow competition report

This directory contains a modular LaTeX report template and an evidence-aware
editing plan for the S.-T. Yau High School Science Award project.

## Local build

From the repository root:

```bash
Report/scripts/sync_figures.sh
latexmk -pdf -cd Report/main.tex
```

The generated PDF is `Report/main.pdf`. Run `latexmk -C -cd Report/main.tex` to
remove build products. The build products are ignored by Git.

Drafting notes are visible in blue. Before submission, change
`\reportdrafttrue` to `\reportdraftfalse` in `main.tex` and search for every
remaining `\placeholder{...}`.

## Recommended local and Overleaf workflow

The cleanest long-term setup is a **small report-only Git repository** mirrored
to Overleaf, while IAFlowCloud remains the code/data repository. The report
repository should contain only LaTeX, selected final figures, and lightweight
result tables. Codex can work with both local repositories, and
the explicit figure-sync script keeps every plot traceable to IAFlowCloud.

For the current integrated layout, `Report/` can remain in IAFlowCloud while the
outline is changing. Before two-way student editing begins:

1. create a report-only repository from the contents of `Report/`;
2. set `IAFLOW_PROJECT_ROOT` in the local environment if the report repository
   is no longer nested inside IAFlowCloud;
3. add the Overleaf project as a Git remote if the project owner has access to
   the premium Git integration;
4. pull before starting a local editing session, commit one logical change at a
   time, and push after the local PDF builds successfully; and
5. avoid editing the same paragraph both locally and in Overleaf at the same
   time.

A practical collaboration rhythm is:

```text
student edits in Overleaf during an agreed editing window
        ↓
local pull and conflict check
        ↓
Codex updates text, tables, or figure links against the code/results
        ↓
local LaTeX build and visual review
        ↓
commit and push to Overleaf
```

Use Git commits as review units: for example, “clarify weak-lensing background,”
“freeze Table 2 validation values,” or “replace reconstruction figure.” Never
paste a new metric into the prose without updating its source table or artifact.

Overleaf's Git bridge currently exposes one linear branch named `master`; a
different local branch can be pushed to `master`. It does not preserve every
normal Git feature, and Git pushes can displace Overleaf tracked changes or
comments. Do not mix active Track Changes/comment review with Git syncing. Use
ordinary edits in agreed windows, preserve important feedback in Git issues or
an external review log, and label an Overleaf version before major
synchronizations. See the official [Overleaf Git documentation](https://docs.overleaf.com/integrations-and-add-ons/git-integration-and-github-synchronization/git).

Overleaf features and competition policies can change. Recheck them before the
student collaboration and final submission stages.
