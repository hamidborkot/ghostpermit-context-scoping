# GhostPermit unified manuscript: assembly note

## One paper, one source

Use `GhostPermit_JointUse_CEE.tex` as the Overleaf main document. It contains the abstract, highlights, keywords, all eight sections, eight tables, both algorithms, all appendices, and declaration headings. Do not paste its sections alongside older versions; replace the old main document. No separate numbers file is loaded.

The scientific center is the joint-use authorization specification. Historical labels provide the empirical characterization; PARS, follow-ups, classifier observations and memory repetition support that story rather than becoming competing contributions.

## Table formatting

All tables use CAS `table`/`table*`, `width=...`, and `tabular*{\tblwidth}`, following the supplied template. Column specifications use standard `p{...}`, `l`, `c`, and `r` with `array` loaded. There are no `tabularx` environments, custom X/Y definitions or standalone column-specification text. The main cell matrix is full-width with stacked source headings. Summary and follow-up tables are single-column; the PARS table is full-width for readable ranges.

The order is automatic: 1 related work; 2 notation; 3 configurations; 4 headline labels; 5 full matrix; 6 conditional recovery; 7 PARS; 8 appendix variants. The recovery/PARS table numbers follow their order of discussion. All references use labels rather than typed numbers.

## Existing Overleaf assets

Keep your CAS bundle and verified `cas-refs.bib`. The source embeds two statistical BibTeX entries in a `filecontents*` block and generates `ghostpermit-jointuse-statistics.bib` when compiled. Their keys are prefixed `gp` to avoid collision with existing entries. Wilson's metadata and McCullagh–Nelder's book metadata were checked against source listings during this response; this does not make the approximate scaled-deviance test an exact test.

The source uses numeric natbib citations and `cas-model2-names`, as in the sample. All manuscript citations are `\cite{...}`.

Your artwork is sought in `figures/` (PDF or PNG) or `figs/` (PDF):
- fig1_concept
- fig2_combination_profile
- fig3_pars
- fig4_propagation

Figure 1 must distinguish the proposed authorization path from the historical observation path. Figure 4 must show instruction reinsertion. These are existing-figure editing instructions, not new generated images. If a file is absent, a labelled placeholder is rendered so the source still assembles; replace it with the actual artwork before submission.

## Checks already performed

The accompanying JSON records structural checks. All internal labels are defined once; braces and environments are balanced. There are eight sections, eight tables, two algorithms and four figure environments. Every cell of the main count matrix was checked against the consolidated ledger. Protocol A totals 545/1,435 and all seven total 636/2,035. The reanalysis script was executed successfully.

The empirical script is offline arithmetic/model reanalysis, not the original generation or judge implementation. It reads the supplied-count reconstruction and cannot recover lost response texts. The paper distinguishes that provenance explicitly.

No TeX engine or CAS class bundle is installed in the execution environment, so the manuscript has not been compiled or visually proofed here. Exact pages and overfull-box warnings must be checked in Overleaf after your artwork and bibliography are present.

## Final author-controlled entries

Author metadata were deferred at your request. Fill author/affiliation/CRediT, the truthful competing-interest and funding statements, repository identifier and actual AI-use disclosure. These are commented fields, not invented claims. The data statement identifies the records actually retained rather than promising a response-level dataset.

## Content decisions kept fixed

- Joint-use authorization is conditional on sound provenance, authenticated context, correct predicate evaluation and complete mediation.
- The proposed release algorithm is a specification; historical model runs did not execute it.
- Positive labels are original automated outcomes, not verified retrieval, policy violations or validated safety decisions.
- Protocol A and B differ in system prompt and judge and are not used for an underlying-model security ranking.
- Main model: exploratory, grouped counts, exact dimensions/reference df documented.
- Gate result: F=2.31, p=0.14 under the stated approximation.
- Prompt Guard 2: 400 scores, 200 per size; no detector bypass or false-positive claim.
- Memory: 85/196 analyzed calls; no autonomous-replication or unverified trend claim.
- PARS: author-assigned illustrative composition assumptions, exact threshold rule, numerical sensitivity only.
- Source counts and historical descriptions derive from the supplied documents and author confirmations; arithmetic checks are not independent validation of original label accuracy.
