# Frozen sensitivity run configurations

Run `H1`, `H2`, `H3`, and `H4` with `scripts/run_or_resume.py`. Then run
`scripts/select_sensitivity.py`; it reports whether `F1`/`F2` are triggered and
which single `_norm` configuration is required. Do not run every normalization
variant.

All files inherit the primary reproduction config, but the loader expands and
hashes the complete merged mapping. Each result directory stores that expanded
snapshot. The selector accepts only completed runs whose snapshot hash matches
the corresponding frozen configuration.

`configs/locked/` is intentionally absent until validation-only selection is
complete. Generate it with `scripts/select_sensitivity.py --write-lock`, review
and commit it, and only then start the three final seeds.
