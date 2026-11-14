# National Bridge Inventory state-border discontinuities

Reproducible Python analysis accompanying *Testing cross-state comparability of National Bridge Inventory condition ratings using geographic discontinuities* (Pawan Bhattarai).

## Study overview

The analysis examines reported deck, superstructure, and substructure condition ratings in the 2025 legacy National Bridge Inventory (NBI). It screens 108 shared U.S. state borders without reference to the ratings and estimates geographic discontinuities for 13 eligible borders. Kansas–Missouri is examined in detail using inspection-recency controls, spatial and sample sensitivity analyses, PRISM climate normals, and fixed-specification comparisons using the 2023 and 2024 NBI files. A discontinuity is **not** evidence that an inspection agency is biased.

## What is included

- `scripts/` and `run_all.py`: data preparation, screening, estimation, statistical inference, validation, and figure production.
- `results/frozen/`: original archived numerical results and `SHA256SUMS.txt` for integrity verification.
- `outputs/figures/`: archived research figures (the manuscript uses a selected subset with updated presentation).
- `data/external/`: derived PRISM climate and metropolitan-boundary inputs with provenance.
- `tests/`: synthetic unit tests.

The large raw FHWA and Census datasets are **not redistributed**. Their original source URLs, sizes, and SHA-256 hashes are recorded in `results/frozen/input_manifest.json`.

## Reproduce

Python 3.11 or newer is recommended. From the repository directory:

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python scripts/download_public_inputs.py
python run_all.py
python scripts/12_make_manuscript_figures.py
python scripts/11_freeze_results.py
python scripts/13_check_manuscript_values.py
```

`run_all.py` uses the archived climate and metropolitan data already supplied. To regenerate those external datasets from PRISM and Census sources instead, run `python run_all.py --prepare-external` (requires an internet connection). A full reanalysis requires the original FHWA and TIGER inputs; passing the synthetic tests or checking stored hashes alone does not reproduce the regressions from raw data. Regenerating/finalizing will replace files in `results/generated/` and may update `results/frozen/`; preserve the distributed snapshot before doing so.

## Main archived results

The 2025 analysis retained 464,584 conventional bridges and identified 13 eligible borders. Holm-adjusted conventional inference identified five statistically significant borders; the additional few-cluster bootstrap assessment retained four. For Kansas–Missouri, the primary 16.1 km inspection-recency-adjusted estimate was **+0.401 rating points** (95% CI 0.311–0.491). Fixed-specification estimates were +0.270 (2023), +0.281 (2024), and +0.401 (2025); the pooled 2023–2025 estimate was +0.307. The pooled 8,639 records include repeated bridges and are not independent samples.

The archived numerical outputs and input-manifest hashes were checked for integrity in this release; the raw-data pipeline has not been rerun as part of the repository cleanup.

## Manuscript figures

The manuscript figures correspond in subject to `Fig1_All_Border_Forest`, `Fig4_KSMO_RD`, `Fig5_KSMO_Robustness`, and `Fig7_Temporal_Replication`. The archived figure generator may use a simpler plotting layout than the final manuscript artwork; the numerical estimates and confidence intervals are the comparison standard. Additional figures are supporting analyses.

## License

MIT (see `LICENSE`). Cite the accompanying manuscript when discussing study findings.
