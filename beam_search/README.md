# Beam Search Experiments

This directory contains the vanilla beam search baseline used to compare against the GA layer search pipeline.

## Layout

- `vanilla_beam_search.py`: standalone beam search implementation aligned with the GA layer constraints (`limit`, `min_layers`, `max_layers`, `beam_width`). Includes checkpointing, evaluation accounting and stderr suppression for MMLU runs.
- `parse_beam_progress.py`: utility for parsing beam search logs (`beam_*_log_*.txt`) and summarising progress / best scores.
- `filter_log.sh`: helper to remove noisy warnings from beam search logs, generating `*_clean.txt` summaries with statistics.
- `visualizations/`: plotting utilities for experiment analysis.
  - `visualize_fair_comparison.py`: produces the fair comparison charts between beam search and GA (matching MMLU limits and evaluation budgets).
  - `visualize_ga_patterns.py`: visualises GA pattern mining output (frequency / quality trends plus detailed tables).
- `results/`: generated artefacts and reference data for the fair comparison runs.
  - `beam_fair_results_20251104_110757.json`: raw beam search results for the fair comparison run (`beam_width=3`, `limit=10`).
  - `fair_comparison_data.json`: serialised convergence histories used by the comparison plots.
  - `fair_comparison_beam_vs_ga.png`: final fair-comparison figure (beam vs GA).
  - `ga_pattern_evolution.png`, `ga_pattern_details.png`: outputs from the GA pattern visualisation.

## Usage

1. Activate the GA environment (`conda activate ga_layer_search`).
2. Launch the beam baseline via `python beam_search/vanilla_beam_search.py` (or see `Jet-Nemotron-Comparison/run_beam_fair.sh` for a tmux example).
3. Clean logs with `bash beam_search/filter_log.sh <logfile>` and monitor progress through `python beam_search/parse_beam_progress.py --log <clean_log>`.
4. Generate visualisations with the scripts in `beam_search/visualizations/`.

These assets allow direct apples-to-apples comparisons between GA and beam search under matched evaluation budgets.
