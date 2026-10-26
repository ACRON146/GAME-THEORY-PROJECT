# Game-Theoretic GAN Privacy

A reproducible semester-project implementation for studying privacy-utility strategy selection in a tabular GAN. It trains an ordinary GAN and score-gap-regularized variants, evaluates utility and membership inference separately, then solves a finite-grid Stackelberg game for the privacy strength.

## Project layout

```text
configs/default.yaml       Experiment, split, game and attack settings
data/README.md             Dataset source and preprocessing notes
src/data.py                UCI Adult download, encoding and disjoint splits
src/gan.py                 Tabular GAN and differentiable regularizer
src/attacks.py             Distance and shadow-model membership attacks
src/evaluation.py          Held-out utility and feature-distribution metrics
src/game.py                Validated finite-grid Stackelberg solver
src/experiments.py         Seeded experiment runner, summaries and report
tests/                     Mathematical and pipeline tests
results/raw/               Per-run metrics and failure records
results/tables/            Aggregates, strategies and sensitivity results
results/figures/           Generated analysis plots
reports/research_report.md Report regenerated from saved outputs
```

## Environment and data

Python 3.10–3.13 is supported. From this directory:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m src.cli prepare-data
```

The preparation command downloads the UCI Adult benchmark files into `data/raw/` and writes the selected, cleaned columns under `data/processed/`. See [data/README.md](data/README.md) for attribution, selected columns, and preprocessing. Encoders are fit on GAN member rows only; unknown reference categories are safely ignored by the one-hot encoder.

## Mechanism and threat model

The generator and discriminator are compact MLPs. The discriminator objective is

```text
L_D = BCE(real, 1) + BCE(fake, 0) + lambda * (mean(D(member)) - mean(D(reference)))^2
```

The score-gap term remains connected to discriminator parameters and its adversarial and privacy components are logged separately. `lambda=0` recovers the ordinary GAN objective. The reference records are used by the training objective, so they are separate from the untouched membership-attack non-members. The split also reserves an independent held-out test set for utility measurements.

The baseline black-box attacker scores a candidate by distance to the nearest released row, using categorical mismatch and scaled numeric distance. The shadow-model attacker fits a logistic classifier on release-distance summaries from separate shadow models. Its classifier is trained using shadow membership labels; target evaluation labels are used only to report metrics. Both attacks are run for each configured privacy strength and release budget. Results include ROC-AUC, attack advantage (`max(TPR - FPR)` over score thresholds), candidate-resampling confidence intervals, and random baselines.

The regularizer is an empirical training heuristic. It is not differential privacy and offers no formal privacy guarantee. A low AUC for these attacks does not establish that all membership information has been removed.

## Game model

The owner (leader) chooses the privacy weight `lambda`. After seeing it, the attacker (follower) chooses `b`, the number of independent synthetic release batches. The project uses

```text
U_A(lambda, b) = P(lambda, b) - c_A * b
U_O(lambda, b) = Q(lambda) - alpha * P(lambda, b) - C_O(lambda)
C_O(lambda) = c_O * lambda / (1 + lambda)
```

`P` is the strongest measured attack advantage among the configured attacks. `Q` is the F1 score of a downstream classifier trained on synthetic data and evaluated on held-out real data; the real-trained F1 and the synthetic-trained F1 are both reported. `c_A` and `c_O` are configured cost scales. The attacker maximizes its payoff for each candidate `lambda`; the owner then maximizes payoff against that best response. Ties go to lower `b`, then lower `lambda`. The reported result is optimal only over the tested finite grids, not a continuous global equilibrium.

The comparison table includes: (A) ordinary GAN (`lambda=0`), (B) configured fixed regularization, (C) game-selected regularization, and (D) the `lambda` selected by mean utility without considering attacker payoff. Each is evaluated at the attacker's measured best-response budget.

## Run the workflow

```bash
# Fast pipeline check; results are explicitly marked as smoke evidence
python -m src.cli smoke-test

# Full configured study (3 seeds, 6 lambdas, 4 release budgets, 2 shadow models)
python -m src.cli run-experiments --mode full
python -m src.cli analyze

# Optional individual steps
python -m src.cli baseline --mode full
python -m src.cli mia-check
python -m src.cli solve-game
python -m pytest -q
```

All experiment settings live in `configs/default.yaml`. Change the seed list, lambda grid, budgets, epochs, shadow count, split fractions, payoff values, or output locations there. Full runs save the exact config snapshot to `results/run_config.json`. Each seed uses the same split protocol across all lambdas; split fingerprints make runs traceable. Failed configurations are recorded in `results/raw/failures.csv`, and successful measurements are saved in `results/raw/experiments.csv`.

## Outputs

- `results/raw/experiments.csv`: per-seed, per-lambda, per-budget utility, attack, loss and runtime metrics, including per-epoch target and shadow loss histories.
- `results/raw/failures.csv`: failed runs with error type and message.
- `results/tables/summary.csv`: means, standard deviations and run counts.
- `results/tables/game_solution.csv`: attacker best response and owner payoff for every lambda and alpha.
- `results/tables/strategy_comparison.csv`: the four strategy conditions.
- `results/tables/sensitivity.csv`: attacker-cost and lambda-grid sensitivity.
- `results/figures/`: utility, attack, runtime, privacy-utility, game response and condition plots.
- `reports/research_report.md`: regenerated report based on the saved experiment rows.
- `results/processed/synthetic_preview.csv`: a small synthetic data preview.

Run `analyze` after the full experiments to regenerate the tables, game solution, figures and report. The solver and report only use saved successful rows; no unrun full result is implied. The `smoke-test` command is for pipeline validation and must not be cited as confirmatory research evidence.

## Limitations

The score-gap regularizer may fail to improve privacy or may reduce data utility. The shadow attack uses a simple distance-feature classifier and is not exhaustive. The project uses one benchmark dataset and a compact GAN; conclusions may not generalize to other data or model families. Training time increases with the number of seeds, lambdas, epochs and shadow models. No DP baseline is included because a correct privacy accountant and compatible tabular training implementation would be a separate research contribution.

## References

- Xu et al. (2019), “Modeling Tabular Data using Conditional GAN,” NeurIPS. <https://papers.nips.cc/paper/8953-modeling-tabular-data-using-conditional-gan>
- Shokri et al. (2017), “Membership Inference Attacks against Machine Learning Models,” IEEE S&P. <https://doi.org/10.1109/SP.2017.41>
- Hyeong et al. (2022), “An Empirical Study on the Membership Inference Attack against Tabular Data Synthesis Models.” <https://arxiv.org/abs/2208.08114>
- UCI Adult dataset. <https://archive.ics.uci.edu/dataset/2/adult>
