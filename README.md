# Game-Theoretic GAN Privacy
### A Game-Theoretic Approach to Privacy–Utility Trade-offs in Tabular Generative Adversarial Networks

## 1. Project Overview

This project investigates the trade-off between **data privacy and data utility** in Tabular Generative Adversarial Networks (GANs) using Game Theory.

Generative Adversarial Networks can generate synthetic datasets that resemble real-world data. However, they may unintentionally expose information about records used during training, creating privacy risks such as Membership Inference Attacks (MIA).

To address this problem, this project introduces a game-theoretic framework in which a data owner and a potential attacker interact strategically.

The data owner selects a privacy regularization strength (`lambda`) to reduce membership leakage while preserving the usefulness of generated synthetic data. The attacker observes this choice and selects the number of synthetic data release batches to maximize its attack payoff.

A finite-grid **Stackelberg Game** is used to identify the optimal privacy strategy by considering both the owner's utility and the attacker's best response.

The project implements and evaluates:
- A standard Tabular GAN baseline.
- Score-gap-regularized GAN models.
- Black-box membership inference attacks.
- Shadow-model-based membership inference attacks.
- Privacy and utility evaluation.
- Stackelberg game-based strategy optimization.
- Sensitivity analysis across different game parameters.

The implementation uses the UCI Adult Income Dataset as its benchmark dataset.

---

## 2. Project Information

| Detail | Information |
|---|---|
| Project Title | Game-Theoretic GAN Privacy |
| Project Domain | Artificial Intelligence, Generative AI, Game Theory, Data Privacy |
| Model | Tabular Generative Adversarial Network |
| Game Theory Approach | Stackelberg Game |
| Dataset | UCI Adult Income Dataset |
| Programming Language | Python |
| Framework | PyTorch |
| Evaluation | Utility Metrics and Membership Inference Attacks |

### Group Members

| Student Name | Roll Number |
|---|---|
| ARUNJAY SACHAN | I24AI024 |
| PURVA SOLANKI | I24AI013 |

---

## 3. GitHub Repository

**GitHub Repository Link:**

[PASTE YOUR GITHUB REPOSITORY LINK HERE]

The complete source code, configuration files, experiment results, documentation, and testing files will be maintained in this repository.

---

## 4. Dataset Resources

### UCI Adult Income Dataset

This project uses the UCI Adult Income Dataset, which contains demographic and employment-related information used to predict whether an individual's annual income exceeds $50,000.

**Official Dataset Resource:**

https://archive.ics.uci.edu/dataset/2/adult

**Dataset Information:**

- Dataset Name: Adult (Census Income)
- Source: UCI Machine Learning Repository
- Data Type: Tabular
- Features: Numerical and categorical attributes
- Target Variable: Income
- Primary Application: Income classification and synthetic tabular data generation.

### Dataset Preprocessing

The project performs the following preprocessing operations:

1. Downloads the dataset from the UCI repository.
2. Selects and cleans the required dataset columns.
3. Handles missing and categorical values.
4. Encodes categorical features using One-Hot Encoding.
5. Processes numerical features.
6. Creates separate and disjoint data splits for GAN training, reference data, membership attack evaluation, and held-out utility evaluation.

The encoders are fitted using GAN member training records only. Unknown categories in reference records are handled safely.

The dataset is downloaded automatically during the data preparation stage, so manual downloading is not required.

---

## 5. Implementation Environment

The project is implemented using Python and PyTorch.

### Software Requirements

| Component | Requirement |
|---|---|
| Programming Language | Python 3.10–3.13 |
| Deep Learning Framework | PyTorch |
| Numerical Computing | NumPy |
| Data Processing | Pandas, Scikit-learn |
| Visualization | Matplotlib |
| Configuration | PyYAML |
| Testing | Pytest |
| Experiment Configuration | YAML |
| Operating System | Windows / Linux / macOS |

### Hardware Requirements

Recommended environment:

- Processor: Modern multi-core CPU
- RAM: 8 GB or higher
- Storage: Sufficient space for datasets and experiment outputs
- GPU: Optional; CUDA-compatible GPU recommended for faster training.

The project can also run on a CPU, although full experiments may take longer.

---

## 6. Environment Setup and Installation

Follow the steps below to configure and execute the project.

### Step 1: Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
```

Navigate to the project directory:

```bash
cd game-theoretic-gan-privacy
```

### Step 2: Create a Virtual Environment

```bash
python -m venv .venv
```

Activate the environment.

**Windows (PowerShell):**

```powershell
.venv\Scripts\Activate.ps1
```

**Linux / macOS:**

```bash
source .venv/bin/activate
```

### Step 3: Install Dependencies

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

Install the required libraries:

```bash
python -m pip install -r requirements.txt
```

### Step 4: Prepare the Dataset

Execute the following command:

```bash
python -m src.cli prepare-data
```

This command downloads the UCI Adult dataset, performs preprocessing, and saves the processed dataset in the appropriate directory.

### Step 5: Run a Pipeline Test

Before running the complete experiments, execute:

```bash
python -m src.cli smoke-test
```

This checks the basic functionality of the pipeline. Smoke-test results are intended for validation and should not be treated as confirmatory research results.

---

## 7. Project Methodology

The project consists of three major components.

### 7.1 Generative Adversarial Network (GAN)

The GAN consists of two neural networks:

**Generator:**
- Learns the underlying distribution of the training data.
- Generates synthetic tabular records.
- Attempts to produce records that resemble real data.

**Discriminator:**
- Distinguishes between real and generated records.
- Learns to identify differences between real and synthetic data.

The discriminator objective includes an additional privacy regularization term.

The implemented objective is:

```text
L_D = BCE(real, 1) + BCE(fake, 0)
      + lambda * (mean(D(member)) - mean(D(reference)))^2
```

Where:

- `BCE` represents Binary Cross Entropy.
- `D(real)` represents discriminator scores.
- `member` represents records used to train the GAN.
- `reference` represents separate reference records.
- `lambda` controls the strength of privacy regularization.

When `lambda = 0`, the privacy regularization term becomes zero, recovering the ordinary GAN discriminator objective.

### 7.2 Membership Inference Attacks

The project evaluates privacy leakage using two membership inference attacks.

**A. Distance-Based Black-Box Attack**

- Measures the distance between candidate records and released synthetic records.
- Uses categorical mismatch and scaled numerical distances.
- Uses nearest-neighbor distance to determine membership-related scores.

**B. Shadow-Model-Based Attack**

- Trains separate shadow models to simulate membership behavior.
- Extracts release-distance features.
- Trains a logistic regression classifier using shadow membership labels.
- Evaluates the attack against the target model without using target evaluation labels for training.

Both attacks are evaluated using:

- ROC-AUC
- Attack Advantage
- Candidate-resampling Confidence Intervals
- Random Baseline Comparisons

### 7.3 Stackelberg Game Theory Model

The project implements a leader-follower Stackelberg Game.

**Leader: Data Owner**

The data owner selects the privacy regularization parameter:

`lambda`

**Follower: Attacker**

After observing the owner's decision, the attacker selects:

`b = Number of Independent Synthetic Release Batches`

The payoff functions are:

**Attacker's Utility:**

```text
U_A(lambda, b) = P(lambda, b) - c_A * b
```

**Owner's Utility:**

```text
U_O(lambda, b) = Q(lambda) - alpha * P(lambda, b) - C_O(lambda)
```

**Owner's Regularization Cost:**

```text
C_O(lambda) = c_O * lambda / (1 + lambda)
```

Where:

- `P(lambda, b)` = Strongest measured membership attack advantage.
- `Q(lambda)` = F1 score of a downstream classifier trained on synthetic data and evaluated on held-out real data.
- `c_A` = Attacker's release cost.
- `c_O` = Owner's privacy regularization cost scale.
- `alpha` = Privacy-risk penalty assigned by the owner.
- `lambda` = Privacy regularization strength.
- `b` = Number of synthetic release batches.

The attacker first maximizes its payoff for each possible privacy strategy. The owner then selects the privacy strategy that maximizes its payoff against the attacker's best response.

The solver evaluates a finite set of candidate strategies and returns the optimal strategy within the configured search grids.

---

## 8. Experiment Execution

All experiment parameters are configurable through:

```text
configs/default.yaml
```

This includes:
- Random seeds.
- Privacy regularization values.
- Synthetic release budgets.
- GAN training epochs.
- Shadow-model configurations.
- Data split fractions.
- Game-theoretic payoff parameters.
- Output directories.

### Run the Complete Experiment

Execute:

```bash
python -m src.cli run-experiments --mode full
```

After the experiments finish, generate the analysis:

```bash
python -m src.cli analyze
```

### Additional Commands

Run the baseline:

```bash
python -m src.cli baseline --mode full
```

Run membership inference checks:

```bash
python -m src.cli mia-check
```

Solve the Stackelberg game:

```bash
python -m src.cli solve-game
```

Run unit and pipeline tests:

```bash
python -m pytest -q
```

The configured full experiment uses three random seeds, six privacy regularization values, four release budgets, and two shadow models.

---

## 9. Project Directory Structure

```text
game-theoretic-gan-privacy/
│
├── configs/
│   └── default.yaml
│
├── data/
│   ├── README.md
│   ├── raw/
│   └── processed/
│
├── src/
│   ├── data.py
│   ├── gan.py
│   ├── attacks.py
│   ├── evaluation.py
│   ├── game.py
│   ├── experiments.py
│   └── cli.py
│
├── tests/
│
├── results/
│   ├── raw/
│   ├── tables/
│   ├── figures/
│   └── processed/
│
├── reports/
│   └── research_report.md
│
├── requirements.txt
└── README.md
```

### Description of Important Modules

| File | Functionality |
|---|---|
| `data.py` | Dataset downloading, preprocessing, encoding, and splitting |
| `gan.py` | GAN architecture and privacy regularization |
| `attacks.py` | Membership inference attack implementations |
| `evaluation.py` | Utility and feature-distribution evaluation |
| `game.py` | Stackelberg game solver and payoff calculations |
| `experiments.py` | Experiment execution, result aggregation, and reporting |
| `cli.py` | Command-line interface |
| `default.yaml` | Experiment and game configuration |

---

## 10. Experimental Outputs

The project generates several outputs to evaluate the effectiveness of the proposed approach.

| Output File | Description |
|---|---|
| `results/raw/experiments.csv` | Individual experiment measurements |
| `results/raw/failures.csv` | Failed experiment configurations |
| `results/tables/summary.csv` | Aggregated results and statistical summaries |
| `results/tables/game_solution.csv` | Attacker best responses and owner payoffs |
| `results/tables/strategy_comparison.csv` | Comparison of four strategy conditions |
| `results/tables/sensitivity.csv` | Sensitivity analysis |
| `results/figures/` | Generated experimental plots |
| `results/processed/synthetic_preview.csv` | Preview of generated synthetic data |
| `reports/research_report.md` | Automatically generated research report |

### Strategy Comparison

The project evaluates four conditions:

1. **Ordinary GAN:** No privacy regularization (`lambda = 0`).
2. **Fixed Regularization:** Uses a predefined privacy regularization strength.
3. **Game-Selected Regularization:** Uses the privacy strategy selected by the Stackelberg game.
4. **Utility-Selected Regularization:** Selects the privacy strength based on mean utility without considering the attacker's payoff.

All four conditions are evaluated using the attacker's measured best-response release budget.

---

## 11. Research Limitations

The project has the following limitations:

- The score-gap regularization mechanism is an empirical heuristic and does not provide formal differential privacy guarantees.
- A low membership inference attack score does not establish that all privacy leakage has been eliminated.
- Only one benchmark dataset is used.
- The shadow-model attack is based on a relatively simple distance-feature classifier.
- The GAN architecture is compact and may not generalize to more complex datasets.
- The Stackelberg solution is optimal only within the configured finite strategy grids.
- No formal Differential Privacy (DP) baseline is included.

These limitations are considered when interpreting the experimental results.

---

## 12. References and Resources

The following research papers and resources provide the theoretical foundation for the implementation.

**1. Conditional GAN for Tabular Data**

Xu et al. (2019).

"Modeling Tabular Data using Conditional GAN."

NeurIPS 2019.

https://papers.nips.cc/paper/8953-modeling-tabular-data-using-conditional-gan

**2. Membership Inference Attacks**

Shokri et al. (2017).

"Membership Inference Attacks against Machine Learning Models."

IEEE Symposium on Security and Privacy.

https://doi.org/10.1109/SP.2017.41

**3. Membership Inference Attacks on Tabular Data Synthesis**

Hyeong et al. (2022).

"An Empirical Study on the Membership Inference Attack against Tabular Data Synthesis Models."

https://arxiv.org/abs/2208.08114

**4. UCI Adult Income Dataset**

UCI Machine Learning Repository.

https://archive.ics.uci.edu/dataset/2/adult

**5. PyTorch Documentation**

https://pytorch.org/docs/stable/index.html

**6. Scikit-learn Documentation**

https://scikit-learn.org/stable/

---

## 13. Conclusion

This project demonstrates how Game Theory can be integrated with Generative Adversarial Networks to study privacy–utility trade-offs in synthetic tabular data generation.

By combining privacy regularization, membership inference attacks, and Stackelberg game optimization, the implementation provides an experimental framework for understanding how a data owner can select privacy strategies while accounting for an adaptive attacker.

The primary objective is to investigate whether game-theoretic strategy selection can provide a better balance between synthetic data utility and resistance to membership inference attacks.

---

**Project Contributors**

- ARUNJAY SACHAN (I24AI024)
- PURVA SOLANKI (I24AI013)

**Project Domain:** Game Theory + Generative Adversarial Networks + Data Privacy

**Repository:** [GitHub Repository Link – To Be Added]
