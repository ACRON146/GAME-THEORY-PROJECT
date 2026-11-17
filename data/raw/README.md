# Dataset

The project downloads the UCI Machine Learning Repository Adult dataset (UCI dataset 2), commonly distributed as `adult.data` and `adult.test`. It prepares the configured columns `age`, `education-num`, `hours-per-week`, `workclass`, `education`, `marital-status`, `occupation`, `relationship`, and `income`.

Preparation strips whitespace, maps `?` values to missing, drops rows missing a selected field, and removes the trailing period in the test-file income labels. Categorical columns are one-hot encoded and numeric columns are min-max scaled by an encoder fitted on GAN member data only. Unknown categories in other splits are ignored by the encoder.

Each seeded split contains four disjoint roles:

1. GAN members used as real samples in the adversarial objective.
2. A regularizer-reference set read by the member/reference score-gap penalty.
3. Untouched attack non-members used only for membership inference evaluation.
4. A held-out real test set used for downstream utility measurements.

The reference set is used during training and must not be described as an untouched membership-attack non-member set. The project stores the original download in `data/raw/` and the cleaned data in `data/processed/`; the generation script can recreate them.

UCI lists the dataset under CC BY 4.0. Retain attribution to Barry Becker and Ronny Kohavi and review the terms on the [UCI dataset page](https://archive.ics.uci.edu/dataset/2/adult). The benchmark is not a current census estimate and should not be treated as representative of present-day populations.
