from __future__ import annotations

import hashlib
from pathlib import Path
import requests
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

ADULT_COLUMNS = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"]
URLS = {
    "adult.data": "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
    "adult.test": "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test",
}

def prepare_data(raw_dir: str | Path, out_path: str | Path, features: list[str], target: str = "income") -> Path:
    raw = Path(raw_dir); raw.mkdir(parents=True, exist_ok=True)
    for name, url in URLS.items():
        dest = raw / name
        if not dest.exists():
            response = requests.get(url, timeout=60); response.raise_for_status()
            dest.write_bytes(response.content)
    frames = []
    for name in URLS:
        df = pd.read_csv(raw / name, header=None, names=ADULT_COLUMNS, skipinitialspace=True, comment="|", na_values="?")
        df["income"] = df["income"].astype(str).str.strip().str.rstrip(".")
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)[features + [target]].dropna().reset_index(drop=True)
    for column in data.select_dtypes(include="object").columns: data[column] = data[column].str.strip()
    out = Path(out_path); out.parent.mkdir(parents=True, exist_ok=True); data.to_csv(out, index=False)
    return out

def load_dataset(path: str | Path, sample_rows: int = 0, seed: int = 11) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Processed dataset not found at {path}. Run `python -m src.cli prepare-data` first.")
    df = pd.read_csv(path)
    if sample_rows and len(df) > sample_rows: df = df.sample(sample_rows, random_state=seed).reset_index(drop=True)
    return df

class TabularEncoder:
    """Train-fitted one-hot/min-max transform with a deterministic inverse."""
    def __init__(self, categorical: list[str], numeric: list[str]):
        self.categorical, self.numeric = categorical, numeric
        self.ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.scaler = MinMaxScaler()
    def fit(self, frame: pd.DataFrame):
        self.ohe.fit(frame[self.categorical].astype(str))
        self.scaler.fit(frame[self.numeric].astype(float))
        self.cat_sizes = [len(c) for c in self.ohe.categories_]
        self.cat_names = list(self.ohe.get_feature_names_out(self.categorical))
        self.columns = self.cat_names + self.numeric
        return self
    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        cats = self.ohe.transform(frame[self.categorical].astype(str))
        nums = self.scaler.transform(frame[self.numeric].astype(float))
        return np.concatenate([cats, nums], axis=1).astype(np.float32)
    def inverse(self, matrix: np.ndarray) -> pd.DataFrame:
        k = sum(self.cat_sizes); cat = self.ohe.inverse_transform(matrix[:, :k])
        num = self.scaler.inverse_transform(np.clip(matrix[:, k:], 0, 1))
        out = pd.DataFrame(cat, columns=self.categorical)
        for j, name in enumerate(self.numeric): out[name] = num[:, j]
        return out[self.categorical + self.numeric]

def split_data(frame: pd.DataFrame, seed: int, member_fraction=.6, reference_fraction=.1,
               nonmember_fraction=.15, test_fraction=None):
    """Create disjoint member, regularizer-reference, attack non-member and test sets.

    Reference records are explicitly separate from attack non-members because
    the privacy regularizer reads reference records during training.
    """
    fractions = [float(member_fraction), float(reference_fraction), float(nonmember_fraction)]
    if test_fraction is None:
        test_fraction = 1.0 - sum(fractions)
    fractions.append(float(test_fraction))
    if any(not np.isfinite(value) or value <= 0 for value in fractions) or not np.isclose(sum(fractions), 1.0, atol=1e-8):
        raise ValueError("Member, reference, nonmember and test fractions must be positive and sum to 1.")
    if len(frame) < 20 or "income" not in frame.columns:
        raise ValueError("A sufficiently large frame with an income column is required for stratified splits.")

    member, remaining = train_test_split(
        frame, train_size=fractions[0], random_state=int(seed), stratify=frame["income"])
    reference_fraction_of_rest = fractions[1] / (1.0 - fractions[0])
    reference, remaining = train_test_split(
        remaining, train_size=reference_fraction_of_rest, random_state=int(seed) + 1,
        stratify=remaining["income"])
    nonmember_fraction_of_rest = fractions[2] / (fractions[2] + fractions[3])
    nonmember, test = train_test_split(
        remaining, train_size=nonmember_fraction_of_rest, random_state=int(seed) + 2,
        stratify=remaining["income"])
    split_material = pd.util.hash_pandas_object(frame, index=True).values.tobytes()
    fp = hashlib.sha256(split_material + (str(seed) + str(tuple(fractions))).encode()).hexdigest()[:12]
    clean = lambda part: part.reset_index(drop=True)
    return clean(member), clean(reference), clean(nonmember), clean(test), fp
