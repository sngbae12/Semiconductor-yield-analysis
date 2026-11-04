"""Verified SECOM data and fold-local preprocessing for the final workflow."""
from __future__ import annotations
import hashlib
import os
from pathlib import Path
import tempfile
from urllib.request import urlopen
import zipfile
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted, validate_data

SEED = 42
DATA_URL = 'https://archive.ics.uci.edu/static/public/179/secom.zip'
DATA_SHA256 = 'eea568baf3c2229096d7d294cf0b096b5502bd96d92c0b80a65b84714059be8e'
PROJECT_DIR = Path(__file__).resolve().parent


class MissingRateFilter(TransformerMixin, BaseEstimator):
    """Learn which sensors to retain using only the data passed to fit."""

    def __init__(self, max_missing_rate=0.4):
        self.max_missing_rate = max_missing_rate

    def fit(self, X, y=None):
        if not 0 <= self.max_missing_rate < 1:
            raise ValueError('max_missing_rate must be in [0, 1).')
        values = validate_data(self, X, reset=True, ensure_all_finite='allow-nan')
        self.missing_rates_ = np.isnan(values).mean(axis=0)
        self.support_ = self.missing_rates_ <= self.max_missing_rate
        if not self.support_.any():
            raise ValueError('No sensors satisfy the missing-rate limit in this training fold.')
        return self

    def transform(self, X):
        check_is_fitted(self, 'support_')
        values = validate_data(self, X, reset=False, ensure_all_finite='allow-nan')
        return values[:, self.support_]


def _verified_archive(data_dir):
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    archive = data_dir / 'secom.zip'
    if archive.exists():
        payload = archive.read_bytes()
    else:
        with urlopen(DATA_URL, timeout=60) as response:
            payload = response.read()
    if hashlib.sha256(payload).hexdigest() != DATA_SHA256:
        raise ValueError('SECOM archive SHA-256 mismatch; verify the source before replacing it.')
    if not archive.exists():
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=data_dir, delete=False) as output:
                temporary = Path(output.name)
                output.write(payload)
            temporary.replace(archive)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return archive


def load_secom(data_dir=None):
    """Download/cache the official archive, returning raw sensors, binary labels, dates."""
    if data_dir is None:
        data_dir = os.environ.get('SECOM_DATA_DIR', PROJECT_DIR / 'data')
    with zipfile.ZipFile(_verified_archive(data_dir)) as archive:
        if archive.testzip() is not None:
            raise ValueError('SECOM archive CRC verification failed.')
        X = pd.read_csv(archive.open('secom.data'), sep=r'\s+', header=None)
        labels = pd.read_csv(
            archive.open('secom_labels.data'), sep=r'\s+', header=None,
            names=['label', 'timestamp'],
        )
    if X.shape != (1567, 590) or labels.shape != (1567, 2):
        raise ValueError('Unexpected SECOM data shape.')
    if labels['label'].value_counts().to_dict() != {-1: 1463, 1: 104}:
        raise ValueError('Unexpected SECOM labels.')
    timestamps = pd.to_datetime(labels['timestamp'], format='%d/%m/%Y %H:%M:%S', errors='raise')
    if timestamps.isna().any():
        raise ValueError('SECOM timestamps must be present.')
    X.columns = [f'sensor_{index:03d}' for index in X.columns]
    y = labels['label'].replace(-1, 0).rename('defect')
    return X, y, timestamps


def split_indices(y, random_state=SEED):
    """Stratified 60/20/20 split, before fitting any data-dependent operation."""
    indices = np.arange(len(y))
    development, test = train_test_split(
        indices, test_size=0.2, stratify=y, random_state=random_state,
    )
    train, validation = train_test_split(
        development, test_size=0.25, stratify=np.asarray(y)[development],
        random_state=random_state,
    )
    return {'train': train, 'validation': validation, 'test': test}
