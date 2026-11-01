"""Reproducible SECOM analysis with training-only preprocessing and threshold selection."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import tempfile
from urllib.request import urlopen
import zipfile

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score, classification_report, confusion_matrix,
    precision_recall_curve, roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted, validate_data
from xgboost import XGBClassifier

SEED = 42
DATA_URL = 'https://archive.ics.uci.edu/static/public/179/secom.zip'
DATA_SHA256 = 'eea568baf3c2229096d7d294cf0b096b5502bd96d92c0b80a65b84714059be8e'
PROJECT_DIR = Path(__file__).resolve().parent
PARAM_GRID = {
    'xgb__max_depth': [3, 5, 7],
    'xgb__learning_rate': [0.01, 0.1, 0.2],
    'xgb__n_estimators': [50, 100, 200],
}


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


def make_pipeline():
    return Pipeline([
        ('missing', MissingRateFilter(0.4)),
        ('imputer', SimpleImputer(strategy='median')),
        ('variance', VarianceThreshold(threshold=0.05)),
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=SEED)),
        ('xgb', XGBClassifier(
            eval_metric='logloss', random_state=SEED, n_jobs=1,
            tree_method='hist', scale_pos_weight=1,
        )),
    ])


def select_threshold(y_validation, probabilities):
    """Maximize validation defect F1; use the highest threshold for exact ties."""
    labels = np.asarray(y_validation)
    probabilities = np.asarray(probabilities, dtype=float)
    if labels.ndim != 1 or probabilities.shape != labels.shape or set(np.unique(labels)) != {0, 1}:
        raise ValueError('Threshold selection requires matching 1-D arrays and both binary classes.')
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError('Probabilities must be finite and in [0, 1].')
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    denominator = precision[:-1] + recall[:-1]
    f1 = np.divide(2 * precision[:-1] * recall[:-1], denominator,
                   out=np.zeros_like(denominator), where=denominator > 0)
    # The final precision/recall point has no corresponding threshold.
    best_index = np.flatnonzero(f1 == f1.max())[-1]
    return {
        'threshold': float(thresholds[best_index]),
        'validation_f1': float(f1[best_index]),
        'validation_precision': float(precision[best_index]),
        'validation_recall': float(recall[best_index]),
        'selection_rule': 'Maximum validation defect F1; highest threshold for exact ties',
        'comparison': '>=',
    }


def evaluate_predictions(y, probabilities, threshold):
    predictions = (np.asarray(probabilities) >= threshold).astype(int)
    return {
        'threshold': float(threshold),
        'confusion_matrix': confusion_matrix(y, predictions, labels=[0, 1]).tolist(),
        'classification_report': classification_report(
            y, predictions, labels=[0, 1], output_dict=True, zero_division=0,
        ),
    }


def write_predictions(path, indices, timestamps, y, probabilities, threshold):
    """Preserve float32 model probabilities exactly when exporting decimal CSV."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    pd.DataFrame({
        'source_row': indices, 'timestamp': timestamps.iloc[indices].to_numpy(),
        'actual_defect': y.iloc[indices].to_numpy(), 'probability_defect': probabilities,
        'prediction': (probabilities >= threshold).astype(int),
    }).to_csv(path, index=False, float_format='%.17g')


def _plots(y_validation, validation_probabilities, selection, test_metrics, output_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    precision, recall, thresholds = precision_recall_curve(y_validation, validation_probabilities)
    denominator = precision[:-1] + recall[:-1]
    f1 = np.divide(2 * precision[:-1] * recall[:-1], denominator,
                   out=np.zeros_like(denominator), where=denominator > 0)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, precision[:-1], label='Precision')
    ax.plot(thresholds, recall[:-1], label='Recall')
    ax.plot(thresholds, f1, label='F1')
    ax.axvline(selection['threshold'], color='black', linestyle='--', label='Selected on validation')
    ax.set(xlabel='Probability threshold (>=)', ylabel='Score', ylim=(0, 1.05),
           title='Threshold selection: validation split only')
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / 'validation_threshold.png', dpi=150)
    plt.close(fig)

    matrix = np.asarray(test_metrics['confusion_matrix'])
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(matrix, cmap='Blues')
    for row in range(2):
        for col in range(2):
            ax.text(col, row, str(matrix[row, col]), ha='center', va='center',
                    color='white' if matrix[row, col] > matrix.max()/2 else 'black', fontsize=15)
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=['Normal', 'Defect'],
           yticklabels=['Normal', 'Defect'], xlabel='Predicted', ylabel='Actual',
           title='Test split: frozen validation threshold')
    fig.tight_layout()
    fig.savefig(output_dir / 'test_confusion_matrix.png', dpi=150)
    plt.close(fig)


def run_analysis(data_dir=None, output_dir=None, jobs=2):
    output_dir = Path(output_dir) if output_dir is not None else PROJECT_DIR / 'results'
    output_dir.mkdir(parents=True, exist_ok=True)
    X, y, timestamps = load_secom(data_dir)
    splits = split_indices(y)
    train, validation, test = (splits[name] for name in ['train', 'validation', 'test'])
    search = GridSearchCV(
        make_pipeline(), PARAM_GRID, scoring='average_precision',
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED),
        n_jobs=jobs, pre_dispatch=jobs if jobs > 0 else '2*n_jobs',
        error_score='raise', refit=True, verbose=1,
    )
    search.fit(X.iloc[train], y.iloc[train])
    model = search.best_estimator_
    validation_probabilities = model.predict_proba(X.iloc[validation])[:, 1]
    selection = select_threshold(y.iloc[validation], validation_probabilities)
    # Freeze both model and threshold. No fitting or selection uses the test split.
    test_probabilities = model.predict_proba(X.iloc[test])[:, 1]
    test_selected = evaluate_predictions(y.iloc[test], test_probabilities, selection['threshold'])
    test_default = evaluate_predictions(y.iloc[test], test_probabilities, 0.5)
    report = {
        'completed_at': datetime.now(timezone.utc).isoformat(),
        'data': {'source': DATA_URL, 'archive_sha256': DATA_SHA256,
                 'samples': len(y), 'sensors': X.shape[1], 'defects': int(y.sum()),
                 'yield_rate': float((y == 0).mean()), 'defect_rate': float(y.mean())},
        'seed': SEED,
        'splits': {name: {'samples': len(indices), 'defects': int(y.iloc[indices].sum())}
                   for name, indices in splits.items()},
        'search': {'scoring': 'average_precision', 'candidates': len(search.cv_results_['params']),
                   'folds': search.n_splits_, 'cv_fits': len(search.cv_results_['params'])*search.n_splits_,
                   'best_cv_average_precision': float(search.best_score_),
                   'best_parameters': search.best_params_},
        'preprocessing': {'retained_after_missing_filter': int(model.named_steps['missing'].support_.sum()),
                          'selected_features': int(model.named_steps['variance'].get_support().sum()),
                          'fit_scope': 'CV training fold; final refit on training split only'},
        'threshold_selection': selection,
        'test_selected_threshold': test_selected,
        'test_default_threshold': test_default,
        'test_average_precision': float(average_precision_score(y.iloc[test], test_probabilities)),
        'test_roc_auc': float(roc_auc_score(y.iloc[test], test_probabilities)),
        'limitations': [
            'The same seed-42 test split was inspected in the earlier flawed analysis; this is not a new external validation.',
            'Validation and test each contain only 21 defects; threshold and metric uncertainty is substantial.',
            'Random stratified splitting does not assess future production, batch effects or temporal drift.',
            'Maximum validation F1 is a transparent heuristic, not a validated business-cost optimum.',
        ],
        'versions': {name: importlib.metadata.version(name) for name in
                     ['numpy', 'pandas', 'scipy', 'scikit-learn', 'imbalanced-learn', 'matplotlib']},
    }
    import xgboost
    report['versions']['xgboost'] = xgboost.__version__
    (output_dir / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n')
    pd.DataFrame(search.cv_results_).to_csv(output_dir / 'grid_search_results.csv', index=False)
    for name, indices, probabilities in [
        ('validation', validation, validation_probabilities), ('test', test, test_probabilities),
    ]:
        write_predictions(output_dir / f'{name}_predictions.csv', indices, timestamps, y,
                          probabilities, selection['threshold'])
    _plots(y.iloc[validation], validation_probabilities, selection, test_selected, output_dir)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=None)
    parser.add_argument('--output-dir', type=Path, default=PROJECT_DIR / 'results')
    parser.add_argument('--jobs', type=int, default=2)
    args = parser.parse_args()
    if args.jobs == 0:
        parser.error('--jobs must be nonzero')
    print(json.dumps(run_analysis(args.data_dir, args.output_dir, args.jobs), indent=2))
