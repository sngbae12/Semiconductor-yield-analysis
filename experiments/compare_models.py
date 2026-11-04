"""Training-only nested CV; historical test rows are never evaluated.

Run from the repository root: python -m experiments.compare_models
No winner is declared until this script completes on checksum-verified real data.
"""
import argparse
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectPercentile, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             precision_score, recall_score, f1_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

from secom_analysis import MissingRateFilter, load_secom, split_indices, select_threshold


from model_components import FoldWeightedClassifier


def models():
    return {
        'xgboost': (XGBClassifier(tree_method='hist', n_jobs=1, random_state=42), {
            'max_depth': [2, 3, 5], 'n_estimators': [100, 300],
            'learning_rate': [.02, .08], 'reg_lambda': [1., 10.],
            'min_child_weight': [1, 5], 'colsample_bytree': [.6, 1.]}),
        'catboost': (CatBoostClassifier(verbose=False, allow_writing_files=False,
                                       thread_count=1, random_seed=42), {
            'depth': [3, 5, 7], 'iterations': [150, 350],
            'learning_rate': [.02, .08], 'l2_leaf_reg': [3., 15.]}),
        'lightgbm': (LGBMClassifier(verbosity=-1, n_jobs=1, random_state=42), {
            'num_leaves': [7, 15, 31], 'n_estimators': [100, 300],
            'learning_rate': [.02, .08], 'reg_lambda': [1., 10.],
            'min_child_samples': [15, 40], 'colsample_bytree': [.6, 1.]}),
        'random_forest': (RandomForestClassifier(n_jobs=1, random_state=42), {
            'n_estimators': [250, 500], 'max_depth': [4, 8, None],
            'min_samples_leaf': [2, 5, 10], 'max_features': ['sqrt', .3]}),
        'logistic_regression': (LogisticRegression(max_iter=3000, random_state=42), {
            'C': [.001, .01, .1, 1., 10.], 'solver': ['liblinear'],
            'penalty': ['l1', 'l2']}),
    }


def pipeline(estimator):
    return Pipeline([
        ('missing', MissingRateFilter()), ('imputer', SimpleImputer()),
        ('variance', VarianceThreshold()), ('scale', StandardScaler()),
        ('select', SelectPercentile(f_classif)), ('sample', 'passthrough'),
        ('model', FoldWeightedClassifier(estimator)),
    ])


def candidates(space, per_strategy=8):
    """Equal search budget, explicitly cover all three imbalance strategies."""
    rng = np.random.default_rng(42)
    trials = []
    for strategy in ('none', 'weighted', 'smote'):
        for i in range(per_strategy):
            params = {
                'missing__max_missing_rate': [.2, .4][i % 2],
                'imputer__strategy': ['median', 'mean'][i // 2 % 2],
                'variance__threshold': [0., .05][i // 4 % 2],
                'select__percentile': [25, 50, 100][i % 3],
                'sample': SMOTE(random_state=42, k_neighbors=3) if strategy == 'smote' else 'passthrough',
                'model__balanced': strategy == 'weighted',
            }
            params.update({'model__estimator__'+k: values[int(rng.integers(len(values)))]
                           for k, values in space.items()})
            trials.append({k: [v] for k, v in params.items()})
    return trials


def metrics(y, probabilities, threshold):
    pred = probabilities >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(ap=float(average_precision_score(y, probabilities)),
                accuracy=float(accuracy_score(y, pred)), precision=float(precision_score(y, pred, zero_division=0)),
                recall=float(recall_score(y, pred, zero_division=0)), f1=float(f1_score(y, pred, zero_division=0)),
                false_positives=int(fp), false_positive_rate=float(fp/(tn+fp)),
                confusion_matrix=[[int(tn), int(fp)], [int(fn), int(tp)]], threshold=float(threshold))


def search_for(estimator, space, X, y, jobs):
    search = GridSearchCV(pipeline(estimator), candidates(space),
                         scoring='average_precision', cv=StratifiedKFold(3, shuffle=True, random_state=42),
                         n_jobs=jobs, pre_dispatch=jobs, error_score='raise', refit=True)
    return search.fit(X, y)


def run(output, jobs=2, data_dir=None):
    # Fail before creating results if real data are unavailable/corrupt.
    X, y, _ = load_secom(data_dir)
    splits = split_indices(y)
    train, validation = splits['train'], splits['validation']
    assert not set(train) & set(splits['test'])
    assert not set(validation) & set(splits['test'])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output/'split_rows.json').write_text(json.dumps({k: v.tolist() for k, v in splits.items()}, indent=2))
    Xt, yt = X.iloc[train], y.iloc[train]
    outer = list(StratifiedKFold(3, shuffle=True, random_state=2026).split(Xt, yt))
    summary, fitted = [], {}
    for name, (estimator, space) in models().items():
        folds = []
        for fold, (a, b) in enumerate(outer):
            tuned = search_for(estimator, space, Xt.iloc[a], yt.iloc[a], jobs)
            inner = StratifiedKFold(3, shuffle=True, random_state=42)
            p = cross_val_predict(clone(tuned.best_estimator_), Xt.iloc[a], yt.iloc[a],
                                  cv=inner, method='predict_proba', n_jobs=jobs)[:, 1]
            threshold = select_threshold(yt.iloc[a], p)['threshold']
            result = metrics(yt.iloc[b], tuned.predict_proba(Xt.iloc[b])[:, 1], threshold)
            result['fold'] = fold
            folds.append(result)
            print(name, 'outer', fold, result, flush=True)
        tuned = search_for(estimator, space, Xt, yt, jobs)
        pd.DataFrame(tuned.cv_results_).to_csv(output/f'{name}_search.csv', index=False)
        serial = {k: ('smote' if isinstance(v, SMOTE) else v) for k, v in tuned.best_params_.items()}
        (output/f'{name}_parameters.json').write_text(json.dumps(serial, indent=2))
        (output/f'{name}_nested.json').write_text(json.dumps(folds, indent=2))
        row = dict(model=name, nested_cv_ap=np.mean([f['ap'] for f in folds]),
                   nested_cv_ap_std=np.std([f['ap'] for f in folds], ddof=1),
                   search_cv_ap=tuned.best_score_, best_parameters=repr(tuned.best_params_))
        for key in ('accuracy', 'precision', 'recall', 'f1', 'false_positive_rate'):
            row['nested_cv_'+key] = np.mean([f[key] for f in folds])
        summary.append(row)
        fitted[name] = tuned.best_estimator_
        pd.DataFrame(summary).to_csv(output/'training_comparison.csv', index=False)
    # Freeze family selection BEFORE consulting validation labels or probabilities.
    winner = max(summary, key=lambda row: row['nested_cv_ap'])['model']
    (output/'selected_family.json').write_text(json.dumps({'model': winner, 'criterion': 'maximum mean nested CV AP'}))
    validation_rows = []
    for name, model in fitted.items():
        p = model.predict_proba(X.iloc[validation])[:, 1]
        threshold = select_threshold(y.iloc[validation], p)['threshold']
        row = dict(model=name, **metrics(y.iloc[validation], p, threshold))
        validation_rows.append(row)
        pd.DataFrame({'source_row': validation, 'actual': y.iloc[validation].to_numpy(),
                      'probability': p}).to_csv(output/f'{name}_validation.csv', index=False, float_format='%.17g')
        # Auditable threshold tradeoffs, not test-guided threshold selection.
        pd.DataFrame([metrics(y.iloc[validation], p, t) for t in np.unique(p)]).to_csv(
            output/f'{name}_thresholds.csv', index=False)
    pd.DataFrame(validation_rows).to_csv(output/'validation_comparison.csv', index=False)
    import joblib
    selected = next(r for r in validation_rows if r['model'] == winner)
    joblib.dump({'model': fitted[winner], 'threshold': selected['threshold'], 'family': winner}, output/'selected_model.joblib')
    report = {'winner': winner, 'validation': selected, 'test_evaluated': False,
              'caveats': ['Validation metrics are threshold-tuning metrics, not independent estimates.',
                          'Inner threshold OOF predictions follow hyperparameter tuning on the same inner data; outer folds remain excluded.',
                          'Selecting a family among five induces selection bias even in reported nested CV.',
                          'The historical validation and test data were previously inspected; no new independent evaluation exists.'],
              'versions': {p: importlib.metadata.version(p) for p in
                           ['numpy', 'pandas', 'scikit-learn', 'catboost', 'lightgbm', 'xgboost-cpu']}}
    (output/'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output-dir', default='experiments/results')
    parser.add_argument('--data-dir', default=None)
    parser.add_argument('--jobs', type=int, default=2)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error('--jobs must be positive')
    run(args.output_dir, args.jobs, args.data_dir)
