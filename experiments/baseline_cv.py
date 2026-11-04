"""Baseline on exactly the same outer folds; never evaluate historical test rows."""
import argparse
import json
from pathlib import Path
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from secom_analysis import load_secom, split_indices, make_pipeline, PARAM_GRID, select_threshold
from experiments.compare_models import metrics


def run(output_dir):
    X, y, _ = load_secom()
    splits = split_indices(y)
    Xt, yt = X.iloc[splits['train']], y.iloc[splits['train']]
    folds = []
    for i, (a, b) in enumerate(StratifiedKFold(3, shuffle=True, random_state=2026).split(Xt, yt)):
        search = GridSearchCV(make_pipeline(), PARAM_GRID, scoring='average_precision',
                              cv=StratifiedKFold(3, shuffle=True, random_state=42),
                              n_jobs=2, error_score='raise').fit(Xt.iloc[a], yt.iloc[a])
        p = cross_val_predict(clone(search.best_estimator_), Xt.iloc[a], yt.iloc[a],
                              cv=StratifiedKFold(3, shuffle=True, random_state=42),
                              method='predict_proba', n_jobs=2)[:, 1]
        threshold = select_threshold(yt.iloc[a], p)['threshold']
        result = metrics(yt.iloc[b], search.predict_proba(Xt.iloc[b])[:, 1], threshold)
        result['fold'] = i
        folds.append(result)
        print(result, flush=True)
    # Historically selected baseline parameters, reproduced on original validation.
    model = make_pipeline().set_params(xgb__max_depth=5, xgb__learning_rate=.01,
                                      xgb__n_estimators=200).fit(Xt, yt)
    p = model.predict_proba(X.iloc[splits['validation']])[:, 1]
    labels = y.iloc[splits['validation']]
    threshold = select_threshold(labels, p)['threshold']
    report = {'nested': folds, 'validation': metrics(labels, p, threshold),
              'validation_default': metrics(labels, p, .5)}
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output/'baseline_comparison.json').write_text(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output-dir', default='experiments/run-001')
    args = parser.parse_args()
    run(args.output_dir)
