"""Train the CV-selected model with frozen parameters and threshold.

No search is performed. Historical test rows are excluded from fitting/evaluation.
"""
import argparse
import json
import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.feature_selection import SelectPercentile, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             precision_score, recall_score, f1_score)
from model_components import FoldWeightedClassifier
from secom_data import MissingRateFilter, load_secom, split_indices

ROOT = Path(__file__).resolve().parent


def make_final_pipeline(config):
    if config['family'] != 'xgboost':
        raise ValueError('This final workflow is frozen to the CV-selected XGBoost model.')
    from xgboost import XGBClassifier
    estimator = XGBClassifier(tree_method='hist', n_jobs=1, random_state=42)
    model = Pipeline([
        ('missing', MissingRateFilter()), ('imputer', SimpleImputer()),
        ('variance', VarianceThreshold()), ('scale', StandardScaler()),
        ('select', SelectPercentile(f_classif)), ('sample', 'passthrough'),
        ('model', FoldWeightedClassifier(estimator)),
    ])
    params = dict(config['parameters'])
    params['sample'] = SMOTE(random_state=42, k_neighbors=3) if params['sample'] == 'smote' else 'passthrough'
    return model.set_params(**params)


def run(data_dir, output_dir, config_path=ROOT/'final_config.json'):
    config = json.loads(Path(config_path).read_text())
    X, y, timestamps = load_secom(data_dir)
    splits = split_indices(y)
    train, validation = splits['train'], splits['validation']
    model = make_final_pipeline(config).fit(X.iloc[train], y.iloc[train])
    p = model.predict_proba(X.iloc[validation])[:, 1].astype(np.float64)
    threshold = config['threshold']
    predictions = p >= threshold
    tn, fp, fn, tp = confusion_matrix(y.iloc[validation], predictions, labels=[0, 1]).ravel()
    result = dict(model=config['family'], evaluation='historically inspected validation; threshold tuning set',
                  independent_evaluation=False, historical_test_evaluated=False,
                  recorded_nested_cv=config.get('nested_cv'),
                  accuracy=float(accuracy_score(y.iloc[validation], predictions)),
                  precision=float(precision_score(y.iloc[validation], predictions, zero_division=0)),
                  recall=float(recall_score(y.iloc[validation], predictions, zero_division=0)),
                  f1=float(f1_score(y.iloc[validation], predictions, zero_division=0)),
                  ap=float(average_precision_score(y.iloc[validation], p)),
                  false_positives=int(fp), false_positive_rate=float(fp/(tn+fp)),
                  confusion_matrix=[[int(tn), int(fp)], [int(fn), int(tp)]], threshold=threshold)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output/'metrics.json').write_text(json.dumps(result, indent=2)+'\n')
    pd.DataFrame({'source_row': validation, 'timestamp': timestamps.iloc[validation].to_numpy(),
                  'actual_defect': y.iloc[validation].to_numpy(), 'probability_defect': p,
                  'prediction': predictions.astype(int)}).to_csv(output/'validation_predictions.csv', index=False, float_format='%.17g')
    joblib.dump({'model': model, 'threshold': threshold, 'features': X.columns.tolist(), 'config': config, 'family': config['family']}, output/'model.joblib')
    os.environ.setdefault('MPLCONFIGDIR', str(output/'.matplotlib'))
    os.environ.setdefault('XDG_CACHE_HOME', str(output/'.cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    ConfusionMatrixDisplay(np.array(result['confusion_matrix']), display_labels=['Normal', 'Defect']).plot(cmap='Blues')
    plt.title('Validation: threshold tuning data')
    plt.tight_layout()
    plt.savefig(output/'confusion_matrix.png', dpi=150)
    plt.close()
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--data-dir', type=Path, default=None)
    parser.add_argument('--output-dir', type=Path, default=ROOT/'final_results')
    args = parser.parse_args()
    run(args.data_dir, args.output_dir)
