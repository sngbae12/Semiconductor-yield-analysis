"""Export the completed, CV-selected configuration without changing the selection."""
import ast
import json
from pathlib import Path
import numpy as np
import pandas as pd


def export(run_dir, destination):
    directory = Path(run_dir)
    report = json.loads((directory/'report.json').read_text())
    table = pd.read_csv(directory/'training_comparison.csv')
    for _, row in table.iterrows():
        # repr contains only scalar hyperparameters and this fixed SMOTE object.
        text = row.best_parameters.replace('SMOTE(k_neighbors=3, random_state=42)', "'smote'")
        parameters = ast.literal_eval(text)
        (directory/f'{row.model}_parameters.json').write_text(json.dumps(parameters, indent=2)+'\n')
    family = report['winner']
    row = table[table.model == family].iloc[0]
    folds = json.loads((directory/f'{family}_nested.json').read_text())
    confusion = np.sum([f['confusion_matrix'] for f in folds], axis=0).tolist()
    nested = {'mean_ap': float(row.nested_cv_ap), 'sd_ap': float(row.nested_cv_ap_std),
              'pooled_confusion_matrix': confusion,
              'note': 'Each outer fold uses its own inner-CV learned threshold; family selection still adds optimism.'}
    for key in ['accuracy', 'precision', 'recall', 'f1', 'false_positive_rate']:
        nested['mean_'+key] = float(row['nested_cv_'+key])
    config = {'family': family,
              'parameters': json.loads((directory/f'{family}_parameters.json').read_text()),
              'threshold': report['validation']['threshold'],
              'selection': 'Highest mean outer 3-fold AP; validation maximum F1, highest threshold for ties',
              'nested_cv': nested, 'experiment': str(directory)}
    Path(destination).write_text(json.dumps(config, indent=2)+'\n')
    print(json.dumps(config, indent=2))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--run-dir', default='experiments/run-001')
    parser.add_argument('--destination', default='final_config.json')
    args = parser.parse_args()
    export(args.run_dir, args.destination)
