"""Synthetic fixtures validate experiment mechanics, never model performance."""
import unittest
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from experiments.compare_models import candidates, models, pipeline, metrics


class ComparisonTests(unittest.TestCase):
    def test_search_covers_exclusive_imbalance_strategies(self):
        trials = candidates({'C': [0.1, 1]})
        self.assertEqual(len(trials), 24)
        counts = {'none': 0, 'weighted': 0, 'smote': 0}
        for trial in trials:
            smote = trial['sample'][0] != 'passthrough'
            weighted = trial['model__balanced'][0]
            self.assertFalse(smote and weighted)
            counts['smote' if smote else 'weighted' if weighted else 'none'] += 1
        self.assertEqual(set(counts.values()), {8})

    def test_all_five_estimators_fit_through_cloned_pipeline(self):
        rng = np.random.default_rng(7)
        X = pd.DataFrame(rng.normal(size=(90, 8)))
        X.iloc[::8, 0] = np.nan
        y = np.tile([0, 0, 0, 0, 1], 18)
        for name, (estimator, space) in models().items():
            for index in [0, 8, 16]:
                with self.subTest(model=name, strategy=index):
                    model = clone(pipeline(estimator))
                    trial = {k: v[0] for k, v in candidates(space)[index].items()}
                    model.set_params(**trial)
                    if name == 'catboost':
                        model.set_params(model__estimator__iterations=5)
                    elif name != 'logistic_regression':
                        model.set_params(model__estimator__n_estimators=5)
                    model.fit(X.iloc[:75], y[:75])
                    probabilities = model.predict_proba(X.iloc[75:])
                    self.assertEqual(probabilities.shape, (15, 2))
                    self.assertTrue(np.isfinite(probabilities).all())
                    np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-6)

    def test_metrics_preserve_false_positive_counts(self):
        result = metrics([0, 0, 0, 1, 1], np.array([.1, .8, .2, .8, .3]), .5)
        self.assertEqual(result['confusion_matrix'], [[2, 1], [1, 1]])
        self.assertEqual(result['false_positives'], 1)
        self.assertEqual(result['false_positive_rate'], 1/3)
        self.assertEqual(result['recall'], .5)


if __name__ == '__main__':
    unittest.main()
