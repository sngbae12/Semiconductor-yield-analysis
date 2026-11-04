"""Training/inference artifact compatibility and frozen preprocessing checks."""
import tempfile
from pathlib import Path
import unittest
import joblib
import numpy as np
import pandas as pd
from final_model import make_final_pipeline


class FinalPipelineTests(unittest.TestCase):
    def test_serialized_pipeline_keeps_predictions_and_training_statistics(self):
        config = {'family': 'xgboost', 'parameters': {
            'missing__max_missing_rate': .4, 'sample': 'passthrough',
            'select__percentile': 50, 'model__balanced': True,
            'model__estimator__n_estimators': 5, 'model__estimator__max_depth': 2}}
        rng = np.random.default_rng(55)
        X = pd.DataFrame(rng.normal(size=(80, 10)), columns=[f's{i}' for i in range(10)])
        X.iloc[::5, 0] = np.nan
        y = np.tile([0, 0, 0, 1], 20)
        model = make_final_pipeline(config).fit(X.iloc[:60], y[:60])
        medians = model.named_steps['imputer'].statistics_.copy()
        predicted = model.predict_proba(X.iloc[60:])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'model.joblib'
            joblib.dump(model, path)
            restored = joblib.load(path)
        np.testing.assert_array_equal(predicted, restored.predict_proba(X.iloc[60:]))
        np.testing.assert_array_equal(medians, restored.named_steps['imputer'].statistics_)
        with self.assertRaises(ValueError):
            restored.predict_proba(X.iloc[60:, ::-1])


if __name__ == '__main__':
    unittest.main()
