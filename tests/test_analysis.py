"""Regression checks for data integrity, leakage prevention, and threshold selection."""
import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_validate

from secom_analysis import (
    MissingRateFilter,
    evaluate_predictions,
    load_secom,
    make_pipeline,
    select_threshold,
    split_indices,
    write_predictions,
)


class PreprocessingTests(unittest.TestCase):
    def test_missing_rate_boundary_and_transform_do_not_refit(self):
        training = pd.DataFrame({
            'boundary': [np.nan, np.nan, 3.0, 4.0, 5.0],
            'over_limit': [np.nan, np.nan, np.nan, 4.0, 5.0],
            'complete': [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        transformer = MissingRateFilter(0.4).fit(training)
        np.testing.assert_array_equal(transformer.support_, [True, False, True])
        learned_rates = transformer.missing_rates_.copy()
        held_out = pd.DataFrame({
            'boundary': [100.0, np.nan], 'over_limit': [1000.0, 2000.0],
            'complete': [np.nan, np.nan],
        })
        transformed = transformer.transform(held_out)
        np.testing.assert_allclose(transformed, [[100.0, np.nan], [np.nan, np.nan]], equal_nan=True)
        np.testing.assert_array_equal(transformer.missing_rates_, learned_rates)
        np.testing.assert_array_equal(transformer.support_, [True, False, True])

    def test_no_surviving_columns_and_invalid_limit_fail(self):
        X = pd.DataFrame({'sensor': [np.nan, np.nan, 1.0]})
        with self.assertRaisesRegex(ValueError, 'No sensors'):
            MissingRateFilter(0.4).fit(X)
        for limit in (-0.1, 1.0, 1.1):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                MissingRateFilter(limit).fit(X)

    def test_schema_mismatch_is_rejected(self):
        X = pd.DataFrame({'first': [1.0, 2.0], 'second': [3.0, 4.0]})
        fitted = MissingRateFilter().fit(X)
        for changed in (X[['second', 'first']], X[['first']], X.rename(columns={'second': 'new'})):
            with self.subTest(columns=list(changed.columns)), self.assertRaises(ValueError):
                fitted.transform(changed)

    def test_each_cv_fold_learns_its_own_missingness_imputation_and_scaling(self):
        X = pd.DataFrame({
            'base': np.arange(90, dtype=float),
            'different_missingness': np.arange(90, dtype=float) * 2,
            'always_missing': np.nan,
        })
        X.loc[np.arange(0, 90, 7), 'base'] = np.nan
        X.loc[:29, 'different_missingness'] = np.nan
        y = np.tile([0, 0, 1], 30)
        # Holding out the first block makes the second sensor fully observed in
        # training. The other two training folds have 50% missingness and drop it.
        folds = []
        for block in range(3):
            held_out = np.arange(block * 30, (block + 1) * 30)
            training = np.setdiff1d(np.arange(90), held_out)
            folds.append((training, held_out))
        pipeline = make_pipeline().set_params(xgb=DummyClassifier(strategy='prior'))
        result = cross_validate(
            pipeline, X, y, cv=folds, scoring='average_precision',
            return_estimator=True, error_score='raise', n_jobs=1,
        )
        self.assertEqual(len(result['estimator']), 3)
        for (training, held_out), fitted in zip(folds, result['estimator']):
            with self.subTest(held_out_first=int(held_out[0])):
                frame = X.iloc[training]
                expected_rates = frame.isna().mean().to_numpy()
                expected_support = expected_rates <= 0.4
                expected_medians = frame.loc[:, expected_support].median().to_numpy()
                imputed = frame.loc[:, expected_support].fillna(frame.loc[:, expected_support].median())
                variance_support = imputed.var(ddof=0).to_numpy() > 0.05
                np.testing.assert_allclose(fitted.named_steps['missing'].missing_rates_, expected_rates)
                np.testing.assert_array_equal(fitted.named_steps['missing'].support_, expected_support)
                np.testing.assert_allclose(fitted.named_steps['imputer'].statistics_, expected_medians)
                np.testing.assert_allclose(
                    fitted.named_steps['scaler'].mean_,
                    imputed.iloc[:, variance_support].mean().to_numpy(),
                )
                learned_medians = fitted.named_steps['imputer'].statistics_.copy()
                learned_means = fitted.named_steps['scaler'].mean_.copy()
                # Predictions on an extreme held-out distribution must not
                # update either learned preprocessing statistic.
                shifted = X.iloc[held_out].copy() + 10000
                probabilities = fitted.predict_proba(shifted)
                self.assertEqual(probabilities.shape, (30, 2))
                np.testing.assert_array_equal(fitted.named_steps['imputer'].statistics_, learned_medians)
                np.testing.assert_array_equal(fitted.named_steps['scaler'].mean_, learned_means)
        supports = [fitted.named_steps['missing'].support_.tolist() for fitted in result['estimator']]
        self.assertEqual(supports, [[True, True, False], [True, False, False], [True, False, False]])

    def test_real_pipeline_predicts_from_raw_unscaled_features(self):
        rng = np.random.default_rng(42)
        X = pd.DataFrame(rng.normal(size=(72, 4)), columns=list('abcd'))
        y = np.tile([0, 0, 1], 24)
        X.loc[::9, 'a'] = np.nan
        pipeline = make_pipeline().set_params(xgb__n_estimators=5, xgb__max_depth=2)
        pipeline.fit(X.iloc[:60], y[:60])
        probabilities = pipeline.predict_proba(X.iloc[60:])
        self.assertEqual(probabilities.shape, (12, 2))
        self.assertTrue(np.isfinite(probabilities).all())
        self.assertTrue(((probabilities >= 0) & (probabilities <= 1)).all())
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)


class SplitAndThresholdTests(unittest.TestCase):
    def test_stratified_splits_are_deterministic_disjoint_and_complete(self):
        y = pd.Series([0] * 1463 + [1] * 104)
        first = split_indices(y)
        second = split_indices(y)
        expected = {'train': (939, 62), 'validation': (314, 21), 'test': (314, 21)}
        for name, (size, defects) in expected.items():
            np.testing.assert_array_equal(first[name], second[name])
            self.assertEqual(len(first[name]), size)
            self.assertEqual(int(y.iloc[first[name]].sum()), defects)
        all_indices = np.concatenate(list(first.values()))
        np.testing.assert_array_equal(np.sort(all_indices), np.arange(len(y)))
        self.assertEqual(len(np.unique(all_indices)), len(y))

    def test_selected_threshold_matches_brute_force_f1_with_repeated_scores(self):
        for seed in range(5):
            with self.subTest(seed=seed):
                rng = np.random.default_rng(seed)
                labels = np.tile([0, 0, 1], 20)
                probabilities = np.round(rng.random(len(labels)), 1)
                selected = select_threshold(labels, probabilities)
                candidates = np.unique(probabilities)
                scores = [f1_score(labels, probabilities >= threshold) for threshold in candidates]
                self.assertAlmostEqual(selected['validation_f1'], max(scores))
                self.assertAlmostEqual(
                    f1_score(labels, probabilities >= selected['threshold']), max(scores),
                )
                self.assertIn(selected['threshold'], candidates)
                self.assertEqual(selected['comparison'], '>=')

    def test_exact_f1_tie_chooses_highest_threshold(self):
        # Thresholds 0.6 and 0.9 both yield F1=2/3; intermediate thresholds are worse.
        selected = select_threshold([1, 0, 0, 1], [0.9, 0.8, 0.7, 0.6])
        self.assertEqual(selected['threshold'], 0.9)
        self.assertAlmostEqual(selected['validation_f1'], 2 / 3)
        self.assertEqual(selected['validation_precision'], 1.0)
        self.assertEqual(selected['validation_recall'], 0.5)

    def test_probability_boundary_is_classified_positive(self):
        result = evaluate_predictions([0, 1, 1, 0], [0.5, 0.5, 0.49, 0.1], 0.5)
        self.assertEqual(result['confusion_matrix'], [[1, 1], [1, 1]])

    def test_threshold_selection_rejects_invalid_inputs(self):
        invalid = [
            ([0, 1], [0.2]),
            ([0, 1], [[0.2], [0.8]]),
            ([1, 1], [0.2, 0.8]),
            ([-1, 1], [0.2, 0.8]),
            ([0, 1], [np.nan, 0.8]),
            ([0, 1], [np.inf, 0.8]),
            ([0, 1], [-0.1, 0.8]),
            ([0, 1], [0.2, 1.1]),
        ]
        for labels, probabilities in invalid:
            with self.subTest(labels=labels, probabilities=probabilities), self.assertRaises(ValueError):
                select_threshold(labels, probabilities)


class PredictionExportTests(unittest.TestCase):
    def test_float32_boundary_round_trip_preserves_values_predictions_and_metrics(self):
        boundary = np.float32(0.41138148)
        threshold = float(boundary)
        probabilities = np.array([
            np.nextafter(boundary, np.float32(0)),
            boundary,
            np.nextafter(boundary, np.float32(1)),
            np.float32(0.9),
        ], dtype=np.float32)
        indices = np.array([4, 1, 5, 0])
        timestamps = pd.Series(pd.date_range('2008-07-19', periods=6, freq='h'))
        labels = pd.Series([0, 1, 0, 0, 0, 1])
        expected_predictions = np.array([0, 1, 1, 1])
        expected_metrics = evaluate_predictions(labels.iloc[indices], probabilities, threshold)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'predictions.csv'
            write_predictions(path, indices, timestamps, labels, probabilities, threshold)
            # pandas' default float parser may lose one ULP even with all
            # decimal digits present. The documented reader uses round_trip.
            loaded = pd.read_csv(path, float_precision='round_trip')
        np.testing.assert_array_equal(loaded.source_row, indices)
        np.testing.assert_array_equal(loaded.actual_defect, labels.iloc[indices])
        np.testing.assert_array_equal(loaded.probability_defect, probabilities.astype(np.float64))
        np.testing.assert_array_equal(loaded.prediction, expected_predictions)
        np.testing.assert_array_equal(
            (loaded.probability_defect >= threshold).astype(int), expected_predictions,
        )
        self.assertEqual(loaded.probability_defect.iloc[1], threshold)
        self.assertEqual(
            evaluate_predictions(loaded.actual_defect, loaded.probability_defect, threshold),
            expected_metrics,
        )


class ArchiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Self-contained fixture with the official schema. Only the test's
        # expected digest is patched; production checksum verification remains on.
        buffer = io.BytesIO()
        sensor_rows = [f'{row} ' + ' '.join(['0'] * 589) for row in range(1567)]
        label_rows = [
            f'{-1 if row < 1463 else 1} "19/07/2008 11:55:00"' for row in range(1567)
        ]
        with zipfile.ZipFile(buffer, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr('secom.data', '\n'.join(sensor_rows) + '\n')
            archive.writestr('secom_labels.data', '\n'.join(label_rows) + '\n')
        cls.payload = buffer.getvalue()
        cls.digest = hashlib.sha256(cls.payload).hexdigest()

    def test_valid_cached_archive_is_verified_parsed_and_never_downloaded(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / 'secom.zip'
            archive.write_bytes(self.payload)
            with patch('secom_analysis.DATA_SHA256', self.digest), patch('secom_analysis.urlopen') as request:
                X, y, timestamps = load_secom(directory)
            request.assert_not_called()
            self.assertEqual(X.shape, (1567, 590))
            self.assertEqual(y.value_counts().to_dict(), {0: 1463, 1: 104})
            self.assertEqual(X.columns[0], 'sensor_000')
            self.assertEqual(X.columns[-1], 'sensor_589')
            self.assertEqual(X.iloc[-1, 0], 1566)
            self.assertEqual(timestamps.iloc[0], pd.Timestamp('2008-07-19 11:55:00'))
            self.assertFalse(timestamps.isna().any())
            self.assertEqual(archive.read_bytes(), self.payload)

    def test_corrupted_cache_is_rejected_without_replacing_or_downloading(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / 'secom.zip'
            corrupted = self.payload + b'corruption'
            archive.write_bytes(corrupted)
            with patch('secom_analysis.DATA_SHA256', self.digest), patch('secom_analysis.urlopen') as request:
                with self.assertRaisesRegex(ValueError, 'SHA-256 mismatch'):
                    load_secom(directory)
            request.assert_not_called()
            self.assertEqual(archive.read_bytes(), corrupted)

    def test_mocked_download_is_verified_before_cache_is_written(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch('secom_analysis.DATA_SHA256', self.digest), patch(
                'secom_analysis.urlopen', return_value=io.BytesIO(self.payload),
            ) as request:
                X, _, _ = load_secom(directory)
            request.assert_called_once()
            self.assertEqual(X.shape, (1567, 590))
            self.assertEqual((Path(directory) / 'secom.zip').read_bytes(), self.payload)
            self.assertEqual([path.name for path in Path(directory).iterdir()], ['secom.zip'])

    def test_invalid_download_does_not_leave_a_cache_file(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch('secom_analysis.DATA_SHA256', self.digest), patch(
                'secom_analysis.urlopen', return_value=io.BytesIO(b'invalid archive'),
            ):
                with self.assertRaisesRegex(ValueError, 'SHA-256 mismatch'):
                    load_secom(directory)
            self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == '__main__':
    unittest.main()
