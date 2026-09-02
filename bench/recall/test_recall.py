"""Focused tests for the recall harness's integrity checks and row mapping."""

import os
import tempfile
import unittest

import numpy as np

from corpus import unit_normalise
from ground_truth import merge_top_k
from recall_sweep import check_ids, refuse_existing_outputs


class RecallHarnessTests(unittest.TestCase):
    """Exercise helpers whose failure can invalidate a reported recall value."""

    def test_unit_normalise_preserves_zero_rows(self):
        matrix = np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32)

        normalised = unit_normalise(matrix)

        np.testing.assert_allclose(normalised[0], np.array([0.6, 0.8], dtype=np.float32))
        np.testing.assert_array_equal(normalised[1], np.array([0.0, 0.0], dtype=np.float32))

    def test_merge_top_k_preserves_ids_across_shards(self):
        best_ids = np.array([[10, 11], [10, 11]], dtype=np.int64)
        best_scores = np.array([[0.9, 0.4], [0.8, 0.7]], dtype=np.float32)
        new_ids = np.array([[20, 21], [20, 21]], dtype=np.int64)
        new_scores = np.array([[0.8, 0.7], [0.95, 0.1]], dtype=np.float32)

        ids, scores = merge_top_k(best_ids, best_scores, new_ids, new_scores, 2)

        np.testing.assert_array_equal(ids, np.array([[10, 20], [20, 10]], dtype=np.int64))
        np.testing.assert_allclose(scores, np.array([[0.9, 0.8], [0.95, 0.8]], dtype=np.float32))

    def test_check_ids_requires_k_unique_results(self):
        self.assertIsNone(check_ids([1, 2, 3], 3))
        self.assertEqual(check_ids([1, 2], 3), "returned 2 results, asked for 3")
        self.assertEqual(
            check_ids([1, 1, 2], 3),
            "returned 3 results holding only 2 distinct ids",
        )

    def test_existing_success_output_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            output = os.path.join(directory, "result.json")
            with open(output, "w", encoding="utf-8") as handle:
                handle.write("old result")

            with self.assertRaisesRegex(SystemExit, "refusing to reuse"):
                refuse_existing_outputs(output)

    def test_existing_rejected_output_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            output = os.path.join(directory, "result.json")
            with open(output + ".rejected", "w", encoding="utf-8") as handle:
                handle.write("old rejection")

            with self.assertRaisesRegex(SystemExit, "refusing to reuse"):
                refuse_existing_outputs(output)


if __name__ == "__main__":
    unittest.main()
