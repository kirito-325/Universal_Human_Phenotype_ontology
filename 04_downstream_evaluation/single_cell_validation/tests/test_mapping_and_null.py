import unittest

import numpy as np

from cellphenorank_validation.analysis import (
    _expression_match_pools, _module_scores, _sample_expression_matched_indices,
    map_prediction_to_label,
)


class SingleCellMappingTests(unittest.TestCase):
    def test_fine_cell_maps_to_dataset_parent(self):
        rows = [{"dataset_id": "study", "dataset_label": "neuron", "cell_id": "CL:parent"}]
        parents = {"CL:fine": {"CL:middle"}, "CL:middle": {"CL:parent"}}
        self.assertEqual(
            map_prediction_to_label("CL:fine", "study", rows, parents),
            ("neuron", "predicted_child_to_dataset_parent", 2),
        )

    def test_gene_zscore_module_and_matched_null(self):
        expression = np.asarray([
            [1., 2., 0., 4., 8., 2.], [2., 3., 0., 5., 7., 3.],
            [4., 5., 0., 7., 5., 6.], [5., 7., 0., 8., 4., 7.],
        ])
        scores = _module_scores(expression[:, [0, 1]], "gene_zscore")
        self.assertAlmostEqual(float(scores.mean()), 0.0)
        target = np.asarray([0, 1], dtype=int)
        pools, valid = _expression_match_pools(expression, target, bins=2)
        sampled = _sample_expression_matched_indices(pools, valid, target, np.random.default_rng(2026))
        self.assertEqual(len(sampled), len(target))
        self.assertTrue(set(sampled).isdisjoint(set(target)))
