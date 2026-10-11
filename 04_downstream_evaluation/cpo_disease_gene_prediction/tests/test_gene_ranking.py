import unittest

import numpy as np

from uhpo_prediction.gene_rank import _rank_interval


class GeneRankingTests(unittest.TestCase):
    def test_gene_rank_interval_preserves_ties(self):
        scores = np.asarray([0.9, 0.8, 0.8, 0.1, np.nan, 0.0])
        self.assertEqual(_rank_interval(scores, 0.8), (2, 3, 2))
