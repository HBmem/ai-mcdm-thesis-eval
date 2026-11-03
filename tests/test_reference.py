"""Analytical and behavioral tests independent of the legacy service."""
from copy import deepcopy
from itertools import combinations
import unittest

import numpy as np

from src.load_experiments import load_experiments
from src.reference import SAATY, ahp, compare, evaluate, quantize_ratio, topsis, validate_ratio
from src.runner import normalized_inputs


class ReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inputs = normalized_inputs(load_experiments()[0])

    def test_saaty_values_and_quantization(self):
        for value in SAATY:
            validate_ratio(value)
            self.assertEqual(quantize_ratio(value), value)
        for value in (True, False, 0, -1, float('nan'), float('inf'), 1.5, '3'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_ratio(value)
        self.assertEqual(quantize_ratio(np.sqrt(2)), 1.)
        self.assertEqual(quantize_ratio(1 / np.sqrt(2)), 1.)
        self.assertEqual(quantize_ratio(.55/.15), 4.)

    def test_consistent_weights_for_all_derivations_and_aliases(self):
        w = np.array([.6, .3, .1])
        a = w[:, None] / w[None, :]
        for wd in ('mean', 'm', 'geometric', 'g', 'max_eigen', 'me'):
            result = ahp(a, wd)
            np.testing.assert_allclose(result['weights'], w, atol=1e-12)
            self.assertAlmostEqual(result['consistency_ratio'], 0.)
            self.assertFalse(result['flagged'])
        two = ahp([[1, 3], [1/3, 1]], 'mean')
        np.testing.assert_allclose(two['weights'], [.75, .25])
        self.assertEqual(two['consistency_ratio'], 0.)

    def test_distinct_derivations_and_explicit_cr_convention(self):
        a = np.array([[1, 3, 5, 7], [1/3, 1, 2, 5], [1/5, 1/2, 1, 9], [1/7, 1/5, 1/9, 1]])
        results = [ahp(a, wd) for wd in ('mean', 'geometric', 'max_eigen')]
        expected_mean = (a / a.sum(axis=0)).mean(axis=1)
        np.testing.assert_allclose(results[0]['weights'], expected_mean)
        for r in results[:2]:
            w = np.array(r['weights'])
            self.assertAlmostEqual(r['lambda_max'], np.mean((a @ w) / w))
            self.assertEqual(r['ri'], .9)
        self.assertAlmostEqual(results[2]['lambda_max'], max(np.linalg.eigvals(a).real))
        for left, right in combinations(results, 2):
            self.assertFalse(np.allclose(left['weights'], right['weights']))

    def test_invalid_ahp_and_wd(self):
        for a in ([[1]], [[1, 2], [1, 1]], [[1, 0], [0, 1]], [[1, np.inf], [0, 1]], np.ones((2, 3))):
            with self.assertRaises(ValueError):
                ahp(a)
        with self.assertRaises(ValueError):
            ahp(np.ones((3, 3)), 'unsupported')

    def test_analytical_topsis_cost_and_benefit(self):
        alternatives = [{'id': 'good', 'values': {'cost': 1., 'benefit': 3.}}, {'id': 'bad', 'values': {'cost': 3., 'benefit': 1.}}]
        criteria = [{'id': 'cost', 'direction': 'min'}, {'id': 'benefit', 'direction': 'max'}]
        result = topsis(alternatives, criteria, [.5, .5])
        self.assertEqual([(r['alternative_id'], r['score']) for r in result['rankings']], [('good', 1.), ('bad', 0.)])
        np.testing.assert_allclose(result['normalized_matrix']['values'], np.array([[1, 3], [3, 1]]) / np.sqrt(10))
        self.assertEqual(result['decision_matrix']['row_ids'], ['good', 'bad'])
        self.assertEqual(result['decision_matrix']['column_ids'], ['cost', 'benefit'])

    def test_zero_columns_and_identical_alternatives(self):
        alternatives = [{'id': key, 'values': {'a': 0., 'b': 4.}} for key in ('Z', 'A', 'B')]
        result = topsis(alternatives, [{'id': 'a', 'direction': 'max'}, {'id': 'b', 'direction': 'min'}], [.5, .5])
        self.assertEqual([r['alternative_id'] for r in result['rankings']], ['Z', 'A', 'B'])
        self.assertTrue(all(r['score'] == .5 and r['statistical_rank'] == 2 for r in result['rankings']))
        self.assertTrue(np.all(np.array(result['normalized_matrix']['values'])[:, 0] == 0))

    def test_invalid_topsis(self):
        a = [{'id': 'a', 'values': {'x': 1}}]
        c = [{'id': 'x', 'direction': 'min'}]
        for w in ([0], [-1], [np.nan], [np.inf], [1, 2]):
            with self.assertRaises(ValueError):
                topsis(a, c, w)
        with self.assertRaises(ValueError):
            topsis([{'id':'a','values':{'x':np.inf}}], c, [1])

    def test_aggregation_participant_counts_voting_and_flags(self):
        x = deepcopy(self.inputs)
        # One group favors the left criterion in each pair.
        x['participants'][0]['judgments'] = {key: 9. for key in x['participants'][0]['judgments']}
        initial = evaluate(x)
        gid = x['participants'][0]['group_id']
        first = deepcopy(x['participants'][0])
        for i in range(9):
            x['participants'].append({**deepcopy(first), 'id': f'replica{i}'})
        replicated = evaluate(x)
        np.testing.assert_allclose(initial['preferences']['session']['matrix']['values'], replicated['preferences']['session']['matrix']['values'], atol=1e-12)
        x['groups'][0]['voting_power'] = .9
        changed = evaluate(x)
        self.assertFalse(np.allclose(initial['applied_weights']['C1'], changed['applied_weights']['C1']))
        self.assertEqual(sum(p['flagged'] for p in replicated['preferences']['participants']), 10)
        self.assertTrue(next(g for g in replicated['preferences']['groups'] if g['id']==gid)['flagged'])
        # Session aggregation can mask participant/group inconsistency.
        self.assertFalse(replicated['preferences']['session']['flagged'])
        x['consistency_threshold'] = 100.
        self.assertFalse(any(p['flagged'] for p in evaluate(x)['preferences']['participants']))

    def test_empty_group_and_nonpositive_active_power(self):
        x = deepcopy(self.inputs)
        x['participants'] = x['participants'][:1]
        result = evaluate(x)
        self.assertEqual(len(result['preferences']['groups']), 1)
        self.assertEqual(result['preferences']['groups'][0]['effective_voting_power'], 1.)
        x['groups'][0]['voting_power'] = 0
        with self.assertRaises(ValueError):
            evaluate(x)

    @staticmethod
    def trace(order, weights=None):
        return {'applied_weights': weights or {'C1': .5, 'C2': .5}, 'topsis': {'rankings': [
            {'alternative_id': key, 'score': 1-rank/10, 'rank': i, 'statistical_rank': rank}
            for i, (key, rank) in enumerate(order, 1)]}}

    def test_comparisons_align_ids_and_distinguish_shifts_reversals_ties(self):
        before = self.trace([('A', 1), ('B', 2), ('C', 3)])
        shifted = self.trace([('B', 1), ('C', 2)])
        c = compare(shifted, before)
        self.assertEqual(c['reversals'], [])
        self.assertEqual(c['rank_deltas'], {'B': -1, 'C': -1})
        self.assertAlmostEqual(c['spearman_rho'], 1.)
        reversed_ = compare(self.trace([('C', 1), ('B', 2)]), before)
        self.assertEqual(reversed_['reversals'], [['B', 'C']])
        self.assertAlmostEqual(reversed_['kendall_tau_b'], -1.)
        tied = compare(self.trace([('A', 1.5), ('B', 1.5), ('C', 3)]), before)
        self.assertEqual(tied['tie_transitions'], [['A', 'B']])
        self.assertEqual(tied['reversals'], [])
        undefined = compare(self.trace([('A', 1)]), before)
        self.assertIsNone(undefined['spearman_rho'])
        self.assertTrue(undefined['undefined_reason'])


if __name__ == '__main__':
    unittest.main()
