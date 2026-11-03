from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from src.load_experiments import load_experiments, _participants, _processing
from src.reference import SAATY
from src.results import RunResult, safe_artifact
from src.runner import ROOT, Case, build_plan, execute_batch
from src.variations import ExecutionVariation


class RunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.experiments = load_experiments()
        cls.plan = build_plan(cls.experiments)
        cls.path, results = execute_batch(cls.plan, cls.temporary.name)
        cls.results = {r.case_key: r for r in results}

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_complete_suite_and_artifacts(self):
        self.assertEqual(len(self.results), 40)
        self.assertTrue(all(r.status == 'passed' for r in self.results.values()), [(k, r.execution, r.assertions) for k,r in self.results.items() if r.status!='passed'])
        manifest = json.loads((self.path/'manifest.json').read_text())
        self.assertEqual(manifest['status'], 'completed')
        self.assertEqual(len(manifest['runs']), 40)
        for entry in manifest['runs']:
            artifact = RunResult.model_validate_json(safe_artifact(self.path, entry['file']).read_bytes())
            self.assertEqual(artifact.case_key, entry['case_key'])
            self.assertFalse(any(a['status'] in {'error','failed','pending'} for a in artifact.assertions))
            self.assertEqual(artifact.provenance['engine'], 'independent_numpy_reference')
            self.assertFalse(artifact.provenance['poliinsight_validated'])
            self.assertEqual(set(artifact.provenance['resource_hashes']), {'experiment','criteria','alternatives','dataset'})
        for filename in ('summary.csv','rankings.csv','weights.csv','assertions.csv'):
            self.assertGreater((self.path/filename).stat().st_size, 20)
        deferred = [a for r in self.results.values() for a in r.assertions if a['status']=='deferred']
        self.assertEqual(len(deferred), 3)

    def test_schema_versions_and_no_ambiguous_codes(self):
        self.assertEqual(_participants([{'participant_id':'p','group_id':'g','judgments':{'q':4}}], 'x')[0].judgments['q'].ahp_scale, 9)
        for ratio in SAATY:
            p = _participants([{'participant_id':'p','group_id':'g','judgments':{'q':ratio}}], 'x', '2.0')[0]
            self.assertEqual(p.judgments['q'], ratio)
        self.assertEqual(_participants([{'participant_id':'p','group_id':'g','judgments':{'q':4}}], 'x', '2.0')[0].judgments['q'], 4.)
        for value in (True, '4', 1.5):
            with self.assertRaises(ValueError):
                _participants([{'participant_id':'p','group_id':'g','judgments':{'q':value}}], 'x', '2.0')

    def test_processing_conflicts_and_equivalent_duplicate_parameters(self):
        d={'weighting_method':{'method':'ahp','parameters':{'wd':'mean'}},'weighting':{'wd':'geometric'},'ranking_method':'topsis'}
        with self.assertRaisesRegex(ValueError, 'Conflicting'):
            _processing(d)
        d['weighting']['wd']='mean'
        self.assertEqual(_processing(d).weighting_method.parameters, {'wd':'mean'})

    def test_complete_pair_coverage(self):
        from shutil import copytree
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for name in ('experiments','criteria','alternative','datasets'):
                copytree(ROOT/name,root/name)
            source=root/'experiments/S01_uniform_preferences.json'
            d=json.loads(source.read_text())
            d['stakeholders']['participant'][0]['judgments'].pop('q1')
            source.write_text(json.dumps(d))
            with self.assertRaisesRegex(ValueError, 'missing judgments'):
                load_experiments(root/'experiments')
            copytree(ROOT/'experiments',root/'experiments',dirs_exist_ok=True)
            criteria=root/'criteria/transport_v1.json'
            d=json.loads(criteria.read_text())
            d['pairwise_question_order'].pop()
            criteria.write_text(json.dumps(d))
            with self.assertRaisesRegex(ValueError, 'every unique criterion pair'):
                load_experiments(root/'experiments')

    def test_dependency_selection_order_and_aliases(self):
        cases=build_plan(self.experiments, 'S8')
        self.assertEqual(len(cases),21)
        self.assertEqual(cases[0].key,'S01')
        self.assertEqual([c.key for c in cases],[c.key for c in build_plan(self.experiments,'S08')])
        self.assertEqual([c.key for c in build_plan(self.experiments,'S09_REMOVE')],['S01','S09_ADD','S09_REMOVE'])
        with self.assertRaisesRegex(ValueError, 'No matching'):
            build_plan(self.experiments,'S99')

    def test_graph_and_assertion_configuration_errors(self):
        base=deepcopy(self.experiments[0])
        for variation in (
            ExecutionVariation(variant_id='X',description='x',parent='absent'),
            ExecutionVariation(variant_id='X',description='x',parent='X'),
            ExecutionVariation(variant_id='X',description='x',group_judgments={'absent':{}}),
            ExecutionVariation(variant_id='X',description='x',weights={'C1':0.}),
        ):
            e=deepcopy(base);e.execution.variations=[variation]
            with self.subTest(variation=variation), self.assertRaises(ValueError):
                build_plan([e])
        e=deepcopy(base);e.expected_assertions=[{'id':'bad','check':'typo','expected':True}]
        with self.assertRaisesRegex(ValueError,'Unknown assertion'):
            build_plan([e])
        e=deepcopy(base);e.expected_assertions=[{'id':'external','check':'topsis_closeness_and_ranks_match_reference','expected':True}]
        with self.assertRaisesRegex(ValueError,'explicitly set deferred'):
            build_plan([e])

    def test_schema_two_in_memory_serialization(self):
        e=next(e for e in self.experiments if e.id=='S03')
        json.dumps(e.to_dict(),allow_nan=False)
        self.assertTrue(e.execution.variations)

    def test_configured_ahp_method_reaches_every_level(self):
        e=deepcopy(next(e for e in self.experiments if e.variant_id=='S02_MAX'))
        e.expected_assertions=[]
        e.processing.weighting_method.parameters['wd']='max_eigen'
        with tempfile.TemporaryDirectory() as root:
            _,results=execute_batch(build_plan([e]),root)
        r=results[0]
        self.assertEqual(r.status,'passed')
        for item in [*r.preferences['participants'],*r.preferences['groups'],r.preferences['session']]:
            self.assertEqual(item['wd'],'max_eigen')
            self.assertEqual(item['consistency_convention'],'principal_eigenvalue')

    def test_profiles_counts_and_inconsistency(self):
        s4=self.results['S04']
        env=next(g for g in s4.preferences['groups'] if g['id']=='environmental_advocates')
        target=s4.inputs['target_profiles'][env['id']]
        self.assertFalse(np.allclose(target,env['weights']))
        a,b,c=(self.results[k] for k in ('S05_A','S05_B','S05_C'))
        np.testing.assert_allclose(list(a.applied_weights.values()),list(c.applied_weights.values()))
        self.assertFalse(np.allclose(list(a.applied_weights.values()),list(b.applied_weights.values())))
        self.assertEqual(len(c.inputs['participants']),50)
        self.assertEqual(next(g for g in c.preferences['groups'] if g['id']=='environmental_advocates')['effective_voting_power'],.2)
        for key,count in (('S10_A',1),('S10_B',2),('S10_C',10)):
            self.assertEqual(sum(p['flagged'] for p in self.results[key].preferences['participants']),count)
            self.assertEqual(len(self.results[key].preferences['participants']),10)

    def test_weights_reused_restricted_and_overridden(self):
        baseline=self.results['S01']
        self.assertEqual(self.results['S06'].applied_weights,baseline.applied_weights)
        self.assertNotIn(baseline.topsis['rankings'][0]['alternative_id'],[a['id'] for a in self.results['S06'].inputs['alternatives']])
        for cid in ('C1','C2','C3','C4'):
            r=self.results[f'S07_{cid}']
            self.assertNotIn(cid,r.applied_weights)
            np.testing.assert_allclose(list(r.applied_weights.values()),[1/3]*3)
            self.assertEqual(r.preferences['session'],baseline.preferences['session'])
            self.assertTrue(r.preferences['source'].startswith('reused:'))
            self.assertEqual(self.results[f'S08_{cid}_W4'].applied_weights[cid],.85)
        restore=self.results['S09_REMOVE']
        self.assertEqual(restore.topsis,baseline.topsis)
        self.assertEqual(restore.comparison['reference_case_key'],'S01')
        intervals=[r.comparison['winner_change_interval'] for r in self.results.values() if r.comparison and 'winner_change_interval' in r.comparison]
        self.assertTrue(intervals)
        self.assertTrue(all(len(i)==2 and i[0]<i[1] for i in intervals))

    def test_repeat_runs_have_identical_scientific_outputs_and_immutable_batches(self):
        path,repeated=execute_batch(self.plan,self.temporary.name)
        self.assertNotEqual(path,self.path)
        for r in repeated:
            original=self.results[r.case_key]
            self.assertNotEqual(r.run_id,original.run_id)
            self.assertEqual(r.inputs,original.inputs)
            self.assertEqual(r.topsis,original.topsis)
            self.assertEqual(r.applied_weights,original.applied_weights)
            self.assertEqual(r.provenance['input_hash'],original.provenance['input_hash'])
            pref=deepcopy(r.preferences);old=deepcopy(original.preferences)
            pref.pop('source');old.pop('source')
            self.assertEqual(pref,old)

    def test_failed_parent_blocks_descendants_and_other_runs_continue(self):
        plan=build_plan(self.experiments,'S06')
        independent=deepcopy(next(c for c in self.plan if c.key=='S04'))
        independent.variation.parent=None
        from src.reference import evaluate
        calls=0
        def inject(inputs, reused):
            nonlocal calls
            calls+=1
            if calls==1:
                raise ValueError('injected arithmetic failure')
            return evaluate(inputs,reused)
        with tempfile.TemporaryDirectory() as root, patch('src.runner.evaluate',side_effect=inject):
            _,results=execute_batch([*plan,independent],root)
        self.assertEqual([r.status for r in results],['failed','blocked','passed'])
        self.assertIn('injected arithmetic failure',results[0].execution['messages'][0])

    def test_cli_exit_codes_and_foreign_working_directory(self):
        with tempfile.TemporaryDirectory() as root:
            command=[sys.executable,str(ROOT/'run_experiment.py')]
            valid=subprocess.run([*command,'validate','--scenario','S08'],cwd=root,capture_output=True,text=True)
            self.assertEqual(valid.returncode,0,valid.stderr)
            self.assertIn('21 runs',valid.stdout)
            invalid=subprocess.run([*command,'run','--scenario','S99','--output',root],cwd=root,capture_output=True,text=True)
            self.assertEqual(invalid.returncode,2)
            self.assertFalse(list(Path(root).iterdir()))
            success=subprocess.run([*command,'run','--scenario','S01','--output',root],cwd=root,capture_output=True,text=True)
            self.assertEqual(success.returncode,0,success.stderr)

    def test_cli_calculation_assertion_failure_exit_code(self):
        from shutil import copytree
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for name in ('experiments','criteria','alternative','datasets'):
                copytree(ROOT/name,root/name)
            source=root/'experiments/S01_uniform_preferences.json'
            config=json.loads(source.read_text())
            config['expected_assertions'].append({'id':'deliberate_failure','check':'normalized_weights','expected':False})
            source.write_text(json.dumps(config))
            command=[sys.executable,str(ROOT/'run_experiment.py'),'run','--experiments',str(root/'experiments'),'--scenario','S06','--output',str(root/'outputs')]
            result=subprocess.run(command,cwd=root,capture_output=True,text=True)
            self.assertEqual(result.returncode,1,result.stderr)
            manifest=json.loads(next((root/'outputs').glob('*/manifest.json')).read_text())
            self.assertEqual([e['status'] for e in manifest['runs']],['failed','blocked'])
            self.assertFalse(manifest['successful'])

    def test_manifest_traversal_and_artifact_version(self):
        for path in ('../elsewhere.json','/tmp/elsewhere.json','summary.csv'):
            with self.assertRaises(ValueError):
                safe_artifact(self.path,path)
        d=self.results['S01'].model_dump();d['schema_version']='99'
        with self.assertRaises(ValueError):
            RunResult.model_validate(d)


if __name__=='__main__':
    unittest.main()
