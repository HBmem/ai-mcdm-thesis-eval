import json
from pathlib import Path
import sys
import tempfile
import unittest

from dcl_fakes import execute_test_batch, CloudServer
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from src.load_experiments import load_experiments
from src.runner import ROOT, build_plan, execute_batch


execute_batch = execute_test_batch

class DashboardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary=tempfile.TemporaryDirectory()
        cls.path,cls.results=execute_batch(build_plan(load_experiments()),cls.temporary.name)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def app(self, root):
        return AppTest.from_file(ROOT/'app.py',default_timeout=15)

    def run_app(self, app, root):
        # AppTest.from_file has no args parameter in the installed version.
        with patch.object(sys,'argv',['app.py','--results',str(root)]):
            app.run()
        self.assertFalse(app.exception, [str(x.value) for x in app.exception])
        return app

    def test_empty_root(self):
        with tempfile.TemporaryDirectory() as root:
            app=self.run_app(self.app(root),root)
            self.assertTrue(any('No readable result batches' in x.value for x in app.info))

    def test_scenario_selection_displayed_values_and_sensitivity(self):
        root=self.temporary.name
        before={p: p.stat().st_mtime_ns for p in Path(root).rglob('*') if p.is_file()}
        app=self.run_app(self.app(root),root)
        self.assertEqual(app.metric[0].value,'passed')
        self.assertEqual(app.metric[1].value,self.results[0].topsis['rankings'][0]['alternative_id'])
        self.assertEqual(app.metric[2].value,'0.000000')
        # Sidebar selectors: Batch, Scenario, Variant.
        app.sidebar.selectbox[1].select('S04')
        self.run_app(app,root)
        self.assertTrue(any('Target and achieved profiles' in x.value for x in app.markdown))
        self.assertTrue(any('target' in x.value.columns for x in app.dataframe))
        app.sidebar.selectbox[1].select('S08')
        self.run_app(app,root)
        self.assertTrue(any('Winner changes are bounded' in x.value for x in app.caption))
        self.assertTrue(any(x.label=='Sensitivity criterion' for x in app.selectbox))
        app.sidebar.selectbox[2].select_index(4)
        self.run_app(app,root)
        self.assertTrue(any("S08_C1_W4" in x.value for x in app.caption))
        self.assertEqual(before,{p: p.stat().st_mtime_ns for p in Path(root).rglob('*') if p.is_file()})

    def test_failed_run_messages(self):
        with tempfile.TemporaryDirectory() as root:
            batch=Path(root)/'batch';batch.mkdir()
            value=self.results[0].model_dump()
            value.update(status='failed',preferences={},topsis={},inputs={},applied_weights={})
            value['execution']['messages']=['Injected calculation error']
            (batch/'run.json').write_text(json.dumps(value))
            (batch/'manifest.json').write_text(json.dumps({'schema_version':'1.0','status':'completed','runs':[{'experiment_id':'S01','case_key':'S01','file':'run.json','status':'failed'}]}))
            app=self.run_app(self.app(root),root)
            self.assertEqual(app.metric[0].value,'failed')
            self.assertTrue(any('Injected calculation error' in x.value for x in app.error))

    def test_invalid_manifest_version_and_path_escape(self):
        for manifest in (
            {'schema_version':'99','runs':[]},
            {'schema_version':'1.0','status':'completed','runs':[{'experiment_id':'S01','case_key':'S01','file':'../outside.json','status':'passed'}]},
        ):
            with self.subTest(manifest=manifest), tempfile.TemporaryDirectory() as root:
                batch=Path(root)/'batch';batch.mkdir()
                (batch/'manifest.json').write_text(json.dumps(manifest))
                app=self.run_app(self.app(root),root)
                self.assertTrue(app.warning)
                self.assertTrue(app.info)

    def test_unsupported_artifact_and_missing_artifact(self):
        for payload in (None,json.dumps({'schema_version':'99'})):
            with self.subTest(payload=payload), tempfile.TemporaryDirectory() as root:
                batch=Path(root)/'batch';batch.mkdir()
                (batch/'manifest.json').write_text(json.dumps({'schema_version':'1.0','status':'completed','runs':[{'experiment_id':'S01','case_key':'S01','file':'run.json','status':'passed'}]}))
                if payload:
                    (batch/'run.json').write_text(payload)
                app=self.run_app(self.app(root),root)
                self.assertTrue(app.error)
                self.assertTrue(any('Cannot read result' in x.value for x in app.error))

    def test_ai_call_selection_matches_exact_messages_and_run(self):
        root=self.temporary.name
        app=self.run_app(self.app(root),root)
        result=self.results[0]
        selector=next(w for w in app.selectbox if w.label=='AI call')
        self.assertEqual(len(selector.options),3)
        for index in (0,1,2):
            next(w for w in app.selectbox if w.label=='AI call').select_index(index)
            self.run_app(app,root)
            call=result.comprehension.calls[index]
            displayed=[element.value for element in app.code]
            self.assertIn(call.messages[0]['content'],displayed)
            self.assertIn(call.messages[1]['content'],displayed)
            self.assertIn(call.raw_output,displayed)
            self.assertTrue(any(call.call_id in item.value and result.run_id in item.value for item in app.caption))
            values=[json.loads(item.value) for item in app.json]
            self.assertIn(call.input,values)
            self.assertIn(call.parsed_output,values)
        app.sidebar.selectbox[1].select('S04')
        self.run_app(app,root)
        next_result=next(r for r in self.results if r.case_key=='S04')
        self.assertTrue(any(next_result.run_id in item.value for item in app.caption))
        self.assertFalse(any(result.comprehension.calls[2].call_id in item.value for item in app.caption))

    def test_failed_call_and_legacy_empty_state(self):
        from dcl_fakes import CONFIG, FakeClient
        from src.runner import execute_batch as real_execute_batch
        with tempfile.TemporaryDirectory() as root:
            _,results=real_execute_batch(build_plan(load_experiments(),'S01'),root,
                ai_config=CONFIG,ai_client=FakeClient(lambda value:'malformed output'))
            app=self.run_app(self.app(root),root)
            self.assertTrue(any('JSONDecodeError' in item.value for item in app.error))
            self.assertTrue(any(item.value=='malformed output' for item in app.code))
            batch=next(Path(root).glob('*/manifest.json')).parent
            file=batch/f'{results[0].run_id}.json'
            legacy=results[0].model_dump(mode='json');legacy.update(schema_version='1.0',comprehension=None)
            file.write_text(json.dumps(legacy))
            app=self.run_app(self.app(root),root)
            self.assertTrue(any('No AI calls recorded' in item.value for item in app.info))

    def test_sensitive_call_output_is_redacted_before_display(self):
        with tempfile.TemporaryDirectory() as root:
            batch=Path(root)/'batch';batch.mkdir()
            value=self.results[0].model_dump(mode='json')
            value['comprehension']['calls'][0]['raw_output']='password=hidden-value person@example.org'
            (batch/'run.json').write_text(json.dumps(value))
            (batch/'manifest.json').write_text(json.dumps({'schema_version':'1.0','status':'completed',
                'runs':[{'experiment_id':'S01','case_key':'S01','file':'run.json','status':'passed'}]}))
            app=self.run_app(self.app(root),root)
            rendered=' '.join(item.value for item in app.code)
            self.assertNotIn('hidden-value',rendered)
            self.assertNotIn('person@example.org',rendered)
            self.assertIn('[REDACTED]',rendered)

    def test_retry_attempt_errors_remain_with_the_selected_call(self):
        with tempfile.TemporaryDirectory() as root:
            batch=Path(root)/'batch';batch.mkdir()
            value=self.results[0].model_dump(mode='json')
            calls=value['comprehension']['calls']
            retry={**calls[0], 'call_id':'retry-attempt', 'status':'retrying', 'attempt':1,
                   'raw_output':None,'parsed_output':None,'verification':[],'receipt':None,
                   'error':{'type':'APIConnectionError','message':'Transient transport failure'}}
            calls[0]['attempt']=2
            value['comprehension']['calls']=[retry,*calls]
            (batch/'run.json').write_text(json.dumps(value))
            (batch/'manifest.json').write_text(json.dumps({'schema_version':'1.0','status':'completed',
                'runs':[{'experiment_id':'S01','case_key':'S01','file':'run.json','status':'passed'}]}))
            app=self.run_app(self.app(root),root)
            selector=next(w for w in app.selectbox if w.label=='AI call')
            self.assertEqual(len(selector.options),4)
            self.assertTrue(any('Transient transport failure' in item.value for item in app.error))
            self.assertTrue(any('retry-attempt' in item.value for item in app.caption))
            selector.select_index(1)
            self.run_app(app,root)
            self.assertFalse(any('Transient transport failure' in item.value for item in app.error))
            self.assertTrue(any(calls[0]['call_id'] in item.value for item in app.caption))

    def test_batch_preflight_is_displayed_separately_from_run_calls(self):
        from src.preflight import PreflightRecord
        with tempfile.TemporaryDirectory() as root:
            batch=Path(root)/'batch';batch.mkdir()
            value=self.results[0].model_dump(mode='json')
            (batch/'run.json').write_text(json.dumps(value))
            (batch/'manifest.json').write_text(json.dumps({'schema_version':'1.0','status':'completed',
                'runs':[{'experiment_id':'S01','case_key':'S01','file':'run.json','status':'passed'}]}))
            record=PreflightRecord(call_id='batch-probe',batch_id='batch',started_at='start',completed_at='end',
                duration_seconds=.1,messages=[{'role':'user','content':'Connectivity check.'}],
                input={'model':'test-model','max_tokens':8},configuration={'max_retries':0},
                raw_output='OK',response={'choices':[{'finish_reason':'stop'}]},receipt={'response_id':'probe-id'})
            (batch/'preflight.json').write_text(record.model_dump_json())
            before={p:p.stat().st_mtime_ns for p in batch.iterdir()}
            app=self.run_app(self.app(root),root)
            self.assertTrue(any(item.label=='Batch startup check' for item in app.expander))
            self.assertTrue(any('batch-probe' in item.value and 'Batch: batch' in item.value for item in app.caption))
            self.assertIn('Connectivity check.',[item.value for item in app.code])
            self.assertIn('OK',[item.value for item in app.code])
            self.assertEqual(len(next(w for w in app.selectbox if w.label=='AI call').options),3)
            self.assertEqual(before,{p:p.stat().st_mtime_ns for p in batch.iterdir()})
            bad=record.model_copy(update={'batch_id':'other-batch'})
            (batch/'preflight.json').write_text(bad.model_dump_json())
            app=self.run_app(self.app(root),root)
            self.assertTrue(any('Cannot read batch startup check' in item.value for item in app.warning))


if __name__=='__main__':
    unittest.main()
