import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from src.load_experiments import load_experiments
from src.runner import ROOT, build_plan, execute_batch


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


if __name__=='__main__':
    unittest.main()
