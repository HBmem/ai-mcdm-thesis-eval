from contextlib import ExitStack
import io
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import httpx
from openai import APIConnectionError, APITimeoutError, APIStatusError

import run_experiment
from dcl_fakes import CONFIG
from src.comprehension import AIConfig, DEFAULT_ENV_FILE
from src.preflight import EXIT_PAYMENT_REQUIRED, PREFLIGHT_MESSAGES, PreflightError, check_model_access
from src.results import BatchStore


def probe_client(finish='stop', content='OK', refusal=None):
    client=MagicMock()
    client.__enter__.return_value=client
    client.chat.completions.create.return_value=SimpleNamespace(id='probe-response',model='model-alias',usage=None,
        choices=[SimpleNamespace(finish_reason=finish,message=SimpleNamespace(content=content,refusal=refusal))])
    return client


class EnvironmentTests(unittest.TestCase):
    def test_explicit_file_load_and_environment_precedence(self):
        with tempfile.TemporaryDirectory() as root, patch.dict(os.environ,{},clear=True):
            file=Path(root)/'.env';file.write_text('OLLAMA_API_KEY=file-key\nOLLAMA_MODEL=file-model\n')
            config=AIConfig.from_env(env_file=file)
            self.assertEqual((config.api_key,config.model),('file-key','file-model'))
            os.environ['OLLAMA_MODEL']='environment-model'
            self.assertEqual(AIConfig.from_env(env_file=file).model,'environment-model')
            os.environ['OLLAMA_API_KEY']=''
            with self.assertRaisesRegex(ValueError,'OLLAMA_API_KEY'):
                AIConfig.from_env(env_file=file)

    def test_default_path_is_repository_root_from_foreign_directory(self):
        previous=Path.cwd()
        with tempfile.TemporaryDirectory() as root, patch.dict(os.environ,{'OLLAMA_API_KEY':'key','OLLAMA_MODEL':'model'},clear=True), patch('src.comprehension.load_dotenv') as load:
            try:
                os.chdir(root)
                AIConfig.from_env()
            finally:
                os.chdir(previous)
            load.assert_called_once_with(dotenv_path=DEFAULT_ENV_FILE,override=False)
            self.assertEqual(DEFAULT_ENV_FILE,Path(run_experiment.__file__).resolve().parent/'.env')

    def test_missing_file_and_explicit_disablement(self):
        with tempfile.TemporaryDirectory() as root, patch.dict(os.environ,{},clear=True):
            with self.assertRaisesRegex(ValueError,'Missing required'):
                AIConfig.from_env(env_file=Path(root)/'missing')
            with patch('src.comprehension.load_dotenv') as load:
                with self.assertRaises(ValueError): AIConfig.from_env(env_file=None)
                load.assert_not_called()
        with patch.dict(os.environ,{'OLLAMA_API_KEY':'key','OLLAMA_MODEL':'model'},clear=True):
            self.assertEqual(AIConfig.from_env(env_file=None).model,'model')

    def test_default_sdk_client_disables_retries(self):
        with CONFIG.client() as client:
            self.assertEqual(client.max_retries,0)


class PreflightTests(unittest.TestCase):
    def test_success_records_exact_bounded_request(self):
        client=probe_client()
        record=check_model_access(CONFIG,client)
        request=client.chat.completions.create.call_args.kwargs
        self.assertEqual(request,{'model':CONFIG.model,'messages':PREFLIGHT_MESSAGES,
                                  'stream':False,'max_tokens':8,'timeout':30})
        self.assertEqual(record.input,request)
        self.assertEqual(record.messages,request['messages'])
        self.assertEqual(record.receipt['returned_model'],'model-alias')
        self.assertIsNone(record.receipt['input_tokens'])
        self.assertNotIn(CONFIG.api_key,record.model_dump_json())
        self.assertEqual(record.configuration['max_retries'],0)

    def test_length_finish_with_no_visible_content_is_accepted(self):
        for content in (None,'','Partial'):
            with self.subTest(content=content):
                self.assertEqual(check_model_access(CONFIG,probe_client('length',content)).status,'passed')

    def test_malformed_empty_refused_or_unsupported_response(self):
        clients=[probe_client(content=''),probe_client(refusal='No'),probe_client(finish='tool_calls')]
        no_choices=probe_client();no_choices.chat.completions.create.return_value.choices=[];clients.append(no_choices)
        bad_content=probe_client(content={'unexpected':'shape'});clients.append(bad_content)
        for client in clients:
            with self.subTest(client=client):
                with self.assertRaises(PreflightError) as caught: check_model_access(CONFIG,client)
                self.assertEqual(caught.exception.exit_code,1)
                client.chat.completions.create.assert_called_once()

    def test_status_errors_are_classified_without_secret_leakage(self):
        for status in (400,401,402,403,404,405,422,408,429,500,503):
            with self.subTest(status=status):
                client=probe_client()
                client.chat.completions.create.side_effect=APIStatusError(CONFIG.api_key,
                    response=httpx.Response(status,request=httpx.Request('POST','https://example.test')),
                    body={'api_key':CONFIG.api_key})
                with self.assertRaises(PreflightError) as caught: check_model_access(CONFIG,client)
                expected_code = EXIT_PAYMENT_REQUIRED if status == 402 else 2 if status in (400,401,403,404,405,422) else 1
                self.assertEqual(caught.exception.exit_code,expected_code)
                if status == 402:
                    self.assertIn("billing", str(caught.exception))
                self.assertNotIn(CONFIG.api_key,str(caught.exception))
                client.chat.completions.create.assert_called_once()

    def test_connection_timeout_and_response_redaction(self):
        request=httpx.Request('POST','https://example.test')
        for error in (APIConnectionError(request=request),APITimeoutError(request=request)):
            client=probe_client();client.chat.completions.create.side_effect=error
            with self.assertRaises(PreflightError) as caught: check_model_access(CONFIG,client)
            self.assertEqual(caught.exception.exit_code,1)
            client.chat.completions.create.assert_called_once()
        record=check_model_access(CONFIG,probe_client(content=f'Bearer {CONFIG.api_key}'))
        self.assertTrue(record.redacted)
        self.assertNotIn(CONFIG.api_key,record.model_dump_json())

    def test_batch_audit_is_saved_once_with_batch_association(self):
        record=check_model_access(CONFIG,probe_client())
        with tempfile.TemporaryDirectory() as root:
            store=BatchStore(root)
            store.save_preflight(record)
            from src.preflight import PreflightRecord
            saved=PreflightRecord.model_validate_json((store.path/'preflight.json').read_bytes())
            self.assertEqual(saved.batch_id,store.batch_id)
            self.assertEqual(saved.messages,record.messages)
            with self.assertRaisesRegex(ValueError,'immutable'): store.save_preflight(record)

    def test_cli_failure_never_executes_or_creates_artifacts(self):
        for code in (1,2):
            with tempfile.TemporaryDirectory() as root, ExitStack() as stack:
                stack.enter_context(patch('run_experiment.load_experiments',return_value=[]))
                stack.enter_context(patch('run_experiment.build_plan',return_value=[]))
                stack.enter_context(patch('run_experiment.AIConfig.from_env',return_value=CONFIG))
                stack.enter_context(patch.object(AIConfig,'client',return_value=probe_client()))
                execute=stack.enter_context(patch('run_experiment.execute_batch'))
                stack.enter_context(patch('run_experiment.check_model_access',side_effect=PreflightError('Access verification unsuccessful.',code)))
                stderr=stack.enter_context(patch('sys.stderr',new_callable=io.StringIO))
                stack.enter_context(patch('sys.stdout',new_callable=io.StringIO))
                output=Path(root)/'results'
                self.assertEqual(run_experiment.main(['run','--output',str(output)]),code)
                execute.assert_not_called();self.assertFalse(output.exists())
                self.assertNotIn(CONFIG.api_key,stderr.getvalue())

    def test_cli_success_reuses_and_closes_client(self):
        client=probe_client()
        with ExitStack() as stack:
            stack.enter_context(patch('run_experiment.load_experiments',return_value=[]))
            stack.enter_context(patch('run_experiment.build_plan',return_value=[]))
            stack.enter_context(patch('run_experiment.AIConfig.from_env',return_value=CONFIG))
            stack.enter_context(patch.object(AIConfig,'client',return_value=client))
            execute=stack.enter_context(patch('run_experiment.execute_batch',return_value=(Path('/tmp/mock-batch'),[])))
            stack.enter_context(patch('sys.stdout',new_callable=io.StringIO))
            self.assertEqual(run_experiment.main(['run']),0)
            self.assertIs(execute.call_args.kwargs['ai_client'],client)
            self.assertEqual(execute.call_args.kwargs['preflight'].stage,'startup_check')
            client.__exit__.assert_called_once()

    def test_validate_remains_offline(self):
        with patch('run_experiment.load_experiments',return_value=[]), patch('run_experiment.build_plan',return_value=[]), patch('run_experiment.AIConfig.from_env') as config, patch('run_experiment.check_model_access') as check, patch('sys.stdout',new_callable=io.StringIO):
            self.assertEqual(run_experiment.main(['validate']),0)
            config.assert_not_called();check.assert_not_called()
