"""Deterministic test double; never used by production experiment execution."""
import json
from types import SimpleNamespace

from src.comprehension import AIConfig
from src.runner import execute_batch as real_execute_batch

CONFIG = AIConfig(model="test-model", api_key="test-credential", max_retries=0)


def output_for(messages):
    supplied = json.loads(messages[1]["content"])["input"]
    stage = supplied["stage"]
    context = supplied["context"]
    evidence_id = "authoritative/scenario/policy_question"
    serial = 0
    def finding(fid=evidence_id, subject=None):
        nonlocal serial
        serial += 1
        return {"claim_id": f"{stage}:{serial}", "claim_type": "supported_interpretation",
                "subject_ids": [subject] if subject else [], "check": {"operation":"copy", "fact_ids":[fid], "value":context["facts"][fid]["value"]}, "dependency_ids": [], "assumptions": [], "uncertainty_category": None,
                "statement": "The modeled decision is conditional on the recorded evidence.", "assessment": "supported",
                "evidence": [{"source": context["facts"][fid]["source"], "fact_id": fid}],
                "caveats": ["Synthetic data; missing metadata and sampled diagnostics limit interpretation."]}
    common = {"schema_version": 1, "warning_codes": supplied["required_warning_codes"],
              "limitations": ["Synthetic inputs, missing metadata and sampled diagnostics limit interpretation."]}
    if stage == "session_understanding":
        return {**common, "decision_frame": finding(),
                "criteria": [{"criterion_id": cid, "meaning_and_measurement": finding(f"authoritative/criteria/{cid}/name",cid)} for cid in context["entities"]["criteria"]],
                "alternatives": [{"alternative_id": aid, "policy_option": finding(f"authoritative/alternatives/{aid}/name",aid)} for aid in context["entities"]["alternatives"]],
                "stakeholder_structure": [finding(f"authoritative/groups/{gid}/name",gid) for gid in context["entities"]["groups"]], "contextual_relationships": [],
                "assumptions_and_gaps": [finding("completeness/objectives")]}
    if stage == "stakeholder_profiles":
        return {**common, "profiles": [{"stakeholder_group_id": gid, "observed_priorities": [finding(f"derived/groups/{gid}/effective_voting_power",gid)],
                "ranking_pattern": finding(f"derived/diagnostics/groups/{gid}/rankings" if f"derived/diagnostics/groups/{gid}/rankings" in context["facts"] else "completeness/configured_groups_without_observations",gid), "representation": finding(f"authoritative/groups/{gid}/participant_count",gid), "tensions_and_limits": [finding()]} for gid in context["entities"]["groups"]]}
    return {**common, "aggregate_priorities": [finding(f"derived/applied_weights/{context["entities"]["criteria"][0]}")], "aggregate_rankings": finding("derived/rankings"),
            **{key: [finding("completeness/objectives" if key == "unresolved_issues" else "derived/diagnostics/limitations")] for key in ("cross_group_comparisons", "agreement_and_disagreement", "stakeholder_influence", "robustness", "unresolved_issues")}}


class FakeClient:
    def __init__(self, transform=None):
        self.chat = SimpleNamespace(completions=self)
        self.requests = []
        self.transform = transform

    def create(self, **kwargs):
        self.requests.append(kwargs)
        output = output_for(kwargs["messages"])
        raw = self.transform(output) if self.transform else json.dumps(output)
        return SimpleNamespace(id=f"response-{len(self.requests)}", model="test-returned-model", usage=None,
                               choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content=raw, refusal=None))])


def execute_test_batch(plan, output):
    return real_execute_batch(plan, output, ai_config=CONFIG, ai_client=FakeClient())


class CloudServer:
    """Local OpenAI-shaped HTTP server for actual SDK/CLI subprocess tests."""
    def __enter__(self):
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        import threading
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                payload = {"id": "test-http-response", "object": "chat.completion", "created": 1,
                           "model": "test-returned-model", "choices": [{"index": 0, "finish_reason": "stop",
                           "message": {"role": "assistant", "content": json.dumps(output_for(request["messages"])) if request.get("max_tokens") != 8 else "OK"}}]}
                data = json.dumps(payload).encode()
                self.send_response(200); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)
            def log_message(self, *args):
                pass
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True); self.thread.start()
        import os
        self.env = {**os.environ, "OLLAMA_API_KEY": "test-credential", "OLLAMA_MODEL": "test-model",
                    "OLLAMA_BASE_URL": f"http://127.0.0.1:{self.server.server_port}/v1"}
        return self

    def __exit__(self, *args):
        self.server.shutdown(); self.server.server_close(); self.thread.join()
