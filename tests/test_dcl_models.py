"""Contract tests for selected-object DCL inputs, responses and audit artifacts.

These tests use genuine MCDM model types and never contact a provider or require
an implemented comprehension pipeline. Legacy runtime tests are separate.
"""
from copy import deepcopy
from datetime import datetime, timezone
import inspect
import json
import unittest

from pydantic import ValidationError

from src.ai.comprehension import models as dcl
from src.ai.models.analysis import Record
from src.mcdm.models.computation import (
    AlternativeRanking, CriterionWeight, EntityCalculation, EntityMetadata,
    RankingResult, WeightResult,
)
from src.mcdm.models.domain import Alternative, Criterion, CriterionValue, StakeholderGroup
from src.mcdm.models.methods import (
    AggregationMethodConfig, ConsistencyMethodConfig, RankingMethodConfig, WeightingMethodConfig,
)
from src.mcdm.enums import (
    CriterionDirection, EmptyGroupPolicy, GroupPowerMethod, MeanMethod,
    OnExceedancePolicy, RankingMethod, WeightingMethod,
)


class DCLModelTests(unittest.TestCase):
    def setUp(self):
        self.group = StakeholderGroup(id="X", name="Group X", description="Modeled constituency", voting_power=.2)
        self.criteria = tuple(Criterion(id=cid, name=name, description=name, direction=direction, unit=unit)
            for cid,name,direction,unit in (
                ("C1","Cost",CriterionDirection.MINIMIZE,"USD"),
                ("C2","Time",CriterionDirection.MINIMIZE,"months")))
        self.alternatives = (Alternative(id="A", name="Option A", description="Authored option",
            criterion_values=[CriterionValue(criterion_id="C1",value=10.), CriterionValue(criterion_id="C2",value=2.)]),)
        self.numerical = EntityCalculation(entity_id="X", scope="group",pairwise_matrix=[[1.,1.],[1.,1.]],
            weighting=WeightResult(criterion_weights=[
                CriterionWeight(criterion_id="C1",criterion_name="Cost",weight=.4),
                CriterionWeight(criterion_id="C2",criterion_name="Time",weight=.6)],
                consistency_ratio=0.,consistency_threshold=.1),
            ranking=RankingResult(rankings=[AlternativeRanking(alternative_id="A",alternative_name="Option A",rank=1,closeness_coefficient=.5)]),
            metadata=EntityMetadata(participant_count=1,voting_power=.2,effective_voting_power=.2,
                ranking_criterion_ids=["C1","C2"],applied_criterion_weights={"C1":.7,"C2":.3},ranking_weight_source="explicit"))
        self.processing = dcl.ProcessingContext(
            weighting_method=WeightingMethodConfig(method=WeightingMethod.AHP),
            ranking_method=RankingMethodConfig(method=RankingMethod.TOPSIS),
            aggregation_method=AggregationMethodConfig(within_group=MeanMethod.GEOMETRIC_MEAN,
                between_groups=MeanMethod.WEIGHTED_GEOMETRIC_MEAN,
                group_power_normalization=GroupPowerMethod.REPRESENTED_GROUPS,empty_group_policy=EmptyGroupPolicy.EXCLUDED),
            consistency_method=ConsistencyMethodConfig(threshold=.1,on_exceedance=OnExceedancePolicy.RETAIN_AND_FLAG))
        self.common = dict(scenario=dcl.ScenarioContext(scenario_id="scenario",title="Decision",description="Synthetic scope",decision_question="Which option?"),
            criteria=self.criteria,alternatives=self.alternatives,processing=self.processing,
            available_evidence_ids=("group:X:criterion_weights","group:X:ranking","group:X:participant_count"),
            required_limitations=("Synthetic observations do not establish population preferences.",))
        self.input = dcl.GroupProfileInput(**self.common,group=self.group,
            calculation=dcl.GroupCalculationView.from_calculation(self.numerical))
        self.draft = dcl.InterpretationDraft(statement="Time receives the higher modeled preference weight.",
            assessment="supported",evidence_ids=("group:X:criterion_weights",),assumptions=(),uncertainties=(),caveats=())
        self.response = dcl.GroupProfileResponse(priorities=self.draft,
            ranking=self.draft.model_copy(update={"evidence_ids":("group:X:ranking",)}),
            representation=self.draft.model_copy(update={"evidence_ids":("group:X:participant_count",)}),diagnostics=(),limitations=())
        self.request = dcl.DCLRequestMetadata(request_id="request",run_id="run",ai_repetition=1,
            stage="stakeholder_group_profile",group_id="X",prompt_version="draft-v1",input_hash="a"*64)
        self.receipt = dcl.ProviderReceipt(requested_model="example",received_at=datetime.now(timezone.utc))

    def artifact(self, **changes):
        fields = dict(artifact_id="artifact",request=self.request,input_snapshot=self.input,response=self.response,
            receipt=self.receipt,findings=(dcl.StoredInterpretation(finding_id="finding",response_path=("priorities",),draft=self.draft,reference_status="validated"),),
            required_limitations=self.input.required_limitations)
        return dcl.GroupProfileArtifact(**{**fields,**changes})

    def test_selected_objects_round_trip_without_roster_matrix_or_fact_catalog(self):
        payload=self.input.model_dump_json()
        restored=dcl.GroupProfileInput.model_validate_json(payload)
        self.assertEqual(restored,self.input)
        self.assertIsInstance(restored.calculation.weighting,WeightResult)
        self.assertIsInstance(restored.processing.aggregation_method,AggregationMethodConfig)
        for absent in ("pairwise_matrix",'"participants"','"facts"','"api_key"'):
            self.assertNotIn(absent,payload)
        self.assertEqual(restored.calculation.metadata.effective_voting_power,.2)

    def test_views_preserve_original_and_applied_weights_and_copy_source(self):
        view=self.input.calculation
        self.assertEqual(view.weighting.criterion_weights[0].weight,.4)
        self.assertEqual(view.metadata.applied_criterion_weights["C1"],.7)
        view.metadata.applied_criterion_weights["C1"] = .9
        view.weighting.criterion_weights[0].weight = .8
        self.assertEqual(self.numerical.metadata.applied_criterion_weights["C1"],.7)
        self.assertEqual(self.numerical.weighting.criterion_weights[0].weight,.4)
        with self.assertRaises(ValidationError):dcl.SessionCalculationView.from_calculation(self.numerical)
        participant=deepcopy(self.numerical);participant.scope="participant"
        with self.assertRaises(ValidationError):dcl.GroupCalculationView.from_calculation(participant)

    def test_optional_effective_power_preserves_unknown_not_zero(self):
        self.assertIsNone(EntityMetadata(participant_count=1,voting_power=.2).effective_voting_power)
        missing=dcl.GroupProfileInput(**{**self.common,"group":self.group,"calculation":None})
        self.assertIsNone(missing.calculation)

    def test_group_scope_and_duplicate_aliases_are_rejected(self):
        wrong=self.input.calculation.model_copy(update={"entity_id":"Y"})
        with self.assertRaises(ValidationError):dcl.GroupProfileInput(**self.common,group=self.group,calculation=wrong)
        with self.assertRaises(ValidationError):dcl.GroupContext(group=self.group,calculation=wrong)
        with self.assertRaises(ValidationError):dcl.GroupProfileInput(**{**self.common,"available_evidence_ids":("same","same")},group=self.group,calculation=None)
        with self.assertRaises(ValidationError):dcl.InterpretationDraft(**{**self.draft.model_dump(),"evidence_ids":("same","same")})

    def test_responses_have_no_application_ids_checks_or_provider_state(self):
        payload=json.loads(self.response.model_dump_json())
        self.assertEqual(set(payload["priorities"]),{"statement","assessment","evidence_ids","assumptions","uncertainties","caveats"})
        self.assertEqual(dcl.GroupProfileResponse.model_validate_json(self.response.model_dump_json()),self.response)
        with self.assertRaises(ValidationError):dcl.InterpretationDraft(**{**self.draft.model_dump(),"check":{"operation":"copy"}})
        stored=dcl.StoredInterpretation(finding_id="finding",response_path=("priorities",),draft=self.draft)
        self.assertEqual(stored.content_review.status,"not_assessed")
        self.assertEqual(stored.reference_status,"not_checked")
        self.assertEqual(stored.draft.assessment,"supported")

    def test_first_and_final_stage_inputs_round_trip(self):
        first=dcl.SessionUnderstandingInput(**self.common,groups=(self.group,))
        session=deepcopy(self.numerical);session.scope="session";session.entity_id="session"
        final=dcl.SessionProfileInput(**self.common,groups=(dcl.GroupContext(group=self.group,calculation=self.input.calculation),),
            calculation=dcl.SessionCalculationView.from_calculation(session))
        self.assertEqual(dcl.SessionUnderstandingInput.model_validate_json(first.model_dump_json()),first)
        self.assertEqual(dcl.SessionProfileInput.model_validate_json(final.model_dump_json()),final)
        prior=dcl.PriorInterpretation(finding_id="finding",source_request_id="request",source_run_id="run",entity_ids=("X",),draft=self.draft,content_status="not_assessed")
        with self.assertRaises(ValidationError):dcl.SessionUnderstandingInput(**{**self.common,"prior_interpretations":(prior,)},groups=(self.group,))
        with self.assertRaises(ValidationError):dcl.SessionProfileInput(**self.common,groups=(final.groups[0],final.groups[0]),calculation=final.calculation)

    def test_artifact_round_trip_keeps_limitations_and_reference_checks_separate(self):
        artifact=self.artifact()
        self.assertEqual(dcl.GroupProfileArtifact.model_validate_json(artifact.model_dump_json()),artifact)
        self.assertEqual(artifact.content_review.status,"not_assessed")
        self.assertEqual(artifact.findings[0].reference_status,"validated")
        with self.assertRaises(ValidationError):self.artifact(required_limitations=())
        changed=dcl.StoredInterpretation(finding_id="finding",response_path=("priorities",),draft=self.draft.model_copy(update={"statement":"Changed prose"}))
        with self.assertRaises(ValidationError):self.artifact(findings=(changed,))
        wrong=dcl.StoredInterpretation(finding_id="finding",response_path=("absent",),draft=self.draft)
        with self.assertRaises(ValidationError):self.artifact(findings=(wrong,))
        restricted=self.input.model_copy(update={"available_evidence_ids":("group:X:ranking",)})
        with self.assertRaises(ValidationError):self.artifact(input_snapshot=restricted)

    def test_request_and_run_state_reject_wrong_stage_group_or_repetition(self):
        with self.assertRaises(ValidationError):dcl.DCLRequestMetadata(**{**self.request.model_dump(),"group_id":None})
        with self.assertRaises(ValidationError):dcl.DCLRequestMetadata(**{**self.request.model_dump(),"stage":"session_profile"})
        other=self.request.model_copy(update={"group_id":"Y"})
        with self.assertRaises(ValidationError):self.artifact(request=other)
        artifact=self.artifact()
        state=dcl.DCLRunState(run_id="run",ai_repetition=1,group_profiles=(artifact,))
        self.assertEqual(dcl.DCLRunState.model_validate_json(state.model_dump_json()),state)
        with self.assertRaises(ValidationError):dcl.DCLRunState(run_id="other",ai_repetition=1,group_profiles=(artifact,))
        with self.assertRaises(ValidationError):dcl.DCLRunState(run_id="run",ai_repetition=2,group_profiles=(artifact,))
        with self.assertRaises(ValidationError):dcl.DCLRunState(run_id="run",ai_repetition=1,group_profiles=(artifact,artifact))

    def test_failed_calls_preserve_raw_output_without_fabricating_a_response(self):
        now=datetime.now(timezone.utc)
        fields=dict(call_id="call",request=self.request,attempt=1,input_snapshot=self.input,status="failed",
            started_at=now,completed_at=now,raw_response="{broken",error=dcl.DCLCallError(phase="parsing",code="invalid_json",message="Invalid JSON"))
        failed=dcl.DCLCallRecord(**fields)
        self.assertIsNone(failed.parsed_response)
        self.assertEqual(dcl.DCLCallRecord.model_validate_json(failed.model_dump_json()),failed)
        with self.assertRaises(ValidationError):dcl.DCLCallRecord(**{**fields,"error":None})
        with self.assertRaises(ValidationError):dcl.DCLCallRecord(**{**fields,"status":"completed","error":None})
        with self.assertRaises(ValidationError):dcl.DCLCallRecord(**{**fields,"request":self.request.model_copy(update={"group_id":"Y"})})
        wrong_response=dcl.SessionProfileResponse(priorities=self.draft,ranking=self.draft,
            group_comparisons=(),agreement=(),influence=(),robustness=(),limitations=())
        with self.assertRaises(ValidationError):dcl.DCLCallRecord(**{**fields,"status":"completed","error":None,"parsed_response":wrong_response})

    def test_diagnostics_mark_unavailable_analyses_explicitly(self):
        diagnostic=dcl.DiagnosticContext(diagnostic_id="analysis",method="existing-method",entity_ids=("X",),
            status="not_evaluable",result=None,limitations=("No observations.",))
        self.assertEqual(dcl.DiagnosticContext.model_validate_json(diagnostic.model_dump_json()),diagnostic)
        with self.assertRaises(ValidationError):dcl.DiagnosticContext(**{**diagnostic.model_dump(),"result":{"score":1.}})
        with self.assertRaises(ValidationError):dcl.DiagnosticContext(**{**diagnostic.model_dump(),"status":"completed"})

    def test_all_concrete_contracts_have_documentation_and_json_schemas(self):
        classes=[value for value in vars(dcl).values() if inspect.isclass(value) and value.__module__==dcl.__name__ and issubclass(value,Record) and value.__pydantic_generic_metadata__.get("origin") is None]
        self.assertGreater(len(classes),20)
        for model in classes:
            with self.subTest(model=model.__name__):
                self.assertTrue(model.__doc__ and model.__doc__.strip())
                schema=model.model_json_schema()
                self.assertEqual(schema['type'],'object')
                json.dumps(schema,allow_nan=False)

    def test_optional_metadata_addition_keeps_existing_positional_arguments(self):
        from src.mcdm.models.computation import WeightingSource
        source=WeightingSource(kind="calculated")
        metadata=EntityMetadata(None,1,.2,source)
        self.assertIs(metadata.weighting_source,source)
        self.assertIsNone(metadata.effective_voting_power)

    def test_other_stage_artifacts_keep_their_specific_response_types(self):
        first=dcl.SessionUnderstandingInput(**self.common,groups=(self.group,))
        first_response=dcl.SessionUnderstandingResponse(decision_frame=self.draft,
            objectives_and_scope=(),criteria_and_alternatives=(),stakeholder_structure=(),relationships=(),assumptions_and_gaps=())
        session=deepcopy(self.numerical);session.entity_id="session";session.scope="session"
        final=dcl.SessionProfileInput(**self.common,groups=(dcl.GroupContext(group=self.group,calculation=self.input.calculation),),
            calculation=dcl.SessionCalculationView.from_calculation(session))
        final_response=dcl.SessionProfileResponse(priorities=self.draft,ranking=self.draft,
            group_comparisons=(),agreement=(),influence=(),robustness=(),limitations=())
        for model,supplied,response,stage in (
            (dcl.SessionUnderstandingArtifact,first,first_response,"session_understanding"),
            (dcl.SessionProfileArtifact,final,final_response,"session_profile")):
            request=dcl.DCLRequestMetadata(request_id=stage,run_id="run",ai_repetition=1,
                stage=stage,prompt_version="draft-v1",input_hash="b"*64)
            artifact=model(artifact_id=stage,request=request,input_snapshot=supplied,response=response,
                receipt=self.receipt,findings=(),required_limitations=supplied.required_limitations)
            self.assertEqual(model.model_validate_json(artifact.model_dump_json()),artifact)
            with self.assertRaises(ValidationError):
                model(artifact_id=stage,request=request,input_snapshot=supplied,response=self.response,
                    receipt=self.receipt,findings=(),required_limitations=supplied.required_limitations)
