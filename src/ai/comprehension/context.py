from pydantic import BaseModel
from dataclasses import fields, is_dataclass

from src.ai.comprehension.models import (
    PROCESSING_EVIDENCE_ID,
    DCLStageInput,
    EvidenceReference,
    GroupCalculationView,
    SessionCalculationView,
    SessionUnderstandingInput,
    SessionProfileInput,
    GroupProfileInput,
)


class EvidenceResolutionError(ValueError):
    pass


def _find_by_id(
    objects: tuple[object, ...],
    object_id: str,
    label: str,
) -> object:
    matches = [
        item
        for item in objects
        if getattr(item, "id", None) == object_id
    ]

    if len(matches) != 1:
        raise EvidenceResolutionError(
            f"Unknown or ambiguous {label}: {object_id}"
        )

    return matches[0]


def _find_diagnostic(
    diagnostics: tuple[object, ...],
    diagnostic_id: str,
) -> object:
    matches = [
        item
        for item in diagnostics
        if item.diagnostic_id == diagnostic_id
    ]

    if len(matches) != 1:
        raise EvidenceResolutionError(
            f"Unknown or ambiguous diagnostic: {diagnostic_id}"
        )

    return matches[0]


def _session_calculations(
    snapshot: SessionProfileInput,
) -> tuple[
    GroupCalculationView
    | SessionCalculationView,
    ...,
]:
    calculations = [
        group.calculation
        for group in snapshot.groups
        if group.calculation is not None
    ]

    if snapshot.calculation is not None:
        calculations.append(
            snapshot.calculation
        )

    return tuple(calculations)


def _resolve_path(
    root: object,
    path: tuple[str | int, ...],
) -> object:
    current = root

    for part in path:
        if isinstance(part, str) and isinstance(current, BaseModel):
            if part not in type(current).model_fields:
                raise EvidenceResolutionError(
                    f"Unknown evidence field: {part}"
                )

            current = getattr(current, part)

        elif (isinstance(part, str) and is_dataclass(current) and not isinstance(current, type)):
            field_names = {
                field.name
                for field in fields(current)
            }

            if part not in field_names:
                raise EvidenceResolutionError(f"Unknown evidence dataclass field: {part}")

            current = getattr(current, part)

        elif isinstance(part, str) and isinstance(current, dict):
            if part not in current:
                raise EvidenceResolutionError(
                    f"Unknown evidence mapping key: {part}"
                )

            current = current[part]

        elif isinstance(part, int) and isinstance(
            current,
            (tuple, list),
        ):
            if not 0 <= part < len(current):
                raise EvidenceResolutionError(
                    f"Evidence index is out of range: {part}"
                )

            current = current[part]

        else:
            raise EvidenceResolutionError(
                f"Cannot resolve evidence path component: {part!r}"
            )

    return current



def resolve_evidence(
    snapshot: DCLStageInput,
    reference: EvidenceReference,
) -> object:
    """Return the referenced value or raise EvidenceResolutionError."""
    if isinstance(snapshot, SessionUnderstandingInput):
        return resolve_session_understanding_evidence(
            snapshot,
            reference,
        )

    if isinstance(snapshot, SessionProfileInput):
        return resolve_session_profile_evidence(
            snapshot,
            reference,
        )
    
    if isinstance(snapshot, GroupProfileInput):
        return resolve_group_profile_evidence(
            snapshot,
            reference,
        )

    raise EvidenceResolutionError(f"Evidence resolution is not implemented for stage {snapshot.stage!r}.")


def resolve_session_understanding_evidence(
    snapshot: SessionUnderstandingInput,
    reference: EvidenceReference,
) -> object:
    if reference.object_type == "scenario":
        if (
            reference.object_id
            != snapshot.scenario.scenario_id
        ):
            raise EvidenceResolutionError(
                "Unknown scenario evidence object."
            )

        current: object = snapshot.scenario

    elif reference.object_type == "objective":
        current = _find_by_id(
            snapshot.scenario.objectives or (),
            reference.object_id,
            "objective",
        )

    elif reference.object_type == "assumption":
        current = _find_by_id(
            snapshot.scenario.assumptions or (),
            reference.object_id,
            "assumption",
        )

    elif reference.object_type == "constraint":
        current = _find_by_id(
            snapshot.scenario.constraints or (),
            reference.object_id,
            "constraint",
        )

    elif reference.object_type == "criterion":
        current = _find_by_id(
            snapshot.criteria,
            reference.object_id,
            "criterion",
        )

    elif reference.object_type == "alternative":
        current = _find_by_id(
            snapshot.alternatives,
            reference.object_id,
            "alternative",
        )

    elif reference.object_type == "group":
        current = _find_by_id(
            snapshot.groups,
            reference.object_id,
            "group",
        )

    elif reference.object_type == "processing":
        if reference.object_id != PROCESSING_EVIDENCE_ID:
            raise EvidenceResolutionError(
                "Unknown processing evidence object."
            )

        current = snapshot.processing

    else:
        raise EvidenceResolutionError(
            "Unsupported session-understanding "
            "evidence object type."
        )

    return _resolve_path(
        current,
        reference.field_path,
    )


def resolve_session_profile_evidence(
    snapshot: SessionProfileInput,
    reference: EvidenceReference,
) -> object:
    if reference.object_type == "scenario":
        if (
            reference.object_id
            != snapshot.scenario.scenario_id
        ):
            raise EvidenceResolutionError(
                "Unknown scenario evidence object."
            )

        current: object = snapshot.scenario

    elif reference.object_type == "objective":
        current = _find_by_id(
            snapshot.scenario.objectives or (),
            reference.object_id,
            "objective",
        )

    elif reference.object_type == "assumption":
        current = _find_by_id(
            snapshot.scenario.assumptions or (),
            reference.object_id,
            "assumption",
        )

    elif reference.object_type == "constraint":
        current = _find_by_id(
            snapshot.scenario.constraints or (),
            reference.object_id,
            "constraint",
        )

    elif reference.object_type == "criterion":
        current = _find_by_id(
            snapshot.criteria,
            reference.object_id,
            "criterion",
        )

    elif reference.object_type == "alternative":
        current = _find_by_id(
            snapshot.alternatives,
            reference.object_id,
            "alternative",
        )

    elif reference.object_type == "group":
        matches = [
            context.group
            for context in snapshot.groups
            if (
                context.group.id
                == reference.object_id
            )
        ]

        if len(matches) != 1:
            raise EvidenceResolutionError(
                "Unknown or ambiguous "
                "session-profile group."
            )

        current = matches[0]

    elif reference.object_type == "calculation":
        matches = [
            calculation
            for calculation
            in _session_calculations(snapshot)
            if (
                calculation.entity_id
                == reference.object_id
            )
        ]

        if len(matches) != 1:
            raise EvidenceResolutionError(
                "Unknown or ambiguous "
                "session-profile calculation."
            )

        current = matches[0]

    elif reference.object_type == "diagnostic":
        current = _find_diagnostic(
            snapshot.diagnostics,
            reference.object_id,
        )

    elif reference.object_type == "processing":
        if (
            reference.object_id
            != PROCESSING_EVIDENCE_ID
        ):
            raise EvidenceResolutionError(
                "Unknown processing evidence object."
            )

        current = snapshot.processing

    else:
        raise EvidenceResolutionError(
            "Unsupported session-profile "
            "evidence object type."
        )

    return _resolve_path(
        current,
        reference.field_path,
    )


def resolve_group_profile_evidence(
    snapshot: GroupProfileInput,
    reference: EvidenceReference,
) -> object:
    if reference.object_type == "scenario":
        if reference.object_id != snapshot.scenario.scenario_id:
            raise EvidenceResolutionError(
                "Unknown scenario evidence object."
            )
        current: object = snapshot.scenario

    elif reference.object_type == "group":
        if reference.object_id != snapshot.group.id:
            raise EvidenceResolutionError(
                "Unknown group evidence object."
            )
        current = snapshot.group

    elif reference.object_type == "calculation":
        calculation = snapshot.calculation

        if (
            calculation is None
            or calculation.entity_id != reference.object_id
        ):
            raise EvidenceResolutionError(
                "Unknown calculation evidence object."
            )

        current = calculation

    elif reference.object_type == "criterion":
        current = _find_by_id(
            snapshot.criteria,
            reference.object_id,
            "criterion",
        )

    elif reference.object_type == "alternative":
        current = _find_by_id(
            snapshot.alternatives,
            reference.object_id,
            "alternative",
        )

    elif reference.object_type == "diagnostic":
        current = _find_diagnostic(
            snapshot.diagnostics,
            reference.object_id,
        )

    else:
        raise EvidenceResolutionError(
            "Unsupported evidence object type."
        )

    return _resolve_path(current, reference.field_path)