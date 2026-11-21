from pydantic import BaseModel

from src.ai.comprehension.models import (
    DCLStageInput,
    EvidenceReference,
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
    if isinstance(snapshot, GroupProfileInput):
        return resolve_group_profile_evidence(
            snapshot,
            reference,
        )

    raise EvidenceResolutionError(f"Evidence resolution is not implemented for stage {snapshot.stage!r}.")
    
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