"""Human-in-the-loop review of assessments that the guardrail sent to pending_review.

- Four eyes: the reviewer must be a different advisor from the one who requested the assessment.
- Only pending_review assessments can be decided, and only once.
- Pending assessments are stored as files so review can happen later, by someone else.
"""

from datetime import UTC, datetime
from pathlib import Path

from app.audit import append_event
from app.data import DATA_DIR
from app.models import AssessmentStatus, FinalAssessment, ReviewDecision
from app.tools import DomainError

PENDING_DIR = DATA_DIR / "pending"


class ReviewError(DomainError):
    pass


class FourEyesViolation(ReviewError):
    pass


def _path(assessment_id: str) -> Path:
    return PENDING_DIR / f"{assessment_id}.json"


def save_pending(assessment: FinalAssessment) -> Path:
    if assessment.status != AssessmentStatus.PENDING_REVIEW:
        raise ReviewError(f"assessment is {assessment.status}, not pending_review")
    PENDING_DIR.mkdir(parents=True, exist_ok=True)
    path = _path(str(assessment.assessment_id))
    path.write_text(assessment.model_dump_json(indent=2), encoding="utf-8")
    return path


def load_pending(assessment_id: str) -> FinalAssessment:
    path = _path(assessment_id)
    if not path.is_file():
        raise ReviewError(f"no pending assessment {assessment_id}")
    return FinalAssessment.model_validate_json(path.read_text(encoding="utf-8"))


def apply_decision(
    assessment: FinalAssessment, reviewer_id: str, decision: ReviewDecision
) -> FinalAssessment:
    if assessment.status != AssessmentStatus.PENDING_REVIEW:
        raise ReviewError(f"assessment is {assessment.status}; only pending_review can be decided")
    if reviewer_id == assessment.advisor_id:
        raise FourEyesViolation(
            f"{reviewer_id} requested this assessment and cannot also approve it "
            "(policies.md#four-eyes)"
        )
    status = (
        AssessmentStatus.APPROVED_BY_HUMAN
        if decision == ReviewDecision.APPROVED
        else AssessmentStatus.REJECTED
    )
    # Re-validate rather than model_copy, so the reviewer ID is checked like any other input.
    decided = FinalAssessment.model_validate(
        assessment.model_dump()
        | {
            "status": status,
            "reviewed_by": reviewer_id,
            "reviewed_at": datetime.now(UTC),
            "review_decision": decision,
        }
    )
    append_event("human_decision", decided, actor=reviewer_id)
    _path(str(assessment.assessment_id)).unlink(missing_ok=True)
    return decided
