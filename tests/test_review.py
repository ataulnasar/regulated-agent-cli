import pytest
from pydantic import ValidationError

from app import review
from app.agent import run_assessment
from app.models import AssessmentStatus, ReviewDecision
from app.providers.fake import FakeProvider
from app.review import FourEyesViolation, ReviewError, apply_decision, load_pending, save_pending


def pending_assessment():
    assessment = run_assessment(
        "CUST-001", "ADV-01", "I want a low-risk investment for retirement", FakeProvider()
    )
    assert assessment.status == AssessmentStatus.PENDING_REVIEW
    return assessment


def test_pending_round_trip() -> None:
    assessment = pending_assessment()

    path = save_pending(assessment)

    assert path.parent == review.PENDING_DIR
    assert load_pending(str(assessment.assessment_id)) == assessment


def test_approval_by_second_advisor() -> None:
    assessment = pending_assessment()
    save_pending(assessment)

    decided = apply_decision(assessment, "ADV-02", ReviewDecision.APPROVED)

    assert decided.status == AssessmentStatus.APPROVED_BY_HUMAN
    assert decided.reviewed_by == "ADV-02"
    assert decided.reviewed_at is not None
    assert decided.released
    # Decided assessments leave the pending queue.
    with pytest.raises(ReviewError):
        load_pending(str(assessment.assessment_id))


def test_rejection_by_second_advisor() -> None:
    decided = apply_decision(pending_assessment(), "ADV-02", ReviewDecision.REJECTED)

    assert decided.status == AssessmentStatus.REJECTED
    assert decided.review_decision == ReviewDecision.REJECTED
    assert not decided.released


def test_four_eyes_requester_cannot_approve() -> None:
    with pytest.raises(FourEyesViolation, match="four-eyes"):
        apply_decision(pending_assessment(), "ADV-01", ReviewDecision.APPROVED)


def test_reviewer_id_is_validated() -> None:
    with pytest.raises(ValidationError):
        apply_decision(pending_assessment(), "bob", ReviewDecision.APPROVED)


def test_cannot_decide_twice() -> None:
    decided = apply_decision(pending_assessment(), "ADV-02", ReviewDecision.APPROVED)

    with pytest.raises(ReviewError, match="only pending_review"):
        apply_decision(decided, "ADV-03", ReviewDecision.REJECTED)


def test_only_pending_assessments_can_be_saved() -> None:
    growth = run_assessment("CUST-002", "ADV-01", "high growth equity", FakeProvider())
    assert growth.status == AssessmentStatus.AUTO_APPROVED

    with pytest.raises(ReviewError):
        save_pending(growth)


def test_unknown_pending_assessment() -> None:
    with pytest.raises(ReviewError, match="no pending assessment"):
        load_pending("does-not-exist")
