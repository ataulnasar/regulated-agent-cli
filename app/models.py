"""Domain models.

Every boundary in the system is typed: data loaded from disk, tool inputs/outputs, and above all
what the model submits. The model's output (AgentRecommendation) deliberately has no fields for
approval or status. Those are decided by deterministic code, and extra fields are rejected, so the
model cannot grant itself authority by adding e.g. {"approved": true}.
"""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Any, Self
from uuid import UUID, uuid4

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, StringConstraints

CustomerId = Annotated[str, StringConstraints(pattern=r"^CUST-\d{3}$")]
ProductId = Annotated[str, StringConstraints(pattern=r"^PROD-\d{3}$")]
AdvisorId = Annotated[str, StringConstraints(pattern=r"^ADV-\d{2,}$")]
# A citation points at a retrievable source: "policies.md#suitability", "products.json#PROD-001".
Citation = Annotated[str, StringConstraints(pattern=r"^[\w.-]+#[\w-]+$")]


class DomainModel(BaseModel):
    """Base for all domain models: immutable, and unknown fields are an error, not ignored."""

    model_config = ConfigDict(frozen=True, extra="forbid")


# --- Enums -------------------------------------------------------------------------------------


class RiskLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Intent(StrEnum):
    RETIREMENT_PLANNING = "retirement_planning"
    SAVINGS = "savings"
    GROWTH_INVESTING = "growth_investing"
    GENERAL_QUESTION = "general_question"
    UNKNOWN = "unknown"


class RecommendedAction(StrEnum):
    RECOMMEND_PRODUCT = "recommend_product"
    ADVISOR_REVIEW = "advisor_review"
    DECLINE = "decline"
    NEED_MORE_INFO = "need_more_info"


class ReviewDecision(StrEnum):
    APPROVED = "approved"
    REJECTED = "rejected"


class AssessmentStatus(StrEnum):
    PENDING_REVIEW = "pending_review"
    AUTO_APPROVED = "auto_approved"
    APPROVED_BY_HUMAN = "approved_by_human"
    REJECTED = "rejected"
    DECLINED = "declined"


# --- Customer ----------------------------------------------------------------------------------


class CustomerProfile(DomainModel):
    """Full customer record. Stays inside the system; the model sees ModelCustomerView instead.

    Horizon and risk tolerance may be missing: an incomplete profile is a real case that must be
    escalated, not rejected at load time.
    """

    customer_id: CustomerId
    name: Annotated[str, StringConstraints(min_length=1)]
    age: Annotated[int, Field(ge=18, le=120)]
    risk_tolerance: RiskLevel | None
    investment_horizon_years: Annotated[int, Field(ge=0, le=50)] | None
    annual_income_sek: Annotated[int, Field(ge=0)]


def _age_band(age: int) -> str:
    low = age // 10 * 10
    return f"{low}-{low + 9}"


def _income_band(income_sek: int) -> str:
    if income_sek < 400_000:
        return "<400k"
    if income_sek <= 800_000:
        return "400k-800k"
    return ">800k"


class ModelCustomerView(DomainModel):
    """The minimised customer view sent to the LLM: no name, banded age and income."""

    customer_ref: CustomerId
    age_band: str
    risk_tolerance: RiskLevel | None
    investment_horizon_years: int | None
    income_band: str

    @classmethod
    def from_profile(cls, profile: CustomerProfile) -> Self:
        return cls(
            customer_ref=profile.customer_id,
            age_band=_age_band(profile.age),
            risk_tolerance=profile.risk_tolerance,
            investment_horizon_years=profile.investment_horizon_years,
            income_band=_income_band(profile.annual_income_sek),
        )


# --- Catalogue and policy ----------------------------------------------------------------------


class Product(DomainModel):
    product_id: ProductId
    name: Annotated[str, StringConstraints(min_length=1)]
    risk_level: RiskLevel
    min_horizon_years: Annotated[int, Field(ge=0, le=50)]
    # MiFID II distinguishes complex products (e.g. leveraged trackers) with stricter rules.
    is_complex: bool = False
    description: str
    source_id: Citation


class PolicySection(DomainModel):
    section_id: Annotated[str, StringConstraints(pattern=r"^[a-z0-9-]+$")]
    heading: str
    text: str
    citation: Citation


# --- Tools -------------------------------------------------------------------------------------


class EligibilityResult(DomainModel):
    allowed: bool
    reasons: list[str] = Field(default_factory=list)


class ToolCall(DomainModel):
    """Record of one tool invocation. result_summary must be short and contain no PII."""

    tool_name: str
    arguments: dict[str, Any]
    success: bool
    result_summary: Annotated[str, StringConstraints(max_length=300)]
    duration_ms: Annotated[float, Field(ge=0)]


# --- Model output and decisions ----------------------------------------------------------------


class AgentRecommendation(DomainModel):
    """The ONLY structure the model is allowed to submit.

    model_confidence is self-reported and uncalibrated. It is recorded for information and is
    never used to gate a compliance decision.
    """

    intent: Intent
    recommended_action: RecommendedAction
    recommended_product_id: ProductId | None = None
    reasoning_summary: Annotated[str, StringConstraints(min_length=1, max_length=600)]
    citations: list[Citation] = Field(default_factory=list)
    model_confidence: Annotated[float, Field(ge=0.0, le=1.0)] | None = None


class ComplianceResult(DomainModel):
    """Output of the deterministic guardrail. Never produced by the model."""

    allowed: bool
    risk_level: RiskLevel
    requires_human_review: bool
    reasons: list[str] = Field(default_factory=list)
    rules_triggered: list[str] = Field(default_factory=list)


class FinalAssessment(DomainModel):
    assessment_id: UUID = Field(default_factory=uuid4)
    trace_id: Annotated[str, StringConstraints(min_length=1)]
    customer_id: CustomerId
    advisor_id: AdvisorId
    request: Annotated[str, StringConstraints(min_length=1, max_length=2000)]
    recommendation: AgentRecommendation
    compliance: ComplianceResult
    tool_calls: list[ToolCall] = Field(default_factory=list)
    status: AssessmentStatus
    created_at: AwareDatetime = Field(default_factory=lambda: datetime.now(UTC))
