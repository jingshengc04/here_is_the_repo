"""Pydantic v2 mirror of docs/API_CONTRACT.md. Field names are the contract's camelCase names."""
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field

RiskTolerance = Literal["CONSERVATIVE", "MODERATE", "AGGRESSIVE"]
RunStatus = Literal["QUEUED", "RUNNING", "COMPLETED", "FAILED"]
AgentId = Literal["DISCOVERY", "FINANCIALS", "LANDSCAPE", "NEWS", "SOCIAL", "ORCHESTRATOR", "VERIFIER"]
AssetClass = Literal["STOCKS", "CRYPTO", "GOLD"]
SeriesKey = Literal["PORTFOLIO", "SPY", "SIXTY_FORTY", "EQUAL_WEIGHT"]
ErrorCode = Literal["VALIDATION_ERROR", "RUN_NOT_FOUND", "RATE_LIMITED", "INTERNAL"]


class IntakeInput(BaseModel):
    amountUsd: float = Field(ge=100, le=10_000_000)
    horizonYears: int = Field(ge=1, le=10)
    riskTolerance: RiskTolerance
    theme: str | None = Field(default=None, max_length=200)


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=2000)


class IntakeTurnRequest(BaseModel):
    messages: list[ChatMessage] = Field(max_length=20)


class IntakeDraft(BaseModel):
    amountUsd: float | None = None
    horizonYears: int | None = None
    riskTolerance: RiskTolerance | None = None
    theme: str | None = None


class IntakeTurn(BaseModel):
    reply: str
    intake: IntakeDraft
    isComplete: bool


class ErrorBody(BaseModel):
    code: ErrorCode
    message: str
    details: object | None = None


class APIError(BaseModel):
    error: ErrorBody


class Source(BaseModel):
    title: str
    url: str


class Finding(BaseModel):
    id: str
    agent: AgentId
    symbol: str | None
    claim: str
    sentiment: Literal["POSITIVE", "NEGATIVE", "NEUTRAL"]
    sources: list[Source]
    numbersVerified: bool | None = None


class Holding(BaseModel):
    symbol: str
    name: str
    weightPct: float
    amountUsd: float
    rationale: str
    findingIds: list[str]
    financialScore: float | None
    sentimentScore: float | None


class AssetClassSlice(BaseModel):
    assetClass: AssetClass
    weightPct: float
    rationale: str
    holdings: list[Holding]


class Allocation(BaseModel):
    slices: list[AssetClassSlice]


class Metrics(BaseModel):
    totalReturnPct: float
    cagrPct: float
    volatilityPct: float
    sharpe: float | None
    maxDrawdownPct: float
    worstMonthPct: float


class EquityCurve(BaseModel):
    dates: list[str]
    series: dict[SeriesKey, list[float]]


class SharpeCi(BaseModel):
    lower: float
    upper: float
    probPositive: float


class CrossCheck(BaseModel):
    sandboxTotalReturnPct: float
    independentTotalReturnPct: float
    isMatch: bool


class Backtest(BaseModel):
    startDate: str
    endDate: str
    metrics: dict[SeriesKey, Metrics]
    equityCurve: EquityCurve
    sharpeCi: SharpeCi | None = None
    crossCheck: CrossCheck


class Check(BaseModel):
    name: str
    isPassed: bool
    detail: str


class Verification(BaseModel):
    isApproved: bool
    checks: list[Check]
    changesRequested: list[str]


class Round(BaseModel):
    round: Literal[1, 2]
    allocation: Allocation
    backtest: Backtest
    verification: Verification


class Report(BaseModel):
    runId: str
    generatedAt: str
    intake: IntakeInput
    summary: str
    finalAllocation: Allocation
    rounds: list[Round]
    findings: list[Finding]
    disclaimer: str


class Run(BaseModel):
    runId: str
    status: RunStatus
    intake: IntakeInput
    createdAt: str
    report: Report | None = None
    error: ErrorBody | None = None


# ---------- SSE events ----------
class _Base(BaseModel):
    runId: str
    seq: int
    ts: str


class RunStarted(_Base):
    type: Literal["run.started"]
    agents: list[AgentId]


class AgentStarted(_Base):
    type: Literal["agent.started"]
    agent: AgentId
    model: str
    task: str


class AgentProgress(_Base):
    type: Literal["agent.progress"]
    agent: AgentId
    message: str


class AgentFinding(_Base):
    type: Literal["agent.finding"]
    finding: Finding


class AgentCompleted(_Base):
    type: Literal["agent.completed"]
    agent: AgentId
    summary: str
    durationMs: int


class AgentFailed(_Base):
    type: Literal["agent.failed"]
    agent: AgentId
    message: str


class AllocationProposed(_Base):
    type: Literal["allocation.proposed"]
    round: Literal[1, 2]
    allocation: Allocation


class BacktestCompleted(_Base):
    type: Literal["backtest.completed"]
    round: Literal[1, 2]
    backtest: Backtest


class VerificationCompleted(_Base):
    type: Literal["verification.completed"]
    round: Literal[1, 2]
    verification: Verification


class RunCompleted(_Base):
    type: Literal["run.completed"]
    report: Report


class RunFailed(_Base):
    type: Literal["run.failed"]
    error: ErrorBody


RunEvent = Annotated[
    Union[
        RunStarted, AgentStarted, AgentProgress, AgentFinding, AgentCompleted, AgentFailed,
        AllocationProposed, BacktestCompleted, VerificationCompleted, RunCompleted, RunFailed,
    ],
    Field(discriminator="type"),
]
