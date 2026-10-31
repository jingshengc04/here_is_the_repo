"""Pydantic v2 mirror of docs/API_CONTRACT.md. Field names are the contract's camelCase names."""
from typing import Annotated, Literal, Union

from pydantic import BaseModel, model_validator, Field

RiskTolerance = Literal["CONSERVATIVE", "MODERATE", "AGGRESSIVE"]
RunStatus = Literal["QUEUED", "RUNNING", "COMPLETED", "FAILED"]
AgentId = Literal["DISCOVERY", "FINANCIALS", "LANDSCAPE", "NEWS", "SOCIAL", "ORCHESTRATOR", "VERIFIER"]
AssetClass = Literal["STOCKS", "CRYPTO", "GOLD", "BONDS"]
SeriesKey = Literal["PORTFOLIO", "SPY", "SIXTY_FORTY", "EQUAL_WEIGHT"]
ErrorCode = Literal["VALIDATION_ERROR", "RUN_NOT_FOUND", "RATE_LIMITED", "INTERNAL"]


Style = Literal["GROWTH", "VALUE", "DIVIDEND", "MOMENTUM", "MEME", "ESG", "DEFENSIVE"]


class Preferences(BaseModel):
    """Personalisation captured by the Intake chat. Every agent reads it; the Verifier checks it."""
    mustInclude: list[str] = Field(default_factory=list, max_length=10)   # symbols the user insists on, e.g. ["DOGE-USD","NVDA"]
    exclude: list[str] = Field(default_factory=list, max_length=20)       # symbols OR plain words: ["TSLA","oil","tobacco"]
    assetClassTargets: dict[AssetClass, float] | None = None             # e.g. {"CRYPTO": 30}; each 0–100, sum ≤ 100
    styles: list[Style] = Field(default_factory=list, max_length=7)
    notes: str | None = Field(default=None, max_length=500)              # free text every agent reads
    onlyInclude: list[str] = Field(default_factory=list, max_length=10)  # restrict the universe: ["biotech"], ["European banks"]

    @model_validator(mode="after")
    def _targets_ok(self):
        t = self.assetClassTargets or {}
        if any(not 0 <= v <= 100 for v in t.values()) or sum(t.values()) > 100:
            raise ValueError("assetClassTargets must each be 0–100 and sum to at most 100")
        if any(len(x) > 40 for x in self.mustInclude + self.exclude + self.onlyInclude):
            raise ValueError("symbols/words must be at most 40 chars")
        return self


class IntakeInput(BaseModel):
    amountUsd: float = Field(ge=100, le=10_000_000)
    horizonYears: float = Field(ge=0.25, le=10, multiple_of=0.25)  # 0.25 = 3 months
    riskTolerance: RiskTolerance
    theme: str | None = Field(default=None, max_length=200)
    preferences: Preferences | None = None


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=2000)


class IntakeTurnRequest(BaseModel):
    messages: list[ChatMessage] = Field(max_length=20)


class IntakeDraft(BaseModel):
    amountUsd: float | None = None
    horizonYears: float | None = None
    riskTolerance: RiskTolerance | None = None
    theme: str | None = None
    preferences: Preferences | None = None


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
    isUserRequested: bool = False   # true if the symbol came from preferences.mustInclude


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
    preferenceNotes: list[str] = Field(default_factory=list)  # how each preference was honoured, or why it was capped


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


# ---------- Internal (not in the API contract): research → decision hand-off ----------
class Candidate(BaseModel):
    """One researched symbol, produced by agents.py and consumed by decide.py."""
    symbol: str
    name: str
    assetClass: AssetClass
    financialScore: float | None = None   # 1–10, stocks only (FINANCIALS)
    sentimentScore: float | None = None   # -1.0 … 1.0 (SOCIAL + NEWS blend)
    isUserRequested: bool = False         # came from preferences.mustInclude


# ---------- Internal: preferences resolved into Yahoo's taxonomy (built by app/preferences.py) ----------
class Filter(BaseModel):
    """Canonical filter. Strings match Yahoo profile fields case-insensitively (substring)."""
    symbols: list[str] = Field(default_factory=list)       # exact tickers, e.g. "NVDA", "DOGE-USD"
    sectors: list[str] = Field(default_factory=list)       # one of preferences.SECTORS
    industries: list[str] = Field(default_factory=list)    # Yahoo industry names/fragments, e.g. "Biotechnology"
    countries: list[str] = Field(default_factory=list)     # e.g. "China", "United States"
    assetClasses: list[AssetClass] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)      # last resort: substring of the company name

    def is_empty(self) -> bool:
        return not any((self.symbols, self.sectors, self.industries, self.countries, self.assetClasses, self.keywords))


class ResolvedPreferences(BaseModel):
    """What the user's words mean, in data terms. User preferences beat our defaults; only risk limits push back."""
    mustInclude: list[str] = Field(default_factory=list)   # resolved tickers
    prefer: Filter = Field(default_factory=Filter)          # soft: steer discovery and weights toward these
    only: Filter = Field(default_factory=Filter)            # hard: if non-empty, every holding in the classes it covers must match
    exclude: Filter = Field(default_factory=Filter)         # hard: no holding may match
    classTargets: dict[AssetClass, float] = Field(default_factory=dict)  # 0 = class not wanted at all
    interpretations: list[str] = Field(default_factory=list)  # "'bio stocks' → industry Biotechnology", shown to the user
