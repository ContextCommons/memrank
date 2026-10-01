"""Typed public contracts for Memrank planning and agent-facing operations."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

CONTRACT_VERSION = 1
Outcome = Literal["ok", "refused", "partial"]
Placement = Literal["none", "local", "cloud"]


class ContractModel(BaseModel):
    """Strict, deterministically serializable application value."""

    model_config = ConfigDict(extra="forbid")


class ModelRef(ContractModel):
    """One model in an explicit component role."""

    provider: str
    model: str
    dimensions: int | None = Field(default=None, ge=1)


class JudgeSettings(ContractModel):
    """The judge controls currently exposed by ``memrank submit``."""

    #: ``None`` means "ask the eval" -- on where the judge is the benchmark's only quality metric
    #: (locomo, longmemeval, beam), off where it scores itself. Resolved at the door that received
    #: the request, never downstream: every consumer past that point reads a decided ``bool``, so
    #: nothing has to carry "undecided" into a run. Omitting the field and sending ``false`` are
    #: different statements, which is why the default is not simply ``False``.
    enabled: bool | None = None
    samples: int = Field(default=1, ge=1)
    cache: bool = True
    allow_empty_coverage: bool = False


class ExperimentSettings(ContractModel):
    """Fixed settings used when an axis does not replace them."""

    engine_llm: ModelRef | None = None
    embedder: ModelRef | None = None
    reader: ModelRef | None = None
    #: Manifest overrides the typed component fields above cannot spell -- ``embedder.endpoint=``
    #: and its siblings -- as the same ``key=value`` tokens the CLI takes positionally. Part of the
    #: settings rather than beside them because they change which components run, and therefore
    #: what the cell IS: two runs differing only here are different experiments.
    #:
    #: Rendered BEFORE the typed fields, so a role spelled both ways resolves to the typed one --
    #: ``parse_overrides`` is last-wins, and the field that has a name in the model is the one a
    #: caller expects to win.
    overrides: list[str] = Field(default_factory=list)
    pricing_model: str = "gpt-4o-mini"
    slice: str | None = None
    tier: str | None = None
    k: int = Field(default=10, ge=1)
    repeats: int = Field(default=3, ge=1)
    token_budget: int = Field(default=5000, ge=1)
    seed: int = 42
    workers: int = Field(default=1, ge=1)
    units: list[str] = Field(default_factory=list)
    judge: JudgeSettings = Field(default_factory=JudgeSettings)


class SweepAxes(ContractModel):
    """Typed alternatives whose Cartesian product forms a sweep."""

    engine_llms: list[ModelRef] = Field(default_factory=list)
    embedders: list[ModelRef] = Field(default_factory=list)
    readers: list[ModelRef | None] = Field(default_factory=list)
    pricing_models: list[str] = Field(default_factory=list)
    slices: list[str | None] = Field(default_factory=list)
    tiers: list[str | None] = Field(default_factory=list)
    k_values: list[int] = Field(default_factory=list)
    repeat_counts: list[int] = Field(default_factory=list)
    token_budgets: list[int] = Field(default_factory=list)
    seeds: list[int] = Field(default_factory=list)
    worker_counts: list[int] = Field(default_factory=list)
    unit_selections: list[list[str]] = Field(default_factory=list)
    judges: list[JudgeSettings] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_positive_axes(self) -> SweepAxes:
        values = self.k_values + self.repeat_counts + self.token_budgets + self.worker_counts
        if any(value < 1 for value in values):
            raise ValueError("k, repeats, token budgets, and workers must be positive")
        return self


class Routing(ContractModel):
    """Where and under whose authority experiments execute.

    Carries no harness image. Which build runs a cloud evaluation is the platform's to decide --
    the server launches its own pinned image and records its tag and digest on every run. Callers
    naming one was the harness-developer door, removed because it assumed a memrank checkout, a
    Docker daemon and ECR push rights, and because it swapped only the image while the API went on
    rendering the task command from its own build.
    """

    placement: Placement = "none"
    org: str | None = None

    @model_validator(mode="after")
    def cloud_requires_org(self) -> Routing:
        if self.placement == "cloud" and not self.org:
            raise ValueError("cloud placement requires org")
        return self


class SweepRequest(ContractModel):
    """A bounded, typed request to preview an experiment matrix."""

    targets: list[str] = Field(min_length=1)
    evals: list[str] = Field(min_length=1)
    settings: ExperimentSettings = Field(default_factory=ExperimentSettings)
    axes: SweepAxes = Field(default_factory=SweepAxes)
    routing: Routing = Field(default_factory=Routing)


class Notice(ContractModel):
    """A stable machine code with an explanation and optional remedy."""

    code: str
    message: str
    experiment_ids: list[str] = Field(default_factory=list)
    remedy: str | None = None


class Requirement(Notice):
    """A condition checked without revealing its value."""

    scope: Literal["local", "org"]
    satisfied: bool | None


class Experiment(ContractModel):
    """One immutable, fully resolved matrix cell."""

    experiment_id: str
    label: str
    target_ref: str
    eval_ref: str
    target: dict[str, Any]
    settings: ExperimentSettings
    overrides: list[str]


class SweepPlan(ContractModel):
    """Self-contained, read-only preview consumed by one-cell submission."""

    contract_version: int = CONTRACT_VERSION
    plan_hash: str
    request: SweepRequest
    experiments: list[Experiment]
    requirements: list[Requirement] = Field(default_factory=list)
    warnings: list[Notice] = Field(default_factory=list)
    refusals: list[Notice] = Field(default_factory=list)
    deduplicated: int = 0


class ToolEnvelope(ContractModel):
    """Common application/MCP result shape."""

    contract_version: int = CONTRACT_VERSION
    outcome: Outcome = "ok"
    data: Any = None
    warnings: list[Notice] = Field(default_factory=list)
    refusals: list[Notice] = Field(default_factory=list)
