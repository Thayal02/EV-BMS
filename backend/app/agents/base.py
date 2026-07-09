"""Shared contracts for the multi-agent system.

Every agent (Data, Prediction, Diagnosis, Maintenance, LLM Expert, Decision)
implements `BaseAgent`. Agents communicate exclusively through an
`AgentContext` that accumulates state as it passes through the pipeline -
this keeps agents decoupled: each one only reads the fields it needs and
writes the fields it owns, so agents can be reordered, parallelized, or
swapped for a graph-based orchestrator later without changing their
internals.

Concrete agents (DataAgent, PredictionAgent, ...) are implemented alongside
the features that need them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class AgentStatus(StrEnum):
    SUCCESS = "success"
    FAILURE = "failure"
    SKIPPED = "skipped"


@dataclass
class AgentContext:
    """Mutable state threaded through the agent pipeline for one analysis run.

    Fields are populated incrementally: Data Agent fills `battery_metadata`
    and `processed_dataset`; Prediction Agent fills `predictions`; Diagnosis
    Agent fills `diagnosis`; and so on. Downstream agents must not assume an
    upstream field is present without checking - an upstream agent may have
    failed or been skipped.
    """

    session_id: str
    battery_metadata: dict[str, Any] | None = None
    processed_dataset: dict[str, Any] | None = None
    predictions: dict[str, Any] | None = None
    explanations: dict[str, Any] | None = None
    diagnosis: dict[str, Any] | None = None
    recommendations: dict[str, Any] | None = None
    final_assessment: dict[str, Any] | None = None
    history: list[AgentResult] = field(default_factory=list)

    def record(self, result: AgentResult) -> None:
        self.history.append(result)


@dataclass
class AgentResult:
    agent_name: str
    status: AgentStatus
    message: str
    data: dict[str, Any] | None = None


class BaseAgent(ABC):
    """Contract every agent in the pipeline must satisfy."""

    name: str

    @abstractmethod
    async def run(self, context: AgentContext) -> AgentResult:
        """Execute this agent's responsibility and mutate `context` in place.

        Must not raise for expected/recoverable failures (e.g. malformed
        input) - those should be reported via `AgentStatus.FAILURE` in the
        returned `AgentResult` so the orchestrator and Decision Agent can
        reason about partial pipeline failures. Reserve exceptions for truly
        unexpected errors.
        """
        raise NotImplementedError
