"""Deprecated: the operations behind `memrank submit` and the MCP server. Evaluate an agent with
`memrank run` instead; see the README. `catalogs.get_eval` is used by the current path
(`cli/evals`) and does not warn.

Presentation-independent operations shared by the CLI and MCP surfaces."""

from memrank.application.planning import plan_sweep
from memrank.application.types import SweepPlan, SweepRequest

__all__ = ["SweepPlan", "SweepRequest", "plan_sweep"]
