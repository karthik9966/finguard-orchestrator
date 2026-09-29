"""The graph's shared memory -- LLD §3.1.

This module used to define `AgentState` and `ComplianceReport` itself. Both now live in
`src/models.py`, where the report's validators sit beside the contracts they constrain, so this
is a re-export: `from src.graph.state import AgentState` keeps reading naturally at the top of a
node, and there is exactly one definition.

The old `CONFIDENCE_THRESHOLD` / `MAX_REFINEMENTS` / `HIGH_RISK_CONFIDENCE` module constants are
gone. They were migration shims over `config.yaml`, and a shim that reads a config value at import
time is worse than either alternative: it looks like a tunable while being frozen at the first
import. Nodes read `get_config().reasoning` where they use it.
"""

from __future__ import annotations

from src.models import AgentState, ComplianceReport, Finding, initial_state

__all__ = ["AgentState", "ComplianceReport", "Finding", "initial_state"]
