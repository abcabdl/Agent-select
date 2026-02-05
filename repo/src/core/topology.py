from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field


class TopologyType(str, Enum):
    SINGLE = "single"
    CENTRALIZED = "centralized"
    DECENTRALIZED = "decentralized"
    CHAIN = "chain"


class TopologyConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    topology: TopologyType = TopologyType.SINGLE
    roles: List[str] = Field(default_factory=list)
    manager_role: Optional[str] = None
    entry_role: Optional[str] = None
    max_steps: int = 6
    flow_type: Optional[str] = None
    edges: List[Tuple[str, str]] = Field(default_factory=list)
    routing_table: Dict[str, List[str]] = Field(default_factory=dict)

    def normalized(self, default_roles: Optional[List[str]] = None) -> "TopologyConfig":
        roles = [role for role in (self.roles or []) if role]
        if not roles:
            roles = [role for role in (default_roles or []) if role]

        entry_role = self.entry_role or (roles[0] if roles else None)
        manager_role = self.manager_role
        if self.topology == TopologyType.CENTRALIZED and not manager_role:
            manager_role = "planner" if "planner" in roles else (roles[0] if roles else None)

        if self.topology == TopologyType.CHAIN and not self.edges and len(roles) > 1:
            edges = [(roles[i], roles[i + 1]) for i in range(len(roles) - 1)]
        else:
            edges = list(self.edges)

        return TopologyConfig(
            topology=self.topology,
            roles=roles,
            manager_role=manager_role,
            entry_role=entry_role,
            max_steps=self.max_steps,
            flow_type=self.flow_type,
            edges=edges,
            routing_table=dict(self.routing_table),
        )

