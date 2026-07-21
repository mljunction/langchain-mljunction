"""User-facing tracing surface: root context, child agent context, tool wrapper.

`MLJunction` builds the `RunnableConfig` that carries tracing identity through
a run. Attach it once at the top and LangChain propagates it to every child
runnable automatically.

The one thing it cannot do on its own is follow an agent you invoke by hand
inside a tool function. LangChain propagates config through *runnables*, not
through arbitrary Python calls, so a subagent invoked as

    subagent.invoke({"messages": messages})

starts a brand new root and the tree silently splits in two - no error, no
warning, just a flat trace and a second root you did not ask for. Passing the
active config through `child_agent_config` (or using `expose_agent_as_tool`,
which does it for you) is what keeps the nesting intact.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import ensure_config

from langchain_mljunction.tracer import NS, MLJunctionTracer, new_id

__all__ = [
    "MAX_AGENT_DEPTH",
    "MLJunction",
    "RunContext",
    "expose_agent_as_tool",
]

# Runaway guard. LangChain's own `recursion_limit` (default 25) protects
# graph recursion, but not an agent that keeps spawning agents - one bad prompt
# can otherwise fan out into an expensive swarm. This ceiling is about cost and
# blast radius, not about what the tree can represent: the span model itself
# handles arbitrary depth.
MAX_AGENT_DEPTH = 12


@dataclass
class RunContext:
    """The config to invoke with, plus the ids that identify this run."""

    config: RunnableConfig
    agent_instance_id: str
    session_id: str | None = None
    task_id: str | None = None

    def __post_init__(self) -> None:
        # Callers pass `context.config` around constantly; making the object
        # itself usable as a config removes an easy footgun.
        self.config.setdefault("metadata", {})


class MLJunction:
    """Entry point for agent tracing.

    Construct once per process and reuse. The underlying exporter owns a
    background thread and an HTTP client, so creating one per request would
    leak both.
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "http://localhost:8001",
        app_id: str | None = None,
        app_name: str | None = None,
        environment: str = "production",
        capture_content: bool = True,
        tracer: MLJunctionTracer | None = None,
    ) -> None:
        self.app_id = app_id
        self.app_name = app_name
        self.environment = environment
        self.tracer = tracer or MLJunctionTracer(
            endpoint=base_url,
            api_key=api_key,
            app_id=app_id,
            app_name=app_name,
            environment=environment,
            capture_content=capture_content,
        )

    # -- root --------------------------------------------------------------

    def context(
        self,
        *,
        agent_name: str,
        session_id: str | None = None,
        task_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> RunContext:
        """Build the root config for one agent run.

        Note what this does NOT take: a request_id. In ML Junction a request id
        is minted by the gateway per HTTP call, so a single agent run produces
        many of them. Spans pick their request id up from the response as each
        model call completes; declaring one up front would be a lie.

        session_id and task_id are yours. They group requests for search and
        navigation and have nothing to do with the execution tree.
        """
        if not agent_name or not agent_name.strip():
            raise ValueError("agent_name must be a non-empty string")

        agent_instance_id = new_id()
        merged = dict(metadata or {})
        merged.update(
            {
                f"{NS}.app_id": self.app_id,
                f"{NS}.environment": self.environment,
                f"{NS}.session_id": session_id,
                f"{NS}.task_id": task_id,
                f"{NS}.agent_name": agent_name,
                f"{NS}.agent_role": "root",
                f"{NS}.agent_instance_id": agent_instance_id,
                f"{NS}.parent_agent_instance_id": None,
                f"{NS}.agent_depth": 0,
            }
        )

        config: RunnableConfig = {
            "callbacks": [self.tracer],
            "metadata": merged,
            "tags": self._deduplicate(
                [
                    "mljunction",
                    *([f"app:{self.app_id}"] if self.app_id else []),
                    f"environment:{self.environment}",
                    f"agent:{agent_name}",
                    *(tags or []),
                ]
            ),
            "run_name": f"agent:{agent_name}",
        }
        return RunContext(
            config=config,
            agent_instance_id=agent_instance_id,
            session_id=session_id,
            task_id=task_id,
        )

    # -- children ----------------------------------------------------------

    def child_agent_config(
        self,
        parent_config: RunnableConfig,
        *,
        agent_name: str,
        agent_role: str = "subagent",
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> RunnableConfig:
        """Derive config for an agent nested beneath the current one.

        Callbacks, session and task ids and the trace relationship are all
        preserved; only agent ownership changes. The tracer notices that the
        new agent_instance_id differs from its parent's and promotes the chain
        to an agent span - a purely local comparison, which is exactly why the
        same function works at every depth with no special case per level.

        Works identically for A->B, B->C, C->D. There is no level-two code.
        """
        if not agent_name or not agent_name.strip():
            raise ValueError("agent_name must be a non-empty string")

        base = ensure_config(parent_config)
        parent_metadata = dict(base.get("metadata") or {})
        parent_instance_id = parent_metadata.get(f"{NS}.agent_instance_id")
        if not parent_instance_id:
            raise ValueError(
                "The parent config carries no ML Junction agent identity. Pass the "
                "config from MLJunction.context() (or the config injected into your "
                "tool), not a fresh one - otherwise this agent starts its own trace."
            )

        try:
            parent_depth = int(parent_metadata.get(f"{NS}.agent_depth", 0))
        except (TypeError, ValueError) as error:
            raise ValueError(f"{NS}.agent_depth must be an integer") from error

        child_depth = parent_depth + 1
        if child_depth > MAX_AGENT_DEPTH:
            raise RuntimeError(
                f"Maximum nested agent depth exceeded: {child_depth} > {MAX_AGENT_DEPTH}. "
                "An agent is spawning agents recursively."
            )

        child_instance_id = new_id()
        child_metadata = dict(parent_metadata)
        child_metadata.update(metadata or {})
        # Reserved hierarchy fields are written last so caller metadata can
        # never accidentally overwrite the identity that builds the tree.
        child_metadata.update(
            {
                f"{NS}.agent_name": agent_name,
                f"{NS}.agent_role": agent_role,
                f"{NS}.agent_instance_id": child_instance_id,
                f"{NS}.parent_agent_instance_id": parent_instance_id,
                f"{NS}.agent_depth": child_depth,
            }
        )

        child_config: RunnableConfig = dict(base)
        # Never reuse the parent's run id - LangChain must mint a fresh one for
        # this invocation, or parent and child collapse into one span.
        child_config.pop("run_id", None)
        child_config["metadata"] = child_metadata
        child_config["tags"] = self._deduplicate(
            [
                *base.get("tags", []),
                "subagent",
                f"agent:{agent_name}",
                f"agent-depth:{child_depth}",
                *(tags or []),
            ]
        )
        child_config["run_name"] = f"agent:{agent_name}"
        return child_config

    @staticmethod
    def _deduplicate(values: list[str]) -> list[str]:
        return list(dict.fromkeys(values))

    def flush(self, timeout: float = 5.0) -> bool:
        """Wait for queued spans to be sent. Useful in scripts and tests."""
        return self.tracer.flush(timeout)

    def close(self) -> None:
        self.tracer.close()

    def __enter__(self) -> MLJunction:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def expose_agent_as_tool(
    *,
    telemetry: MLJunction,
    agent: Any,
    agent_name: str,
    tool_name: str,
    description: str,
) -> Any:
    """Wrap an agent as a tool without breaking the trace.

    This is the blessed way to nest agents. Hand-rolling it is where the tree
    usually breaks: forget to thread `config` and the subagent starts a fresh
    root, silently.

    Relies on LangChain reserving a tool parameter named exactly `config` for
    injecting the active RunnableConfig. The name matters - rename it and the
    injection stops happening.
    """
    from langchain_core.tools import tool

    @tool(tool_name, description=description)
    def call_agent(query: str, config: RunnableConfig) -> str:
        child_config = telemetry.child_agent_config(
            config,
            agent_name=agent_name,
            agent_role="subagent",
            metadata={f"{NS}.delegation_tool": tool_name},
        )
        result = agent.invoke(
            {"messages": [{"role": "user", "content": query}]},
            config=child_config,
        )
        messages = result.get("messages", []) if isinstance(result, dict) else []
        if not messages:
            raise RuntimeError(f"Agent {agent_name!r} returned no messages")
        final = messages[-1]
        return str(getattr(final, "content", final))

    return call_agent
