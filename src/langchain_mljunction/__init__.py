"""LangChain integrations for ML Junction."""

from langchain_mljunction._client import MLJunctionAPIError
from langchain_mljunction.chat_models import ChatMLJunction
from langchain_mljunction.embeddings import MLJunctionEmbeddings
from langchain_mljunction.outcomes import OutcomeReceipt, Outcomes, request_id_of
from langchain_mljunction.telemetry import (
    MAX_AGENT_DEPTH,
    MLJunction,
    RunContext,
    expose_agent_as_tool,
)
from langchain_mljunction.tracer import MLJunctionTracer

__all__ = [
    "MAX_AGENT_DEPTH",
    "ChatMLJunction",
    "MLJunction",
    "MLJunctionAPIError",
    "MLJunctionEmbeddings",
    "MLJunctionTracer",
    "OutcomeReceipt",
    "Outcomes",
    "RunContext",
    "expose_agent_as_tool",
    "request_id_of",
]
__version__ = "0.1.0"
