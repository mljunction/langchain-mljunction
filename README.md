# langchain-mljunction

Native LangChain integrations for ML Junction. This package talks directly to ML Junction's
`/v1/responses` and `/v1/embeddings` APIs; it does not wrap `ChatOpenAI`.

```python
from langchain_mljunction import ChatMLJunction

llm = ChatMLJunction(
    model="gpt-4o-mini",
    api_key="astro_...",
    base_url="https://api.mljunction.com",
    routing={"strategy": "balanced", "service_tier": "auto"},
)

print(llm.invoke("Hello").content)
```

Supported: sync/async invocation, sync/async streaming, LangChain tool binding, structured
output, multimodal LangChain message content, ML Junction routing controls, sessions/tasks,
usage metadata, routing metadata and billing receipts.
