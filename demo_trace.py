"""Demo script to show trace logging in action."""

import asyncio
import os

os.environ["CHAT_TRACE"] = "1"

from lib.db.postgres_client import PostgresClient
from lib.embeddings.google_client import GoogleEmbeddingClient
from lib.kg_agent_loop import KGAgentLoop
from lib.utils.config import config


async def demo_trace() -> None:
    """Run a simple query with trace enabled."""
    print("\n" + "=" * 60)
    print("CHAT TRACE DEMO")
    print("=" * 60)
    print(f"\nCHAT_TRACE enabled: {getattr(config, 'chat_trace', False)}")
    print("\nRunning query: 'What did ministers say about water management?'\n")

    postgres = PostgresClient()
    embedding_client = GoogleEmbeddingClient()
    loop = KGAgentLoop(
        postgres=postgres,
        embedding_client=embedding_client,
        model="gemini-2.5-flash",
    )

    try:
        result = await loop.run(
            user_message="What did ministers say about water management?",
            history=[],
        )
        print("\n" + "=" * 60)
        print("RESULT SUMMARY")
        print("=" * 60)
        print(f"Answer length: {len(result.get('answer', ''))} chars")
        print(f"Citation IDs: {result.get('cite_utterance_ids', [])[:3]}...")
        print(f"Focus nodes: {result.get('focus_node_ids', [])[:3]}...")
    finally:
        postgres.close()

    print("\n" + "=" * 60)


if __name__ == "__main__":
    asyncio.run(demo_trace())
