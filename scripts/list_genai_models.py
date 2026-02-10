from __future__ import annotations

import os

from google import genai
from google.genai import types


def main() -> int:
    api_key = os.getenv("GOOGLE_API_KEY")
    provider = (os.getenv("EMBEDDING_PROVIDER") or "google_ai").strip().lower()

    if provider == "vertex_ai":
        project = os.getenv("VERTEX_PROJECT")
        location = os.getenv("VERTEX_LOCATION")
        if not project or not location:
            print(
                "❌ EMBEDDING_PROVIDER=vertex_ai but VERTEX_PROJECT/VERTEX_LOCATION not set"
            )
            return 1
        client = genai.Client(vertexai=True, project=project, location=location)
    else:
        if not api_key:
            print("❌ GOOGLE_API_KEY is not set in your environment")
            print("   Export it, then rerun: export GOOGLE_API_KEY=... ")
            return 1
        client = genai.Client(api_key=api_key)

    models = list(client.models.list())
    print(f"Found {len(models)} models")
    print("\nModels supporting embedContent:")

    embed_models: list[str] = []
    for m in models:
        name = getattr(m, "name", "")
        methods = getattr(m, "supported_generation_methods", []) or []
        # Some endpoints omit embedContent from supported_generation_methods even when it works.
        if "embedContent" in methods:
            embed_models.append(name)
            print(f"- {name} ({', '.join(methods)})")

    if not embed_models:
        print("(none)")
        print(
            "\nNote: some model listings omit embedContent from supported_generation_methods."
        )
        print("Probing common embedding models with a real embed call...")

        probe_candidates = [
            os.getenv("EMBEDDING_MODEL") or "",
            "gemini-embedding-001",
            "models/gemini-embedding-001",
            "text-embedding-004",
            "models/text-embedding-004",
        ]
        seen: set[str] = set()
        probe_candidates = [
            m for m in probe_candidates if m and not (m in seen or seen.add(m))
        ]

        for model_name in probe_candidates:
            try:
                result = client.models.embed_content(
                    model=model_name,
                    contents="probe",
                    config=types.EmbedContentConfig(
                        output_dimensionality=768,
                        task_type="RETRIEVAL_DOCUMENT",
                    ),
                )
                emb = result.embeddings[0].values
                print(f"✅ embedContent works: {model_name} (dims={len(emb)})")
                print(f'Suggested env var: export EMBEDDING_MODEL="{model_name}"')
                break
            except Exception as e:
                msg = str(e).splitlines()[0] if str(e) else repr(e)
                print(f"❌ {model_name}: {msg}")

        print(
            "\nIf none work, your key/project may not have embeddings enabled on this endpoint."
        )
        print(
            "If you're using Vertex AI instead of AI Studio, set EMBEDDING_PROVIDER=vertex_ai and configure VERTEX_PROJECT/VERTEX_LOCATION."
        )
    else:
        print("\nSuggested env var:")
        print(f'export EMBEDDING_MODEL="{embed_models[0]}"')

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
