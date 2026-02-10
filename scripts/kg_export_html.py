from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone


_REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


from lib.db.postgres_client import PostgresClient


def _color_for_type(node_type: str) -> str:
    t = (node_type or "").lower()
    if "foaf:person" in t:
        return "#2563eb"  # blue
    if "schema:legislation" in t:
        return "#16a34a"  # green
    if "schema:organization" in t:
        return "#f59e0b"  # amber
    if "schema:place" in t:
        return "#9333ea"  # purple
    if "skos:concept" in t:
        return "#0f766e"  # teal
    return "#64748b"  # slate


def _escape_html(text: str) -> str:
    return (
        (text or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def _build_html(*, title: str, nodes: list[dict], edges: list[dict], meta: dict) -> str:
    # Use vis-network via CDN to avoid requiring pyvis.
    payload = {
        "nodes": nodes,
        "edges": edges,
        "meta": meta,
    }
    payload_json = json.dumps(payload)

    template = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>__TITLE__</title>
    <style>
      :root {{
        --bg: #0b1020;
        --panel: rgba(255,255,255,0.06);
        --panel2: rgba(255,255,255,0.08);
        --text: rgba(255,255,255,0.92);
        --muted: rgba(255,255,255,0.65);
        --border: rgba(255,255,255,0.12);
      }}
      html, body {{ height: 100%; }}
      body {{
        margin: 0;
        background: radial-gradient(1200px 800px at 20% 10%, rgba(37,99,235,0.18), transparent 60%),
                    radial-gradient(900px 700px at 80% 20%, rgba(15,118,110,0.18), transparent 55%),
                    radial-gradient(900px 700px at 50% 100%, rgba(22,163,74,0.12), transparent 55%),
                    var(--bg);
        color: var(--text);
        font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
      }}
      .wrap {{ display: grid; grid-template-columns: 360px 1fr; height: 100vh; }}
      .sidebar {{
        padding: 16px;
        border-right: 1px solid var(--border);
        background: linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.03));
        overflow: auto;
      }}
      .card {{
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 14px;
        margin-bottom: 12px;
      }}
      h1 {{ font-size: 16px; margin: 0 0 6px; letter-spacing: 0.2px; }}
      .meta {{ font-size: 12px; color: var(--muted); line-height: 1.45; }}
      .label {{ font-size: 12px; color: var(--muted); margin-top: 10px; }}
      input {{
        width: 100%;
        padding: 10px 10px;
        border-radius: 10px;
        border: 1px solid var(--border);
        background: var(--panel2);
        color: var(--text);
        outline: none;
      }}
      button {{
        width: 100%;
        margin-top: 10px;
        padding: 10px 10px;
        border-radius: 10px;
        border: 1px solid var(--border);
        background: rgba(255,255,255,0.10);
        color: var(--text);
        cursor: pointer;
      }}
      button:hover {{ background: rgba(255,255,255,0.14); }}
      #graph {{ height: 100vh; width: 100%; position: relative; }}
      .hint {{ font-size: 12px; color: var(--muted); margin-top: 10px; }}
      pre {{
        margin: 0;
        white-space: pre-wrap;
        word-break: break-word;
        font-size: 12px;
        color: rgba(255,255,255,0.85);
      }}
      .pill {{
        display: inline-block;
        padding: 4px 8px;
        border-radius: 999px;
        border: 1px solid var(--border);
        background: rgba(255,255,255,0.06);
        font-size: 12px;
        color: rgba(255,255,255,0.82);
      }}
    </style>
    <script src=\"https://unpkg.com/vis-network/standalone/umd/vis-network.min.js\"></script>
  </head>
  <body>
    <div class=\"wrap\">
      <aside class=\"sidebar\">
        <div class=\"card\">
          <h1>__TITLE__</h1>
          <div class=\"meta\" id=\"meta\"></div>
        </div>
        <div class=\"card\">
          <div class=\"label\">Find node (label contains)</div>
          <input id=\"q\" placeholder=\"e.g. Road Traffic Act\" />
          <button id=\"btn\">Focus first match</button>
          <div class=\"hint\">Tip: click nodes/edges to inspect evidence.</div>
        </div>
        <div class=\"card\">
          <div class=\"label\">Selection</div>
          <pre id=\"sel\">Click a node or edge.</pre>
        </div>
      </aside>
      <main id=\"graph\"></main>
    </div>

    <script>
      const payload = __PAYLOAD_JSON__;
      const meta = payload.meta || {{}};
      document.getElementById('meta').innerHTML = [
        `<span class="pill">nodes: ${payload.nodes.length}</span>`,
        `<span class="pill">edges: ${payload.edges.length}</span>`,
        meta.youtube_video_id ? `<div class="meta" style="margin-top:8px">video: <b>${meta.youtube_video_id}</b></div>` : '',
        meta.kg_run_id ? `<div class="meta">run: <b>${meta.kg_run_id}</b></div>` : '',
        meta.generated_at ? `<div class="meta">generated: ${meta.generated_at}</div>` : ''
      ].join(' ');

      const container = document.getElementById('graph');
      const nodes = new vis.DataSet(payload.nodes);
      const edges = new vis.DataSet(payload.edges);
      const network = new vis.Network(container, {{ nodes, edges }}, {{
        layout: {{ improvedLayout: true }},
        physics: {{
          enabled: true,
          solver: 'forceAtlas2Based',
          forceAtlas2Based: {{ gravitationalConstant: -40, springLength: 140, springConstant: 0.04 }},
          stabilization: {{ iterations: 200 }}
        }},
        interaction: {{ hover: true, multiselect: false }},
        nodes: {{ shape: 'dot', scaling: {{ min: 8, max: 24 }} }},
        edges: {{
          arrows: {{ to: {{ enabled: true, scaleFactor: 0.7 }} }},
          smooth: {{ enabled: true, type: 'dynamic', roundness: 0.35 }},
          color: {{ color: 'rgba(255,255,255,0.35)', highlight: 'rgba(255,255,255,0.8)' }},
          font: {{ color: 'rgba(255,255,255,0.75)', size: 10, strokeWidth: 0 }}
        }}
      }});

      // Ensure the graph is visible even if initialization occurs before the
      // container has its final dimensions.
      function fitNow() {{
        try {{ network.fit({{ animation: false }}); }} catch (e) {{ /* ignore */ }}
      }}
      network.once('stabilizationIterationsDone', fitNow);
      window.addEventListener('load', fitNow);
      window.addEventListener('resize', () => {{
        try {{ network.redraw(); }} catch (e) {{ /* ignore */ }}
      }});

      function showSelection(obj) {{
        document.getElementById('sel').textContent = JSON.stringify(obj, null, 2);
      }}

      network.on('selectNode', (params) => {{
        const id = params.nodes[0];
        showSelection(nodes.get(id));
      }});

      network.on('selectEdge', (params) => {{
        const id = params.edges[0];
        showSelection(edges.get(id));
      }});

      document.getElementById('btn').addEventListener('click', () => {{
        const q = (document.getElementById('q').value || '').toLowerCase().trim();
        if (!q) return;
        const all = nodes.get();
        const hit = all.find(n => (n.label || '').toLowerCase().includes(q));
        if (!hit) {{
          showSelection({{ error: 'No match', query: q }});
          return;
        }}
        network.selectNodes([hit.id]);
        network.focus(hit.id, {{ scale: 1.3, animation: true }});
        showSelection(hit);
      }});
    </script>
  </body>
</html>
"""

    # This template originally used doubled braces to escape literal '{' / '}' when
    # it was a Python f-string. Convert them back so CSS/JS parses correctly.
    template = template.replace("{{", "{").replace("}}", "}")

    return template.replace("__TITLE__", _escape_html(title)).replace(
        "__PAYLOAD_JSON__", payload_json
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export canonical KG from Postgres to HTML"
    )
    parser.add_argument("--output-html", default="kg_db.html", help="Output HTML file")
    parser.add_argument(
        "--youtube-video-id",
        default=None,
        help="Filter edges by youtube_video_id (nodes still include endpoints)",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Filter edges by kg_run_id (nodes still include endpoints)",
    )
    parser.add_argument(
        "--limit-nodes",
        type=int,
        default=None,
        help="Limit nodes exported (debug only)",
    )
    parser.add_argument(
        "--limit-edges",
        type=int,
        default=None,
        help="Limit edges exported (debug only)",
    )
    args = parser.parse_args()

    if args.youtube_video_id and args.run_id:
        raise SystemExit("Cannot specify both --youtube-video-id and --run-id")

    def has_column(pg: PostgresClient, table: str, column: str) -> bool:
        row = pg.execute_query(
            """
            SELECT 1
            FROM information_schema.columns
            WHERE table_name = %s AND column_name = %s
            LIMIT 1
            """,
            (table, column),
        )
        return bool(row)

    with PostgresClient() as pg:
        predicate_raw_supported = has_column(pg, "kg_edges", "predicate_raw")

        edge_query = """
            SELECT
                id,
                source_id,
                predicate,
        """
        if predicate_raw_supported:
            edge_query += "predicate_raw,\n"
        else:
            edge_query += "NULL::text as predicate_raw,\n"

        edge_query += """
                target_id,
                evidence,
                earliest_seconds,
                earliest_timestamp_str,
                confidence,
                youtube_video_id,
                kg_run_id
            FROM kg_edges
        """
        params: list[object] = []
        if args.youtube_video_id:
            edge_query += " WHERE youtube_video_id = %s"
            params.append(args.youtube_video_id)
        elif args.run_id:
            edge_query += " WHERE kg_run_id = %s"
            params.append(args.run_id)
        edge_query += " ORDER BY earliest_seconds NULLS LAST"
        if args.limit_edges:
            edge_query += " LIMIT %s"
            params.append(args.limit_edges)

        edge_rows = pg.execute_query(edge_query, tuple(params) if params else None)

        node_ids: set[str] = set()
        for r in edge_rows:
            node_ids.add(r[1])
            node_ids.add(r[4])

        # If no edges are present, fall back to exporting all nodes (or limited).
        if not node_ids:
            nodes_query = "SELECT id, label, type, aliases FROM kg_nodes"
            if args.limit_nodes:
                nodes_query += " LIMIT %s"
                node_rows = pg.execute_query(nodes_query, (args.limit_nodes,))
            else:
                node_rows = pg.execute_query(nodes_query)
        else:
            nodes_query = (
                "SELECT id, label, type, aliases FROM kg_nodes WHERE id = ANY(%s)"
            )
            node_rows = pg.execute_query(nodes_query, (list(node_ids),))

        if args.limit_nodes:
            node_rows = node_rows[: args.limit_nodes]

    nodes: list[dict] = []
    for node_id, label, node_type, aliases in node_rows:
        nodes.append(
            {
                "id": node_id,
                "label": label,
                "group": node_type,
                "color": {
                    "background": _color_for_type(node_type),
                    "border": "rgba(255,255,255,0.25)",
                    "highlight": {
                        "background": _color_for_type(node_type),
                        "border": "rgba(255,255,255,0.85)",
                    },
                },
                "title": _escape_html(
                    f"{label}\n{node_type}\n{node_id}\nAliases: {', '.join((aliases or [])[:8])}"
                ),
            }
        )

    edges: list[dict] = []
    for (
        edge_id,
        source_id,
        predicate,
        predicate_raw,
        target_id,
        evidence,
        earliest_seconds,
        earliest_timestamp_str,
        confidence,
        youtube_video_id,
        kg_run_id,
    ) in edge_rows:
        label = predicate
        if predicate_raw:
            label = f"{predicate} ({predicate_raw})"

        edges.append(
            {
                "id": edge_id,
                "from": source_id,
                "to": target_id,
                "label": label,
                "title": _escape_html(
                    "\n".join(
                        [
                            f"{predicate}",
                            f"predicate_raw: {predicate_raw or ''}",
                            f"t: {earliest_timestamp_str or ''} ({earliest_seconds or ''}s)",
                            f"confidence: {confidence if confidence is not None else ''}",
                            f"video: {youtube_video_id}",
                            f"run: {kg_run_id or ''}",
                            "",
                            (evidence or ""),
                        ]
                    )
                ),
                "arrows": "to",
            }
        )

    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "youtube_video_id": args.youtube_video_id,
        "kg_run_id": args.run_id,
    }

    html = _build_html(
        title="Canonical KG (from Postgres)",
        nodes=nodes,
        edges=edges,
        meta=meta,
    )

    with open(args.output_html, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"✅ Wrote HTML KG: {args.output_html}")
    print(f"   nodes={len(nodes)} edges={len(edges)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
