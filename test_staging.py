#!/usr/bin/env python3
import json
import urllib.request


def req(method, url, data=None):
    b = json.dumps(data).encode() if data else None
    r = urllib.request.Request(
        url, data=b, headers={"Content-Type": "application/json"}, method=method
    )
    return json.loads(urllib.request.urlopen(r, timeout=90).read().decode())


q = "What did ministers say about water management recently?"
for port, name in [(8013, "green"), (8003, "blue")]:
    try:
        t = req("POST", f"http://127.0.0.1:{port}/chat/threads")
        res = req(
            "POST",
            f"http://127.0.0.1:{port}/chat/threads/{t['thread_id']}/messages",
            {"content": q},
        )
        dbg = (res.get("debug") or {}).get("retrieval") or {}
        print(f"[{name}]")
        print(f"  edge_count={dbg.get('edge_count')}")
        print(f"  node_count={dbg.get('node_count')}")
        print(f"  seed_count={dbg.get('seed_count')}")
        print(f"  threshold={dbg.get('edge_rank_threshold')}")
        print(f"  filtered={dbg.get('edges_filtered_by_threshold')}")
        print(f"  skipped={dbg.get('edge_rank_filter_skipped_no_scores')}")
        print(f"  sources={len(res.get('sources') or [])}")
        print(
            f"  snippet={(res.get('assistant_message', {}).get('content', '')[:150]).replace(chr(10), ' ')}"
        )
    except Exception as e:
        print(f"[{name}] ERROR: {e}")
    print()
