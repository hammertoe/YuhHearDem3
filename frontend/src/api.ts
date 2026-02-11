export type ThreadMessage = {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  metadata?: Record<string, unknown> | null;
  created_at: string;
};

export type Thread = {
  id: string;
  title: string | null;
  created_at: string;
  updated_at: string;
  state: Record<string, unknown>;
  messages: ThreadMessage[];
};

export type ChatSource = {
  utterance_id: string;
  youtube_video_id: string;
  youtube_url: string;
  seconds_since_start: number;
  timestamp_str: string;
  speaker_id: string;
  speaker_name: string;
  text: string;
  video_title: string | null;
  video_date: string | null;
};

export type ChatResponse = {
  thread_id: string;
  assistant_message: ThreadMessage;
  sources: ChatSource[];
  focus_node_ids: string[];
  debug?: Record<string, unknown> | null;
};

async function jsonFetch<T>(input: RequestInfo, init?: RequestInit): Promise<T> {
  const res = await fetch(input, {
    ...init,
    headers: {
      'Content-Type': 'application/json',
      ...(init?.headers || {}),
    },
  });

  if (!res.ok) {
    const text = await res.text().catch(() => '');
    throw new Error(text || `Request failed (${res.status})`);
  }
  return (await res.json()) as T;
}

export async function createThread(title?: string | null): Promise<{ thread_id: string }> {
  const query = title ? `?title=${encodeURIComponent(title)}` : '';
  return jsonFetch(`/chat/threads${query}`, { method: 'POST' });
}

export async function getThread(threadId: string): Promise<Thread> {
  return jsonFetch(`/chat/threads/${encodeURIComponent(threadId)}`);
}

export async function sendMessage(threadId: string, content: string): Promise<ChatResponse> {
  return jsonFetch(`/chat/threads/${encodeURIComponent(threadId)}/messages`, {
    method: 'POST',
    body: JSON.stringify({ content }),
  });
}
