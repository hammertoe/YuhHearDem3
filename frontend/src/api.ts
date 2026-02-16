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
  source_kind?: 'utterance' | 'bill_excerpt';
  citation_id?: string | null;
  utterance_id: string;
  youtube_video_id: string;
  youtube_url: string;
  seconds_since_start: number;
  timestamp_str: string;
  speaker_id: string;
  speaker_name: string;
  speaker_title?: string | null;
  text: string;
  video_title: string | null;
  video_date: string | null;
  bill_id?: string | null;
  bill_number?: string | null;
  bill_title?: string | null;
  excerpt?: string | null;
  source_url?: string | null;
  chunk_index?: number | null;
  page_number?: number | null;
  matched_terms?: string[] | null;
};

export type ChatResponse = {
  thread_id: string;
  assistant_message: ThreadMessage;
  sources: ChatSource[];
  focus_node_ids: string[];
  followup_questions: string[];
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

export type ChatProgressCallback = (stage: string, message: string) => void;
export type ChatFinalCallback = (response: ChatResponse) => void;
export type ChatErrorCallback = (error: string) => void;

export function sendMessageStream(
  threadId: string,
  content: string,
  onProgress: ChatProgressCallback,
  onFinal: ChatFinalCallback,
  onError?: ChatErrorCallback
): () => void {
  let cancelled = false;

  (async () => {
    try {
      const response = await fetch(
        `/chat/threads/${encodeURIComponent(threadId)}/messages/stream?content=${encodeURIComponent(content)}`
      );

      if (!response.ok) {
        onError?.(`Server error: ${response.status}`);
        return;
      }

      if (!response.body) {
        onError?.('No response body');
        return;
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (!cancelled) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (cancelled) break;
          if (!line.trim() || !line.startsWith('data: ')) continue;

          try {
            const data = JSON.parse(line.slice(6));

            if (data.stage) {
              onProgress(data.stage, data.message);
            } else if (data.thread_id) {
              onFinal({
                thread_id: data.thread_id,
                assistant_message: data.assistant_message,
                sources: data.sources || [],
                focus_node_ids: data.focus_node_ids || [],
                followup_questions: data.followup_questions || [],
                debug: data.debug || null,
              });
            } else if (data.error) {
              onError?.(data.error);
            }
          } catch {
            // Ignore parse errors
          }
        }
      }
    } catch (e) {
      if (!cancelled) {
        onError?.(e instanceof Error ? e.message : 'Connection error');
      }
    }
  })();

  return () => {
    cancelled = true;
  };
}
