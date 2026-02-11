import { useEffect, useMemo, useRef, useState } from 'react';
import { createThread, getThread, sendMessage, type ChatSource, type ThreadMessage } from './api';

type UIMessage = {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  createdAt?: string;
  sources?: ChatSource[];
};

const EXAMPLE_PROMPTS = [
  'What did ministers say about water management recently?',
  'Who discussed healthcare funding and what did they propose?',
  'Summarize recent debates on tourism development with sources.',
  'What bills were mentioned about education, and where?',
];

function formatSourceLabel(s: ChatSource): string {
  const who = s.speaker_name || s.speaker_id || 'Unknown speaker';
  const when = s.timestamp_str ? ` @ ${s.timestamp_str}` : '';
  const title = s.video_title ? ` - ${s.video_title}` : '';
  return `${who}${when}${title}`;
}

function App() {
  const [threadId, setThreadId] = useState<string | null>(null);
  const [messages, setMessages] = useState<UIMessage[]>([]);
  const [input, setInput] = useState('');
  const [sending, setSending] = useState(false);
  const [connected, setConnected] = useState<boolean | null>(null);
  const listEndRef = useRef<HTMLDivElement>(null);

  const sortedMessages = useMemo(() => messages, [messages]);

  useEffect(() => {
    const load = async () => {
      const saved = localStorage.getItem('yhd_thread_id');
      try {
        if (saved) {
          const t = await getThread(saved);
          setThreadId(saved);
          setConnected(true);
          setMessages(
            t.messages
              .filter((m) => m.role === 'user' || m.role === 'assistant')
              .map((m: ThreadMessage) => ({
                id: m.id,
                role: m.role as 'user' | 'assistant',
                content: m.content,
                createdAt: m.created_at,
              }))
          );
          return;
        }

        const created = await createThread(null);
        localStorage.setItem('yhd_thread_id', created.thread_id);
        setThreadId(created.thread_id);
        setConnected(true);
      } catch (_e) {
        setConnected(false);
      }
    };
    load();
  }, []);

  useEffect(() => {
    listEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [sortedMessages.length, sending]);

  const onClear = async () => {
    localStorage.removeItem('yhd_thread_id');
    setThreadId(null);
    setMessages([]);
    setInput('');
    setConnected(null);

    try {
      const created = await createThread(null);
      localStorage.setItem('yhd_thread_id', created.thread_id);
      setThreadId(created.thread_id);
      setConnected(true);
    } catch (_e) {
      setConnected(false);
    }
  };

  const onSend = async (prompt?: string) => {
    const content = (prompt ?? input).trim();
    if (!content || !threadId || sending) return;

    const optimisticUser: UIMessage = {
      id: `local_user_${Date.now()}`,
      role: 'user',
      content,
      createdAt: new Date().toISOString(),
    };

    setMessages((m) => [...m, optimisticUser]);
    setInput('');
    setSending(true);
    setConnected(true);

    try {
      const res = await sendMessage(threadId, content);
      const assistant: UIMessage = {
        id: res.assistant_message.id,
        role: 'assistant',
        content: res.assistant_message.content,
        createdAt: res.assistant_message.created_at,
        sources: res.sources || [],
      };
      setMessages((m) => [...m, assistant]);
    } catch (e) {
      setConnected(false);
      const err = e instanceof Error ? e.message : 'Failed to send message';
      setMessages((m) => [
        ...m,
        {
          id: `local_err_${Date.now()}`,
          role: 'assistant',
          content: `I couldn't reach the server. ${err}`,
          createdAt: new Date().toISOString(),
        },
      ]);
    } finally {
      setSending(false);
    }
  };

  const onKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      onSend();
    }
  };

  return (
    <div className="min-h-screen app-bg">
      <div className="mx-auto max-w-5xl px-4 py-10 sm:px-6">
        <header className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
          <div>
            <div className="flex items-baseline gap-3">
              <h1 className="text-3xl sm:text-4xl font-semibold tracking-tight text-ink">
                YuhHearDem
              </h1>
              <span className="rounded-full border border-ink/10 bg-white/60 px-2 py-1 text-xs text-ink/70 backdrop-blur">
                hybrid graph-rag
              </span>
            </div>
            <p className="mt-2 max-w-2xl text-sm sm:text-base text-ink/70">
              No more long talk - get straight to what dey really say.
            </p>
          </div>

          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2 rounded-full border border-ink/10 bg-white/60 px-3 py-2 text-sm text-ink/80 backdrop-blur">
              <span
                className={
                  connected === null
                    ? 'status-dot status-unknown'
                    : connected
                      ? 'status-dot status-ok'
                      : 'status-dot status-bad'
                }
              />
              {connected === null ? 'Connecting' : connected ? 'Connected' : 'Disconnected'}
            </div>
            <button
              type="button"
              onClick={onClear}
              className="rounded-xl border border-ink/10 bg-white/60 px-4 py-2 text-sm text-ink/80 backdrop-blur transition hover:bg-white/80"
            >
              Clear Chat
            </button>
          </div>
        </header>

        <section className="mt-8 space-y-6">
          <div className="panel">
            <div className="flex flex-col gap-3">
              <div className="flex items-start gap-3">
                <div className="mt-1 flex h-9 w-9 items-center justify-center rounded-xl bg-ink text-paper">
                  <span className="text-sm font-semibold">YHD</span>
                </div>
                <div>
                  <h2 className="text-lg sm:text-xl font-semibold text-ink">
                    Ask About Barbadian Politics
                  </h2>
                  <p className="mt-1 text-sm text-ink/70">
                    Ask anything about debates, ministers, bills, and decisions. I will answer with
                    receipts - timecoded YouTube clips you can click.
                  </p>
                </div>
              </div>

              <div className="flex flex-wrap gap-2 pt-2">
                {EXAMPLE_PROMPTS.map((p) => (
                  <button
                    key={p}
                    type="button"
                    onClick={() => onSend(p)}
                    className="chip"
                  >
                    {p}
                  </button>
                ))}
              </div>
            </div>
          </div>

          <div className="panel">
            <div className="chat-list" aria-live="polite">
              {sortedMessages.length === 0 ? (
                <div className="py-10 text-center text-sm text-ink/60">
                  Ask a question to start.
                </div>
              ) : (
                sortedMessages.map((m) => (
                  <div
                    key={m.id}
                    className={m.role === 'user' ? 'msg-row msg-user' : 'msg-row msg-assistant'}
                  >
                    <div className={m.role === 'user' ? 'msg-bubble bubble-user' : 'msg-bubble bubble-assistant'}>
                      <div className="whitespace-pre-wrap text-sm leading-relaxed">{m.content}</div>

                      {m.role === 'assistant' && m.sources && m.sources.length > 0 && (
                        <div className="mt-4 border-t border-ink/10 pt-3">
                          <div className="text-xs font-semibold uppercase tracking-wide text-ink/60">
                            Sources
                          </div>
                          <div className="mt-2 space-y-2">
                            {m.sources.slice(0, 8).map((s) => (
                              <a
                                key={s.utterance_id}
                                href={s.youtube_url}
                                target="_blank"
                                rel="noreferrer"
                                className="source-card"
                              >
                                <div className="flex items-start justify-between gap-3">
                                  <div className="text-xs font-medium text-ink">
                                    {formatSourceLabel(s)}
                                  </div>
                                  <div className="text-[11px] text-ink/60 font-mono">
                                    {s.youtube_video_id}
                                  </div>
                                </div>
                                <div className="mt-1 text-xs text-ink/70 source-snippet">
                                  {s.text}
                                </div>
                              </a>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                ))
              )}

              {sending && (
                <div className="msg-row msg-assistant">
                  <div className="msg-bubble bubble-assistant">
                    <div className="typing">
                      <span />
                      <span />
                      <span />
                    </div>
                  </div>
                </div>
              )}

              <div ref={listEndRef} />
            </div>

            <div className="mt-6 flex flex-col gap-3 sm:flex-row sm:items-end">
              <div className="flex-1">
                <label className="text-xs font-semibold uppercase tracking-wide text-ink/60">
                  Your question
                </label>
                <textarea
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={onKeyDown}
                  rows={2}
                  placeholder="Ask about debates, ministers, bills, or policies..."
                  className="composer"
                  disabled={!threadId || sending}
                />
                <div className="mt-2 text-xs text-ink/60">
                  Tip: press Enter to send, Shift+Enter for a new line.
                </div>
              </div>
              <button
                type="button"
                onClick={() => onSend()}
                disabled={!threadId || sending || input.trim().length === 0}
                className="send-btn"
              >
                {sending ? 'Sending...' : 'Send'}
              </button>
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}

export default App;
