import { useEffect, useMemo, useRef, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
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

function formatSpeakerName(s: ChatSource): string {
  const raw = (s.speaker_name || '').trim();
  if (raw && !raw.startsWith('s_')) {
    const looksLower = raw === raw.toLowerCase();
    if (!looksLower) return raw;

    const words = raw.split(/\s+/g).filter(Boolean);
    const fixed = words
      .map((w) => {
        const lw = w.toLowerCase();
        if (lw === 'hon' || lw === 'hon.' || lw === 'honourable') return 'The Honourable';
        if (lw === 'mr' || lw === 'mr.') return 'Mr.';
        if (lw === 'ms' || lw === 'ms.') return 'Ms.';
        if (lw === 'mrs' || lw === 'mrs.') return 'Mrs.';
        if (lw === 'dr' || lw === 'dr.') return 'Dr.';
        return w.slice(0, 1).toUpperCase() + w.slice(1);
      })
      .join(' ')
      .replace(/\s+/g, ' ')
      .trim();

    return fixed;
  }

  const sid = (s.speaker_id || raw || '').trim();
  if (!sid) return 'Unknown speaker';
  return sid
    .replace(/^s_/, '')
    .replace(/_/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

function YouTubeIcon({ className }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 24 24"
      aria-hidden="true"
      className={className}
      focusable="false"
    >
      <path
        fill="currentColor"
        d="M23.5 6.2a3.1 3.1 0 0 0-2.2-2.2C19.3 3.5 12 3.5 12 3.5s-7.3 0-9.3.5A3.1 3.1 0 0 0 .5 6.2 32.4 32.4 0 0 0 0 12s0 3.1.5 5.8a3.1 3.1 0 0 0 2.2 2.2c2 .5 9.3.5 9.3.5s7.3 0 9.3-.5a3.1 3.1 0 0 0 2.2-2.2A32.4 32.4 0 0 0 24 12s0-3.1-.5-5.8ZM9.6 15.5v-7L15.9 12l-6.3 3.5Z"
      />
    </svg>
  );
}

function MarkdownMessage({ content }: { content: string }) {
  return (
    <div className="md">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          a: ({ children, ...props }) => (
            <a {...props} target="_blank" rel="noreferrer" />
          ),
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
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
                      <MarkdownMessage content={m.content} />

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
                                <div className="flex items-start gap-3">
                                  <div className="mt-0.5 flex h-7 w-7 items-center justify-center rounded-lg bg-red-600 text-white shadow-sm">
                                    <YouTubeIcon className="h-4 w-4" />
                                  </div>

                                  <div className="min-w-0 flex-1">
                                    <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
                                      <div className="text-xs font-semibold text-ink">
                                        {formatSpeakerName(s)}
                                      </div>
                                      {s.timestamp_str && (
                                        <span className="source-pill">@ {s.timestamp_str}</span>
                                      )}
                                      {s.video_date && (
                                        <span className="source-pill">{s.video_date}</span>
                                      )}
                                    </div>

                                    {s.video_title && (
                                      <div className="mt-1 text-xs text-ink/70 source-title">
                                        {s.video_title}
                                      </div>
                                    )}

                                    <div className="mt-2 text-xs text-ink/80 source-snippet">
                                      <em>{s.text}</em>
                                    </div>
                                  </div>
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
