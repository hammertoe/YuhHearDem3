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
  citationIds?: string[];
};

function extractCitationIds(metadata: unknown): string[] {
  if (!metadata || typeof metadata !== 'object') return [];
  const rec = metadata as Record<string, unknown>;
  const raw = rec.cite_utterance_ids;
  if (!Array.isArray(raw)) return [];
  return raw
    .map((v) => String(v || '').trim())
    .filter((v) => v.length > 0)
    .slice(0, 12);
}

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

function formatSpeakerTitle(s: ChatSource): string {
  const raw = (s.speaker_title || '').trim();
  if (!raw) return '';
  if (raw.toLowerCase() === 'unknown') return '';

  const smallWords = new Set(['and', 'or', 'of', 'the', 'for', 'to', 'in', 'on', 'with']);
  const normalizeToken = (token: string, i: number): string => {
    const base = token.trim();
    if (!base) return base;

    const lower = base.toLowerCase();
    if (lower === 'm.p.' || lower === 'mp') return 'M.P.';
    if (lower === 'j.p.' || lower === 'jp') return 'J.P.';
    if (lower === 's.c.' || lower === 'sc') return 'S.C.';
    if (lower === 'k.c.' || lower === 'kc') return 'K.C.';
    if (i > 0 && smallWords.has(lower)) return lower;

    return base.slice(0, 1).toUpperCase() + base.slice(1).toLowerCase();
  };

  return raw
    .split(',')
    .map((part) =>
      part
        .trim()
        .split(/\s+/g)
        .filter(Boolean)
        .map((t, i) => normalizeToken(t, i))
        .join(' ')
    )
    .join(', ');
}

function normalizeUtteranceId(id: string): string {
  let raw = decodeURIComponent(String(id || '').trim());
  raw = raw.replace(/^https?:\/\/[^#]+#src:/i, '');
  raw = raw.replace(/^#src:/i, '');
  raw = raw.replace(/^source:/i, '');
  raw = raw.replace(/^utt_/i, '');
  raw = raw.replace(/[\]\[),.;:]+$/g, '');
  return raw.trim();
}

function normalizeCitationHref(href: string): string {
  const raw = decodeURIComponent((href || '').trim());
  if (raw.startsWith('#src:')) return raw;
  if (raw.startsWith('source:')) return `#src:${raw.slice('source:'.length)}`;
  if (/^https?:\/\/[^#]+#src:/i.test(raw)) return `#src:${raw.split('#src:')[1] || ''}`;
  return raw;
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

function MarkdownMessage({
  content,
  sources,
  citationIds,
  messageId,
}: {
  content: string;
  sources?: ChatSource[];
  citationIds?: string[];
  messageId: string;
}) {
  const normalizedInlineContent = useMemo(
    () => content.replace(/\]\(source:([^)]+)\)/gi, '](#src:$1)'),
    [content]
  );

  const inlineLinkIds = useMemo(() => {
    const ids: string[] = [];
    const seen = new Set<string>();
    const re = /\]\(#src:([^)]+)\)/gi;
    let m: RegExpExecArray | null = re.exec(normalizedInlineContent);
    while (m) {
      const id = String(m[1] || '').trim();
      if (id && !seen.has(id)) {
        seen.add(id);
        ids.push(id);
      }
      m = re.exec(normalizedInlineContent);
    }
    return ids;
  }, [normalizedInlineContent]);

  const linkIds = useMemo(() => {
    const ids: string[] = [];
    const seen = new Set<string>();
    const add = (id: string) => {
      const raw = String(id || '').trim();
      if (!raw || seen.has(raw)) return;
      seen.add(raw);
      ids.push(raw);
    };

    inlineLinkIds.forEach(add);
    (sources || []).forEach((s) => add(s.utterance_id));
    (citationIds || []).forEach(add);
    return ids;
  }, [inlineLinkIds, sources, citationIds]);

  const sourceIndex = useMemo(() => {
    const m = new Map<string, number>();
    linkIds.forEach((id, i) => {
      const n = i + 1;
      const raw = (id || '').trim();
      const normalized = normalizeUtteranceId(raw);
      if (raw) m.set(raw, n);
      if (normalized) {
        m.set(normalized, n);
        m.set(`utt_${normalized}`, n);
        m.set(normalized.toLowerCase(), n);
        m.set(`utt_${normalized.toLowerCase()}`, n);
      }
    });
    return m;
  }, [linkIds]);

  const contentWithFallbackLinks = useMemo(() => {
    const hasInline = /\]\(#src:[^)]+\)/i.test(normalizedInlineContent);
    if (hasInline) return normalizedInlineContent;
    if (linkIds.length === 0) return normalizedInlineContent;

    const markers = linkIds
      .slice(0, 8)
      .map((id, i) => `[${i + 1}](#src:${id})`)
      .join(' ');
    return `${normalizedInlineContent}\n\nCitations: ${markers}`;
  }, [linkIds, normalizedInlineContent]);

  const onJumpToSource = (utteranceId: string) => {
    const normalized = normalizeUtteranceId(utteranceId);
    const el =
      document.getElementById(`src-${messageId}-${normalized}`) ||
      document.getElementById(`src-${messageId}-${utteranceId}`);
    if (el) {
      // Update hash so users can copy/paste links and :target styling works.
      history.replaceState(null, '', `#${el.id}`);
      el.scrollIntoView({ behavior: 'smooth', block: 'center' });
      el.classList.add('source-flash');
      window.setTimeout(() => el.classList.remove('source-flash'), 1200);
    }
  };

  return (
    <div className="md">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          a: ({ children, href, ...props }) => {
            const h = normalizeCitationHref(String(href || ''));
            if (h.startsWith('#src:')) {
              const utteranceId = normalizeUtteranceId(h.slice('#src:'.length));
              const n =
                sourceIndex.get(utteranceId) ||
                sourceIndex.get(`utt_${utteranceId}`) ||
                sourceIndex.get(utteranceId.toLowerCase()) ||
                sourceIndex.get(`utt_${utteranceId.toLowerCase()}`);
              const label = String(n || 1);
              return (
                <sup>
                  <button
                    type="button"
                    onClick={() => onJumpToSource(utteranceId)}
                    className="ml-0.5 align-super text-[0.72rem] font-semibold text-ink/65 underline decoration-ink/25 underline-offset-2 hover:text-ink"
                    aria-label={n ? `Jump to source ${n}` : 'Jump to source'}
                  >
                    [{label}]
                  </button>
                </sup>
              );
            }

            return <a href={h} target="_blank" rel="noreferrer" {...props} />;
          },
        }}
      >
        {contentWithFallbackLinks}
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
                citationIds: extractCitationIds(m.metadata),
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
        citationIds: extractCitationIds(res.assistant_message.metadata),
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
                      <MarkdownMessage
                        content={m.content}
                        sources={m.sources}
                        citationIds={m.citationIds}
                        messageId={m.id}
                      />

                      {m.role === 'assistant' && m.sources && m.sources.length > 0 && (
                        <div className="mt-4 border-t border-ink/10 pt-3">
                          <div className="text-xs font-semibold uppercase tracking-wide text-ink/60">
                            Sources
                          </div>
                          <div className="mt-2 space-y-1.5">
                            {m.sources.map((s) => (
                              <a
                                key={s.utterance_id}
                                id={`src-${m.id}-${normalizeUtteranceId(s.utterance_id)}`}
                                href={s.youtube_url}
                                target="_blank"
                                rel="noreferrer"
                                className="source-card"
                              >
                                <div className="flex items-start gap-2">
                                  <div className="mt-0.5 flex h-6 w-6 shrink-0 items-center justify-center rounded bg-red-600 text-white">
                                    <YouTubeIcon className="h-3 w-3" />
                                  </div>

                                  <div className="min-w-0 flex-1">
                                    <div className="flex flex-wrap items-baseline gap-x-2 gap-y-0.5">
                                      <span className="text-xs font-semibold text-ink">
                                        {formatSpeakerName(s)}
                                      </span>
                                      {formatSpeakerTitle(s) && (
                                        <span className="text-[11px] text-ink/60">
                                          {formatSpeakerTitle(s)}
                                        </span>
                                      )}
                                    </div>

                                    <div className="flex flex-wrap items-center gap-x-2 gap-y-0.5 text-[11px] text-ink/60">
                                      {s.video_title && (
                                        <span className="source-title">{s.video_title}</span>
                                      )}
                                      {s.timestamp_str && (
                                        <span className="source-pill">@ {s.timestamp_str}</span>
                                      )}
                                    </div>

                                    <div className="mt-1 text-[11px] text-ink/80 source-snippet">
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
