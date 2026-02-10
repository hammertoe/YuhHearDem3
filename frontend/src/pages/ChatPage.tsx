import { useState, useEffect, useRef } from 'react';
import { chatApi, searchApi } from '../services/api';
import type { Thread, ChatMessageResponse, ChatCitation, ChatFocusNode } from '../types';
import VideoPlayer from '../components/VideoPlayer';

function ChatPage() {
  const [threadId, setThreadId] = useState<string | null>(null);
  const [thread, setThread] = useState<Thread | null>(null);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [response, setResponse] = useState<ChatMessageResponse | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const savedThreadId = localStorage.getItem('chat_thread_id');
    if (savedThreadId) {
      loadThread(savedThreadId);
    } else {
      startNewThread();
    }
  }, []);

  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [thread?.messages, response]);

  const startNewThread = async () => {
    try {
      const result = await chatApi.createThread(null);
      setThreadId(result.thread_id);
      localStorage.setItem('chat_thread_id', result.thread_id);
      loadThread(result.thread_id);
    } catch (error) {
      console.error('Failed to create thread:', error);
    }
  };

  const loadThread = async (id: string) => {
    try {
      const data = await chatApi.getThread(id);
      setThreadId(id);
      setThread(data);
      setResponse(null);
    } catch (error) {
      console.error('Failed to load thread:', error);
    }
  };

  const handleSend = async () => {
    if (!input.trim() || !threadId || loading) return;

    setLoading(true);
    setResponse(null);

    try {
      const result = await chatApi.sendMessage(threadId, input.trim());
      setResponse(result);

      if (threadId) {
        const updatedThread = await chatApi.getThread(threadId);
        setThread(updatedThread);
      }

      setInput('');
    } catch (error) {
      console.error('Failed to send message:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleCitationClick = (citation: ChatCitation) => {
    if (response?.debug?.retrieval) {
      response.debug.retrieval.fallback_used
    }
  };

  const handleNodeClick = (node: ChatFocusNode) => {
    if (!threadId) return;
    searchApi.getGraph(node.id, 2).then(data => {
      console.log('Loaded graph for node:', node.id, data);
    });
  };

  return (
    <div className="space-y-6">
      <div className="bg-white rounded-lg shadow p-6">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-2xl font-bold text-gray-900">Chat with Knowledge Graph</h2>
          <button
            onClick={startNewThread}
            className="px-4 py-2 text-sm bg-gray-100 hover:bg-gray-200 rounded"
          >
            New Thread
          </button>
        </div>

        <div className="flex gap-4 mb-4">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyPress}
            placeholder="Ask about funding, bills, speakers..."
            disabled={loading}
            className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 disabled:bg-gray-50"
          />
          <button
            onClick={handleSend}
            disabled={loading || !input.trim()}
            className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-400"
          >
            {loading ? 'Sending...' : 'Send'}
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">
            Conversation
          </h3>

          <div className="space-y-4 max-h-96 overflow-y-auto">
            {thread?.messages.map((msg) => (
              <div
                key={msg.id}
                className={`p-4 rounded ${
                  msg.role === 'user'
                    ? 'bg-blue-50 ml-8'
                    : 'bg-gray-50 mr-8'
                }`}
              >
                <div className="text-sm text-gray-500 mb-1">
                  {msg.role === 'user' ? 'You' : 'Assistant'}
                </div>
                <div className="text-gray-700">{msg.content}</div>

                {msg.role === 'assistant' && msg.metadata?.citations && (
                  <div className="mt-2 pt-2 border-t border-gray-200">
                    <div className="text-xs text-gray-500 mb-1">Sources:</div>
                    {msg.metadata.citations.slice(0, 3).map((citation: string, idx: number) => (
                      <div
                        key={idx}
                        className="text-xs bg-gray-100 px-2 py-1 rounded mt-1"
                      >
                        {citation}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            ))}

            {response && (
              <div className="p-4 bg-green-50 rounded mr-8">
                <div className="text-sm text-gray-500 mb-1">Assistant</div>
                <div className="text-gray-700">{response.assistant_message.content}</div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>

          {!thread && !response && (
            <div className="text-center py-8 text-gray-500">
              No messages yet. Start a conversation above.
            </div>
          )}
        </div>

        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">
            Sources
            {response?.citations && ` (${response.citations.length})`}
          </h3>

          {response?.citations && response.citations.length > 0 ? (
            <div className="space-y-3 max-h-96 overflow-y-auto">
              {response.citations.map((citation, idx) => (
                <div
                  key={idx}
                  className="p-3 bg-gray-50 rounded border border-gray-200 cursor-pointer hover:bg-gray-100"
                  onClick={() => handleCitationClick(citation)}
                >
                  <div className="flex justify-between items-start mb-2">
                    <span className="font-medium text-gray-900">
                      {citation.speaker_name}
                    </span>
                    <span className="text-xs text-gray-500">
                      {citation.timestamp_str}
                    </span>
                  </div>

                  <div className="text-sm text-gray-700 mb-2">{citation.text}</div>

                  <div className="text-xs text-gray-500">
                    {citation.video_title}
                    {citation.video_date && ` • ${citation.video_date}`}
                  </div>

                  <div className="mt-2 text-xs text-blue-600 font-mono">
                    Utterance ID: {citation.utterance_id}
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-8 text-gray-500">
              {response
                ? 'No citations available'
                : 'Ask a question to see sources'}
            </div>
          )}
        </div>

        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">
            Focus Concepts
            {response?.focus_nodes && ` (${response.focus_nodes.length})`}
          </h3>

          {response?.focus_nodes && response.focus_nodes.length > 0 ? (
            <div className="space-y-2 max-h-96 overflow-y-auto">
              {response.focus_nodes.map((node, idx) => (
                <div
                  key={idx}
                  className="p-3 bg-purple-50 rounded border border-purple-200 cursor-pointer hover:bg-purple-100"
                  onClick={() => handleNodeClick(node)}
                >
                  <div className="text-sm text-gray-500 mb-1">{node.type}</div>
                  <div className="font-medium text-gray-900">{node.label}</div>
                  <div className="mt-1 text-xs text-purple-700 font-mono">
                    ID: {node.id}
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-8 text-gray-500">
              {response
                ? 'No focus concepts identified'
                : 'Ask a question to identify concepts'}
            </div>
          )}

          {response?.debug?.retrieval && (
            <div className="mt-4 pt-4 border-t border-gray-200">
              <div className="text-xs text-gray-500 mb-1">Retrieval Stats:</div>
              <div className="grid grid-cols-2 gap-2 text-xs">
                <div>Candidates: {response.debug.retrieval.candidates}</div>
                <div>Edges: {response.debug.retrieval.edges}</div>
                <div>Sentences: {response.debug.retrieval.sentences}</div>
                <div>
                  Fallback: {response.debug.retrieval.fallback_used ? 'Yes' : 'No'}
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default ChatPage;
