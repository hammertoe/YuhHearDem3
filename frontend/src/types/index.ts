export interface Sentence {
  id: string;
  text: string;
  seconds_since_start: number;
  timestamp_str: string;
  video_id: string;
  speaker_id: string;
  speaker_name: string;
  video_date: string;
  video_title: string;
  paragraph_id: string;
  score?: number;
}

export interface SearchResult {
  id: string;
  text: string;
  timestamp_str: string;
  seconds_since_start: number;
  video_id: string;
  speaker_id: string;
  speaker_name: string;
  video_date: string;
  video_title: string;
  paragraph_id: string;
  score: number;
  search_type: string;
  provenance?: string;
}

export interface Speaker {
  speaker_id: string;
  normalized_name: string;
  full_name: string;
  title: string;
  position: string;
  role_in_video: string;
  first_appearance: string;
  total_appearances: number;
}

export interface Entity {
  entity_id: string;
  entity_type: string;
  entity_text: string;
  confidence?: number;
}

export interface GraphNode {
  id: string;
  label: string;
  type: string;
  properties: Record<string, any>;
}

export interface GraphEdge {
  from: string;
  to: string;
  label: string;
  properties: Record<string, any>;
}

export interface TrendData {
  date: string;
  mentions: number;
  value?: number;
}

export interface TrendResult {
  entity_id: string;
  trends: TrendData[];
  summary: {
    total_mentions: number;
    date_range: string;
    average_daily: number;
    peak_date: string;
    peak_mentions: number;
  };
  moving_average: TrendData[];
}

export interface ThreadMessage {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  metadata: {
    citations?: string[];
    focus_node_ids?: string[];
    retrieval_stats?: {
      candidates: number;
      edges: number;
      sentences: number;
    };
  } | null;
  created_at: string;
}

export interface Thread {
  id: string;
  title: string | null;
  created_at: string;
  updated_at: string;
  state: {
    focus_node_ids?: string[];
    focus_node_labels?: string[];
    last_question?: string;
  };
  messages: ThreadMessage[];
}

export interface CreateThreadResponse {
  thread_id: string;
  title: string | null;
  created_at: string;
}

export interface ChatCitation {
  utterance_id: string;
  youtube_video_id: string;
  seconds_since_start: number;
  timestamp_str: string;
  speaker_id: string;
  speaker_name: string;
  text: string;
  video_title: string | null;
  video_date: string | null;
}

export interface ChatFocusNode {
  id: string;
  label: string;
  type: string;
}

export interface ChatUsedEdge {
  id: string;
  source_id: string;
  predicate: string;
  predicate_raw: string | null;
  target_id: string;
  confidence: number | null;
  evidence: string | null;
  utterance_ids: string[];
}

export interface ChatMessageResponse {
  thread_id: string;
  assistant_message: ThreadMessage;
  citations: ChatCitation[];
  focus_nodes: ChatFocusNode[];
  used_edges: ChatUsedEdge[];
  debug: {
    planner?: {
      intent: string;
      entities: string[];
      predicates: string[];
      node_types: string[];
      followup_requires_focus: boolean;
    };
    retrieval?: {
      candidates: number;
      edges: number;
      sentences: number;
      fallback_used: boolean;
    };
  } | null;
}
