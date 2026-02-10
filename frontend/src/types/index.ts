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
