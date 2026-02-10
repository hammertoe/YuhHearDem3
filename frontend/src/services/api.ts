import axios from 'axios';
import type { SearchResult, TrendResult, GraphNode, GraphEdge, Speaker } from '../types';

const api = axios.create({
  baseURL: '/api',
  timeout: 30000,
});

export interface SearchParams {
  query: string;
  limit?: number;
  alpha?: number;
}

export interface TemporalSearchParams extends SearchParams {
  start_date?: string;
  end_date?: string;
  speaker_id?: string;
  entity_type?: string;
}

export interface TrendParams {
  entity_id: string;
  days?: number;
  window_size?: number;
}

export const searchApi = {
  async search(params: SearchParams): Promise<SearchResult[]> {
    const response = await api.post<SearchResult[]>('/search', params);
    return response.data;
  },

  async temporalSearch(params: TemporalSearchParams): Promise<SearchResult[]> {
    const response = await api.post<SearchResult[]>('/search/temporal', params);
    return response.data;
  },

  async getTrends(params: TrendParams): Promise<TrendResult> {
    const response = await api.get<TrendResult>('/search/trends', { params });
    return response.data;
  },

  async getGraph(entityId: string, maxDepth: number = 2): Promise<{ nodes: GraphNode[], edges: GraphEdge[] }> {
    const response = await api.get<{ nodes: GraphNode[], edges: GraphEdge[] }>('/graph', {
      params: { entity_id: entityId, max_depth: maxDepth }
    });
    return response.data;
  },

  async getSpeakers(): Promise<Speaker[]> {
    const response = await api.get<Speaker[]>('/speakers');
    return response.data;
  },

  async getSpeakerStats(speakerId: string): Promise<Speaker & { recent_contributions: SearchResult[] }> {
    const response = await api.get<Speaker & { recent_contributions: SearchResult[] }>(`/speakers/${speakerId}`);
    return response.data;
  }
};

export default api;
