import { useState } from 'react';
import { searchApi, TemporalSearchParams } from '../services/api';
import type { SearchResult } from '../types';
import VideoPlayer from '../components/VideoPlayer';
import SearchResultItem from '../components/SearchResultItem';
import SearchFilters from '../components/SearchFilters';

function SearchPage() {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<SearchResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [selectedVideo, setSelectedVideo] = useState<string | null>(null);
  const [selectedTimestamp, setSelectedTimestamp] = useState<number | null>(null);
  const [filters, setFilters] = useState<Partial<TemporalSearchParams>>({});

  const handleSearch = async () => {
    if (!query.trim()) return;

    setLoading(true);
    try {
      const searchResults = await searchApi.temporalSearch({
        query,
        ...filters,
      });
      setResults(searchResults);
    } catch (error) {
      console.error('Search failed:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleResultClick = (result: SearchResult) => {
    setSelectedVideo(result.video_id);
    setSelectedTimestamp(result.seconds_since_start);
  };

  return (
    <div className="space-y-6">
      <div className="bg-white rounded-lg shadow p-6">
        <h2 className="text-2xl font-bold text-gray-900 mb-4">Search Parliamentary Discussions</h2>
        
        <div className="flex gap-4 mb-4">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
            placeholder="Enter search query..."
            className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
          />
          <button
            onClick={handleSearch}
            disabled={loading}
            className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-400"
          >
            {loading ? 'Searching...' : 'Search'}
          </button>
        </div>

        <SearchFilters filters={filters} onFiltersChange={setFilters} />
      </div>

      {selectedVideo && (
        <VideoPlayer
          videoId={selectedVideo}
          initialTimestamp={selectedTimestamp}
          onClose={() => {
            setSelectedVideo(null);
            setSelectedTimestamp(null);
          }}
        />
      )}

      <div className="space-y-4">
        <h3 className="text-lg font-semibold text-gray-900">
          Results ({results.length})
        </h3>
        {results.length === 0 && !loading && query && (
          <div className="text-center py-8 text-gray-500">
            No results found for "{query}"
          </div>
        )}
        {results.map((result) => (
          <SearchResultItem
            key={result.id}
            result={result}
            onClick={() => handleResultClick(result)}
          />
        ))}
      </div>
    </div>
  );
}

export default SearchPage;
