import type { SearchResult } from '../types';

interface SearchResultItemProps {
  result: SearchResult;
  onClick: () => void;
}

function SearchResultItem({ result, onClick }: SearchResultItemProps) {
  return (
    <div
      onClick={onClick}
      className="bg-white rounded-lg shadow p-4 hover:shadow-md transition-shadow cursor-pointer border-l-4 border-blue-500"
    >
      <div className="flex items-start justify-between mb-2">
        <div>
          <h4 className="font-semibold text-gray-900">{result.video_title}</h4>
          <p className="text-sm text-gray-600">
            {result.speaker_name} • {result.timestamp_str}
          </p>
        </div>
        <div className="text-right">
          <span className="inline-block px-2 py-1 text-xs font-medium bg-blue-100 text-blue-700 rounded">
            Score: {result.score.toFixed(3)}
          </span>
          <p className="text-xs text-gray-500 mt-1">
            {result.search_type}
          </p>
        </div>
      </div>
      <p className="text-gray-700 mt-2">{result.text}</p>
      {result.provenance && (
        <p className="text-xs text-gray-500 mt-2 italic">
          {result.provenance}
        </p>
      )}
    </div>
  );
}

export default SearchResultItem;
