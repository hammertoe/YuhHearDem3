import type { TemporalSearchParams } from '../services/api';

interface SearchFiltersProps {
  filters: Partial<TemporalSearchParams>;
  onFiltersChange: (filters: Partial<TemporalSearchParams>) => void;
}

function SearchFilters({ filters, onFiltersChange }: SearchFiltersProps) {
  return (
    <div className="grid grid-cols-1 md:grid-cols-4 gap-4 p-4 bg-gray-50 rounded-lg">
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-1">
          Start Date
        </label>
        <input
          type="date"
          value={filters.start_date || ''}
          onChange={(e) => onFiltersChange({ ...filters, start_date: e.target.value })}
          className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm"
        />
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700 mb-1">
          End Date
        </label>
        <input
          type="date"
          value={filters.end_date || ''}
          onChange={(e) => onFiltersChange({ ...filters, end_date: e.target.value })}
          className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm"
        />
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700 mb-1">
          Speaker
        </label>
        <input
          type="text"
          value={filters.speaker_id || ''}
          onChange={(e) => onFiltersChange({ ...filters, speaker_id: e.target.value })}
          placeholder="Speaker ID"
          className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm"
        />
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700 mb-1">
          Entity Type
        </label>
        <select
          value={filters.entity_type || ''}
          onChange={(e) => onFiltersChange({ ...filters, entity_type: e.target.value })}
          className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm"
        >
          <option value="">All Types</option>
          <option value="ORG">Organization</option>
          <option value="PERSON">Person</option>
          <option value="TOPIC">Topic</option>
          <option value="LOCATION">Location</option>
        </select>
      </div>
    </div>
  );
}

export default SearchFilters;
