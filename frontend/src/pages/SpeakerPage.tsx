import { useState, useEffect } from 'react';
import { searchApi } from '../services/api';
import type { Speaker, SearchResult } from '../types';

type SpeakerStats = Speaker & { recent_contributions: SearchResult[] };

function SpeakerPage() {
  const [speakers, setSpeakers] = useState<Speaker[]>([]);
  const [selectedSpeaker, setSelectedSpeaker] = useState<SpeakerStats | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadSpeakers();
  }, []);

  const loadSpeakers = async () => {
    setLoading(true);
    try {
      const data = await searchApi.getSpeakers();
      setSpeakers(data);
    } catch (error) {
      console.error('Failed to load speakers:', error);
    } finally {
      setLoading(false);
    }
  };

  const loadSpeakerStats = async (speakerId: string) => {
    try {
      const stats = await searchApi.getSpeakerStats(speakerId);
      setSelectedSpeaker(stats as SpeakerStats);
    } catch (error) {
      console.error('Failed to load speaker stats:', error);
    }
  };

  return (
    <div className="space-y-6">
      <div className="bg-white rounded-lg shadow p-6">
        <h2 className="text-2xl font-bold text-gray-900 mb-4">Speakers</h2>
        
        {loading ? (
          <div className="text-center py-8">Loading speakers...</div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {speakers.map((speaker) => (
              <div
                key={speaker.speaker_id}
                onClick={() => loadSpeakerStats(speaker.speaker_id)}
                className="p-4 bg-gray-50 rounded-lg hover:bg-gray-100 cursor-pointer transition-colors"
              >
                <h3 className="font-semibold text-gray-900">{speaker.full_name}</h3>
                <p className="text-sm text-gray-600 mt-1">
                  {speaker.position} • {speaker.role_in_video}
                </p>
                <div className="mt-2 text-xs text-gray-500">
                  <p>Appearances: {speaker.total_appearances}</p>
                  <p>First seen: {speaker.first_appearance}</p>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {selectedSpeaker && (
        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex justify-between items-start mb-4">
            <div>
              <h3 className="text-xl font-bold text-gray-900">{selectedSpeaker.full_name}</h3>
              <p className="text-gray-600">
                {selectedSpeaker.position} • {selectedSpeaker.role_in_video}
              </p>
            </div>
            <button
              onClick={() => setSelectedSpeaker(null)}
              className="px-4 py-2 text-gray-600 hover:text-gray-900"
            >
              ✕
            </button>
          </div>

          <div className="grid grid-cols-3 gap-4 mb-6">
            <div className="p-3 bg-blue-50 rounded">
              <p className="text-sm text-blue-600 font-medium">Total Appearances</p>
              <p className="text-2xl font-bold text-blue-900">
                {selectedSpeaker.total_appearances}
              </p>
            </div>
            <div className="p-3 bg-green-50 rounded">
              <p className="text-sm text-green-600 font-medium">First Seen</p>
              <p className="text-lg font-bold text-green-900">
                {selectedSpeaker.first_appearance}
              </p>
            </div>
            <div className="p-3 bg-purple-50 rounded">
              <p className="text-sm text-purple-600 font-medium">Role</p>
              <p className="text-lg font-bold text-purple-900">
                {selectedSpeaker.role_in_video}
              </p>
            </div>
          </div>

          <h4 className="font-semibold text-gray-900 mb-3">Recent Contributions</h4>
          <div className="space-y-3">
            {selectedSpeaker.recent_contributions?.slice(0, 5).map((contribution: SearchResult) => (
              <div
                key={contribution.id}
                className="p-3 bg-gray-50 rounded border-l-4 border-purple-500"
              >
                <p className="text-sm text-gray-600">
                  {contribution.video_title} • {contribution.timestamp_str}
                </p>
                <p className="text-gray-700 mt-1">{contribution.text}</p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export default SpeakerPage;
