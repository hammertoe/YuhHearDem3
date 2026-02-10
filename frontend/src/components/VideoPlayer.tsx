import { useRef, useEffect } from 'react';
import ReactPlayer from 'react-player';

interface VideoPlayerProps {
  videoId: string;
  initialTimestamp: number | null;
  onClose: () => void;
}

function VideoPlayer({ videoId, initialTimestamp, onClose }: VideoPlayerProps) {
  const playerRef = useRef<ReactPlayer>(null);

  useEffect(() => {
    if (initialTimestamp !== null && playerRef.current) {
      playerRef.current.seekTo(initialTimestamp, 'seconds');
    }
  }, [initialTimestamp]);

  return (
    <div className="bg-white rounded-lg shadow p-6">
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-lg font-semibold text-gray-900">Video Player</h3>
        <button
          onClick={onClose}
          className="px-4 py-2 text-gray-600 hover:text-gray-900"
        >
          ✕
        </button>
      </div>
      
      <div className="aspect-video bg-gray-900 rounded-lg overflow-hidden">
        <ReactPlayer
          ref={playerRef}
          url={`https://www.youtube.com/watch?v=${videoId}`}
          width="100%"
          height="100%"
          controls
          playing
        />
      </div>

      {initialTimestamp !== null && (
        <div className="mt-4 p-4 bg-blue-50 rounded-lg">
          <p className="text-sm text-blue-800">
            Jumped to {Math.floor(initialTimestamp / 60)}:{(initialTimestamp % 60).toString().padStart(2, '0')}
          </p>
        </div>
      )}
    </div>
  );
}

export default VideoPlayer;
