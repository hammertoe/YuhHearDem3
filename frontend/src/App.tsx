import { useState } from 'react';
import SearchPage from './pages/SearchPage';
import SpeakerPage from './pages/SpeakerPage';
import GraphPage from './pages/GraphPage';
import ChatPage from './pages/ChatPage';

function App() {
  const [currentPage, setCurrentPage] = useState<'search' | 'speakers' | 'graph' | 'chat'>('search');

  return (
    <div className="min-h-screen bg-gray-50">
      <nav className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between h-16">
            <div className="flex items-center">
              <h1 className="text-xl font-bold text-gray-900">
                Parliamentary Search System
              </h1>
            </div>
             <div className="flex items-center space-x-4">
               <button
                 onClick={() => setCurrentPage('search')}
                 className={`px-3 py-2 rounded-md text-sm font-medium ${
                   currentPage === 'search'
                     ? 'bg-blue-100 text-blue-700'
                     : 'text-gray-700 hover:bg-gray-100'
                 }`}
               >
                 Search
               </button>
               <button
                 onClick={() => setCurrentPage('speakers')}
                 className={`px-3 py-2 rounded-md text-sm font-medium ${
                   currentPage === 'speakers'
                     ? 'bg-blue-100 text-blue-700'
                     : 'text-gray-700 hover:bg-gray-100'
                 }`}
               >
                 Speakers
               </button>
               <button
                 onClick={() => setCurrentPage('graph')}
                 className={`px-3 py-2 rounded-md text-sm font-medium ${
                   currentPage === 'graph'
                     ? 'bg-blue-100 text-blue-700'
                     : 'text-gray-700 hover:bg-gray-100'
                 }`}
               >
                 Graph
               </button>
               <button
                 onClick={() => setCurrentPage('chat')}
                 className={`px-3 py-2 rounded-md text-sm font-medium ${
                   currentPage === 'chat'
                     ? 'bg-blue-100 text-blue-700'
                     : 'text-gray-700 hover:bg-gray-100'
                 }`}
               >
                 Chat
               </button>
             </div>
          </div>
        </div>
      </nav>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {currentPage === 'search' && <SearchPage />}
        {currentPage === 'speakers' && <SpeakerPage />}
        {currentPage === 'graph' && <GraphPage />}
        {currentPage === 'chat' && <ChatPage />}
      </main>
    </div>
  );
}

export default App;
