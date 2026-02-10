# Parliamentary Search System - Frontend

React + TypeScript frontend for the parliamentary search system.

## Features

- **Hybrid Search**: Vector + BM25 + Graph search with re-ranking
- **Temporal Search**: Filter by date range, speaker, entity type
- **Video Player**: Embedded YouTube player with timestamp navigation
- **Graph Visualization**: Interactive graph of entities and relationships
- **Speaker Profiles**: View speaker statistics and contributions
- **Responsive Design**: Mobile-friendly interface

## Tech Stack

- **React 18** - UI library
- **TypeScript 5.4** - Type safety
- **Vite 5.2** - Build tool and dev server
- **Tailwind CSS 3.4** - Styling
- **vis-network 9.1** - Graph visualization
- **react-player 2.16** - YouTube video player
- **axios 1.6** - HTTP client

## Getting Started

\`\`\`bash
cd frontend
npm install
npm run dev
\`\`\`

The app will be available at \`http://localhost:3000\`

## Project Structure

\`\`\`
src/
├── components/       # Reusable UI components
│   ├── SearchFilters.tsx
│   ├── SearchResultItem.tsx
│   └── VideoPlayer.tsx
├── pages/           # Main pages
│   ├── SearchPage.tsx
│   ├── SpeakerPage.tsx
│   └── GraphPage.tsx
├── services/        # API communication
│   └── api.ts
├── types/           # TypeScript type definitions
│   └── index.ts
├── utils/           # Helper functions
├── App.tsx          # Root component
└── main.tsx         # Entry point
\`\`\`

## API Endpoints

The frontend expects the following backend API endpoints:

- \`POST /api/search\` - Hybrid search (vector + graph + re-ranking)
- \`POST /api/search/temporal\` - Temporal search with filters
- \`GET /api/search/trends\` - Trend analysis for entities
- \`GET /api/graph\` - Graph data for visualization
- \`GET /api/speakers\` - List all speakers
- \`GET /api/speakers/:id\` - Get speaker stats

## Features

### Search Page
- Free-text search with filters
- Date range filtering
- Speaker filtering
- Entity type filtering
- Click result to open video player at timestamp

### Video Player
- Embedded YouTube player
- Jump to specific timestamp
- Close button to return to results
- Display jump confirmation

### Graph Visualization
- Interactive graph with drag/zoom
- Color-coded by entity type
- Click nodes to see details
- Max 2 hops traversal

### Speaker Profiles
- List all speakers with stats
- Click speaker to see details
- Recent contributions
- Total appearances
- First seen date
- Role and position information

## Design Patterns

- **Component Composition**: Reusable components with props interfaces
- **State Management**: React hooks (useState, useEffect)
- **API Communication**: Axios with TypeScript types
- **Error Handling**: Try-catch with user feedback
- **Loading States**: Skeleton screens and loading indicators
- **Responsive Design**: Tailwind responsive utilities
