# Parliamentary Search System - Frontend

React + TypeScript frontend for the parliamentary search system.

## Features

- **Streaming Chat**: SSE progress updates during response generation
- **Citations**: Source cards with timestamps and speaker attribution
- **Bill Excerpts**: Inline excerpts when grounded in bill text
- **Follow-up Prompts**: Suggested next questions after each response
- **Responsive Design**: Mobile-friendly layout

## Tech Stack

- **React 18** - UI library
- **TypeScript 5.4** - Type safety
- **Vite 5.2** - Build tool and dev server
- **Tailwind CSS 3.4** - Styling
- **react-markdown 10** - Markdown rendering
- **remark-gfm 4** - GitHub-flavored markdown

## Getting Started

```bash
cd frontend
npm install
npm run dev
```

The app will be available at `http://localhost:3000`. During development, `/chat` requests are
proxied to the backend at `http://localhost:8000` via `vite.config.ts`.

## Project Structure

```
src/
├── App.tsx           # Root component
├── api.ts            # Chat API helpers
├── index.css         # Tailwind styles
├── main.tsx          # Entry point
├── sourceGrouping.ts # Group citations by document
└── timeFormat.ts     # Timestamp formatting helpers
```

## API Endpoints

The frontend currently calls these backend endpoints:

- `POST /chat/threads` - Create a thread
- `POST /chat/threads/:id/messages` - Send message to a thread
- `GET /chat/threads/:id/messages/stream` - Stream response via SSE

Note: the UI attempts to restore a saved thread by calling `GET /chat/threads/:id`, but the
backend does not currently expose that endpoint.

## Features

### Chat Experience

- Prompt suggestions and examples
- Streaming progress stages
- Markdown rendering with citations
- Source grouping by document

## Design Patterns

- **Component Composition**: Reusable components with props interfaces
- **State Management**: React hooks (useState, useEffect)
- **API Communication**: Fetch + TypeScript types
- **Error Handling**: Try-catch with user feedback
- **Loading States**: Skeleton screens and loading indicators
- **Responsive Design**: Tailwind responsive utilities
