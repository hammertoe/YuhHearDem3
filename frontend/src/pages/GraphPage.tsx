import { useState, useEffect, useRef } from 'react';
import { Network } from 'vis-network/standalone';
import { searchApi } from '../services/api';
import type { GraphNode, GraphEdge } from '../types';

function GraphPage() {
  const [selectedEntity, setSelectedEntity] = useState<string>('');
  const [nodes, setNodes] = useState<GraphNode[]>([]);
  const [edges, setEdges] = useState<GraphEdge[]>([]);
  const [loading, setLoading] = useState(false);
  const networkRef = useRef<Network | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (nodes.length > 0 && edges.length > 0 && containerRef.current) {
      const visNodes = nodes.map(node => ({
        id: node.id,
        label: node.label,
        group: node.type,
        title: JSON.stringify(node.properties, null, 2),
      }));

      const visEdges = edges.map(edge => ({
        from: edge.from,
        to: edge.to,
        label: edge.label,
        title: JSON.stringify(edge.properties, null, 2),
      }));

      const data = { nodes: visNodes, edges: visEdges };
      const options = {
        nodes: {
          shape: 'dot',
          size: 16,
          font: { size: 12 },
        },
        edges: {
          width: 2,
          smooth: { enabled: true, type: 'dynamic', roundness: 0.5 },
        },
        groups: {
          Speaker: { color: { background: '#4ade80', border: '#22c55e' } },
          Bill: { color: { background: '#f87171', border: '#ef4444' } },
          Topic: { color: { background: '#60a5fa', border: '#3b82f6' } },
          Organization: { color: { background: '#fbbf24', border: '#f59e0b' } },
          Person: { color: { background: '#a78bfa', border: '#8b5cf6' } },
          Location: { color: { background: '#fb923c', border: '#f97316' } },
        },
        physics: {
          stabilization: true,
          barnesHut: {
            gravitationalConstant: -8000,
            springConstant: 0.04,
            springLength: 95,
          },
        },
      };

      networkRef.current = new Network(containerRef.current, data, options);
    }

    return () => {
      if (networkRef.current) {
        networkRef.current.destroy();
        networkRef.current = null;
      }
    };
  }, [nodes, edges]);

  const loadGraph = async () => {
    if (!selectedEntity) return;

    setLoading(true);
    try {
      const data = await searchApi.getGraph(selectedEntity, 2);
      setNodes(data.nodes);
      setEdges(data.edges);
    } catch (error) {
      console.error('Failed to load graph:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="bg-white rounded-lg shadow p-6">
        <h2 className="text-2xl font-bold text-gray-900 mb-4">Entity Graph</h2>
        
        <div className="flex gap-4">
          <input
            type="text"
            value={selectedEntity}
            onChange={(e) => setSelectedEntity(e.target.value)}
            placeholder="Enter entity ID (e.g., s_speaker_1, L_BILL_123)"
            className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
          />
          <button
            onClick={loadGraph}
            disabled={loading || !selectedEntity}
            className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-400"
          >
            {loading ? 'Loading...' : 'Load Graph'}
          </button>
        </div>

        <div className="mt-4 text-sm text-gray-600">
          <p>Enter an entity ID to visualize its connections (max 2 hops).</p>
        </div>
      </div>

      {nodes.length > 0 && (
        <div className="bg-white rounded-lg shadow p-6">
          <div className="mb-4">
            <h3 className="text-lg font-semibold text-gray-900">
              Graph Visualization
            </h3>
            <p className="text-sm text-gray-600">
              {nodes.length} nodes • {edges.length} edges
            </p>
          </div>

          <div
            ref={containerRef}
            className="w-full h-96 bg-gray-50 rounded-lg border border-gray-200"
          />

          <div className="mt-4 grid grid-cols-3 md:grid-cols-6 gap-2">
            {['Speaker', 'Bill', 'Topic', 'Organization', 'Person', 'Location'].map((type) => (
              <div
                key={type}
                className="flex items-center gap-2 text-sm"
              >
                <div
                  className="w-4 h-4 rounded-full"
                  style={{
                    backgroundColor: {
                      Speaker: '#4ade80',
                      Bill: '#f87171',
                      Topic: '#60a5fa',
                      Organization: '#fbbf24',
                      Person: '#a78bfa',
                      Location: '#fb923c',
                    }[type]
                  }}
                />
                <span>{type}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export default GraphPage;
