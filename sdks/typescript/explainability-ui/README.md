# @your-company/explainability-ui

Lightweight TypeScript SDK and React Hypergraph viewer for rendering HIF v1 (legacy) hypergraphs exposed by the Gateway.

CORS note
- If your Gateway is running remotely and does not enable CORS, browsers will block requests from your local app. Enable CORS in the Gateway so Access-Control-Allow-Origin includes your dev origin (e.g., http://localhost:5173) or use a permissive local setting during development.

What it provides
- A small HTTP client for:
  - GET /v1/traces/{trace_id}/graph
  - GET /v1/traces/{trace_id}/status
- A React component HypergraphViewer with:
  - Client-side filtering by min edge weight
  - Optional supernode grouping for SAE features, grouping by label prefix up to ":" (e.g., "Biology: ...")
  - Simple force-directed layout with D3
  - Polling behavior when a graph is not ready (404), and an expired (410) state

Install
- Install the package (plus React peer deps):
  - npm i @your-company/explainability-ui
  - npm i react react-dom

- Optional: TypeScript types come baked in. No additional configuration is required beyond standard TS settings.

Quick start: React usage
```tsx
import React from 'react';
import { HypergraphViewer } from '@your-company/explainability-ui';

export default function ChatInterface() {
  return (
    <div style={{ height: 480 }}>
      <HypergraphViewer
        traceId="trc_abc123"
        baseUrl="http://localhost:8080"
        minEdgeWeight={0.5}
        grouping="supernode"
        height={420}
      />
    </div>
  );
}
```

Behavior overview
- Loading and polling:
  - On mount, the component fetches /v1/traces/{traceId}/graph.
  - If the API responds 404 (graph not ready), the viewer switches to a "polling" state and retries every refreshMs (default 1000 ms).
  - If the API responds 410 (expired), the viewer switches to an "expired" state and stops polling.
  - Any other HTTP error shows a small error banner.
- Client-side pruning:
  - Incidences (hyperedges) with weight < minEdgeWeight are dropped.
- Grouping:
  - grouping="supernode" (default) compresses multiple sae_feature nodes with similar labels into a synthetic circuit_supernode for visualization.
  - The heuristic groups by the label prefix (text before the first colon ":"). Groups with size >= 2 become circuit_supernode nodes.
  - Clicking a supernode shows a small panel with member count.
- Rendering:
  - A simple force-directed layout using D3. Hyperedges are expanded to pairwise links by connecting each feature node (original or grouped) to each token node in the same incidence.
  - Colors per node type, and tooltips (title) for nodes and edges.

Props
- traceId: string — the trace ID to fetch
- baseUrl: string — Gateway base URL, e.g., http://localhost:8080
- apiKey?: string — optional bearer token (Authorization: Bearer ...)
- minEdgeWeight?: number — default 0.01; slider is provided in the UI
- grouping?: 'none' | 'supernode' — default 'supernode'
- height?: number — default 420
- width?: number | '100%' — default '100%' responsive to container width
- refreshMs?: number — default 1000; polling interval when graph is not ready

SDK: HTTP client usage
```ts
import { ExplainabilityClient } from '@your-company/explainability-ui';

async function demo() {
  const client = new ExplainabilityClient({
    baseUrl: 'http://localhost:8080',
    // apiKey: 'sk-...' // optional
  });

  try {
    // Fetch hypergraph for a trace
    const graph = await client.getGraph('trc_abc123');
    console.log('nodes', graph.nodes.length, 'incidences', graph.incidences.length);

    // Optionally query status
    const status = await client.getStatus('trc_abc123');
    console.log('state', status.state, 'progress', status.progress);
  } catch (err) {
    console.error('Explainability API error:', err);
  }
}

demo();
```

Types
- Compatible with HIF v1 (legacy) shapes used by the Gateway:
  - Node types: 'sae_feature' | 'input_token' | 'output_token' | 'circuit_supernode'
  - HIFGraph: { 'network-type': 'directed', nodes, incidences, meta? }
  - A lightweight type guard isHIFGraph(o) is available for defensive checks

Build and local development
- Build library:
  - npm run build
- Type-check:
  - npm run typecheck
- The package ships as ESM with .d.ts type declarations.

Notes
- Hyperedges are visualized as pairwise links between feature nodes (sae_feature or supernodes) and token nodes (input_token/output_token).
- Tooltips show:
  - Nodes: label, type, layer, activation_strength (+ member count for supernodes)
  - Edges: aggregated weight and method(s) (if present in metadata)
- If the Gateway returns a wrapper object (ExplanationResponse), the SDK attempts to unwrap data.hypergraph.

Troubleshooting
- If you receive CORS errors in your browser console, verify that your Gateway returns appropriate CORS headers for your frontend origin.
- If you see 404 during polling and it never resolves, ensure the trace_id exists and the explainer produced a graph.
- If you see 410 (expired), the viewer will stop polling and display an "expired" state.

License
- Apache-2.0