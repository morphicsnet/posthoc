# TypeScript/React Quickstart: Fetch and Render HIF

This quickstart shows how to:
- Fetch a HIF graph for a trace using the ExplainabilityClient
- Render a graph using the HypergraphViewer React component
- Address CORS for browser-based development

References:
- Client and types: [ExplainabilityClient](sdks/typescript/explainability-ui/src/client.ts:19), [types](sdks/typescript/explainability-ui/src/types.ts:1)
- React viewer: [HypergraphViewer](sdks/typescript/explainability-ui/src/HypergraphViewer.tsx:233)
- API endpoints and headers: [docs/api-reference.md](docs/api-reference.md:1)
- HIF Schema (v1): [docs/hif-schema.md](docs/hif-schema.md:1)

Base URL (local): http://localhost:8080

## Prerequisites

- Node.js 18+
- A local Gateway running at http://localhost:8080, with the Async Sidecar stack
- A trace_id you can poll (create one by calling POST /v1/chat/completions with x-explain-mode: hypergraph)

## Install the UI/SDK package

Option A (install from local folder path while developing in this mono-repo):
```bash
# From your app directory (created via Vite or similar)
npm i ../../sdks/typescript/explainability-ui
# or with pnpm
pnpm add ../../sdks/typescript/explainability-ui
```

Option B (published package name; if available in your registry):
```bash
npm i @your-company/explainability-ui
# or pnpm add @your-company/explainability-ui
```

## Fetch a graph programmatically (TypeScript)

```ts
// src/fetch-graph.ts
import { ExplainabilityClient, type HIFGraph } from '@your-company/explainability-ui';

async function main() {
  const baseUrl = 'http://localhost:8080';
  const traceId = 'trc_abc123'; // replace with a real trace id

  const client = new ExplainabilityClient({
    baseUrl,
    // apiKey: 'sk-...'           // optional
    // headers: { 'X-Provider': 'openai' } // optional vendor routing
  });

  try {
    const graph: HIFGraph = await client.getGraph(traceId);
    console.log('nodes', graph.nodes.length, 'incidences', graph.incidences.length);
  } catch (err) {
    console.error('Failed to fetch graph:', err);
  }
}

main();
```

## Render with React: HypergraphViewer

```tsx
// src/App.tsx
import React from 'react';
import { HypergraphViewer } from '@your-company/explainability-ui';

export default function App() {
  return (
    <div style={{ padding: 16 }}>
      <h3>Hypergraph Viewer</h3>
      <p>Rendering a HIF graph for a given trace:</p>
      <HypergraphViewer
        traceId="trc_abc123"          // replace with your trace id
        baseUrl="http://localhost:8080"
        // apiKey="sk-..."            // optional auth
        minEdgeWeight={0.05}          // client-side pruning threshold
        grouping="supernode"          // 'supernode' | 'none'
        height={420}
        width="100%"
        refreshMs={1000}              // polling interval while graph 404
      />
    </div>
  );
}
```

## CORS for local development

Browsers enforce CORS. If your Gateway is running at http://localhost:8080 and your React app runs on http://localhost:5173 (Vite), you have two common options:

1) Use a Vite dev proxy (recommended)
- This avoids CORS by proxying /v1/* to the Gateway during development.

```ts
// vite.config.ts
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/v1': {
        target: 'http://localhost:8080',
        changeOrigin: true,
        // Optional path rewrite if your frontend mounts under a subpath
        // rewrite: (path) => path.replace(/^\/v1/, '/v1'),
      },
    },
  },
});
```

Then point the component/client to `baseUrl: ''` and fetch `'/v1/traces/.../graph'` (i.e., same origin). For the packaged ExplainabilityClient/HypergraphViewer that expect an absolute baseUrl, keep `baseUrl="http://localhost:8080"`; the proxy still works because the browser requests go to your dev server which forwards to target.

2) Enable CORS on the Gateway (server-side)
- If you control the Gateway, you can enable CORS headers in FastAPI during development. For example, add CORSMiddleware after app initialization at [app = FastAPI(...) ](services/gateway/src/app.py:333):

```python
# Development-only example (do not enable with wildcard in production):
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

Note:
- Production deployments should restrict origins and headers to the minimum required.
- You can also terminate CORS at an ingress/reverse proxy (e.g., NGINX, Envoy) instead of modifying the app.

## Run locally with Vite

```bash
# 1) Bootstrap a React + TypeScript app (if you don't already have one)
npm create vite@latest explainability-viewer -- --template react-ts
cd explainability-viewer

# 2) Install the UI package and React peers
npm i ../../sdks/typescript/explainability-ui
npm i react react-dom

# 3) Add the example component to src/App.tsx (see snippet above)

# 4) (Optional) Configure the Vite proxy (see vite.config.ts above)

# 5) Start the dev server
npm run dev
```

Navigate to http://localhost:5173 and ensure your Gateway is producing a graph for the provided traceId.

## Troubleshooting

- 404 on /graph: The trace graph is not ready; the viewer will poll until ready. Ensure the explainer completed and the Gateway has a graph for the trace id you used.
- 410 Gone: The trace expired; request a new explanation and use the new trace_id.
- CORS blocked by browser: Use the Vite proxy or enable CORS on the Gateway as shown above.
- Graph shape mismatch: Verify your payload conforms to HIF v1; see [docs/hif-schema.md](docs/hif-schema.md:1).