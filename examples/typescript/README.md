# Example: TypeScript + React viewer app (Vite)

This example shows how to bootstrap a minimal React + TypeScript app with Vite, fetch a HIF graph using the Explainability client, and render it using the HypergraphViewer component.

References:
- Client: [ExplainabilityClient](sdks/typescript/explainability-ui/src/client.ts:19)
- Viewer: [HypergraphViewer](sdks/typescript/explainability-ui/src/HypergraphViewer.tsx:233)
- Types: [types.ts](sdks/typescript/explainability-ui/src/types.ts:1)
- API reference: [docs/api-reference.md](docs/api-reference.md:1)
- HIF schema: [docs/hif-schema.md](docs/hif-schema.md:1)

Gateway (local): http://localhost:8080

Note: When AUTH_MODE=static is enabled, direct Gateway calls require Authorization: Bearer $TOKEN (or pass apiKey to the client).

## 0) Prerequisites

- Node.js 18+ and npm or pnpm
- A running Gateway at http://localhost:8080
- A valid trace_id to visualize (create by calling POST /v1/chat/completions with `x-explain-mode: hypergraph`)

## 1) Bootstrap a Vite React + TypeScript app

```bash
npm create vite@latest explainability-viewer -- --template react-ts
cd explainability-viewer
npm i
```

If you use pnpm:

```bash
pnpm create vite explainability-viewer --template react-ts
cd explainability-viewer
pnpm i
```

## 2) Install the UI SDK and React peers

During local development in this monorepo, install from the relative path:

```bash
# from examples/typescript/explainability-viewer or any app dir
npm i ../../sdks/typescript/explainability-ui
npm i react react-dom
```

If the package is published, you can install it from your registry:

```bash
npm i @your-company/explainability-ui
npm i react react-dom
```

## 3) (Optional) Configure a Vite dev proxy

To avoid CORS issues during local development, proxy `/v1` to the Gateway.

Create or edit `vite.config.ts`:

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
      },
    },
  },
});
```

Notes:
- With this proxy, your browser-origin requests to `/v1/...` are forwarded to the Gateway.
- You may still set `baseUrl="http://localhost:8080"` on the client/viewer; the browser dev server will forward to the target.

Server-side CORS alternative:
- If you control the Gateway, you can enable CORS middleware during development. See example in the quickstart and API docs near [app = FastAPI(...)](services/gateway/src/app.py:333).

## 4) Fetch a graph programmatically

Create `src/fetch-graph.ts`:

```ts
import { ExplainabilityClient, type HIFGraph } from '@your-company/explainability-ui';

async function main() {
  const baseUrl = 'http://localhost:8080';
  const traceId = 'trc_replace_me';

  const client = new ExplainabilityClient({
    baseUrl,
    // apiKey: 'sk-...' // optional
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

Run it with:
```bash
npx ts-node src/fetch-graph.ts
```

## 5) Render with React

Edit `src/App.tsx`:

```tsx
import React from 'react';
import { HypergraphViewer } from '@your-company/explainability-ui';

export default function App() {
  return (
    <div style={{ padding: 16 }}>
      <h3>Hypergraph Viewer</h3>
      <p>Provide a traceId that has a completed HIF graph.</p>
      <HypergraphViewer
        traceId="trc_replace_me"        // update this to a real trace id
        baseUrl="http://localhost:8080" // Gateway URL
        // apiKey="sk-..."              // optional bearer token
        minEdgeWeight={0.05}            // client-side pruning
        grouping="supernode"            // 'supernode' | 'none'
        height={420}
        width="100%"
        refreshMs={1000}                // polling interval while graph returns 404
      />
    </div>
  );
}
```

Start the dev server:

```bash
npm run dev
```

Open http://localhost:5173 and verify the viewer loads. If a graph is not ready, the viewer polls. If the trace is expired (410), an "expired" banner is shown.

## 6) End-to-end flow (recap)

- Send chat to Gateway with explain headers; see cURL in [docs/api-reference.md](docs/api-reference.md:36)
- Extract `explanation_metadata.trace_id` from the response
- Fetch status or stream SSE:
  - GET `/v1/traces/{trace_id}/status`
  - GET `/v1/traces/{trace_id}/stream`
- Fetch the graph:
  - GET `/v1/traces/{trace_id}/graph`
- Render the graph with [HypergraphViewer](sdks/typescript/explainability-ui/src/HypergraphViewer.tsx:233)

## 7) Troubleshooting

- CORS blocked:
  - Use the Vite proxy above, or enable CORS on the Gateway.
- 404 on /graph:
  - The graph is not ready; viewer will keep polling. Ensure the Explainer produced a graph for your `traceId`.
- 410 Gone:
  - The trace expired; request a new explanation (new trace).
- Invalid graph shape:
  - Ensure your payload matches HIF v1; see [docs/hif-schema.md](docs/hif-schema.md:1).
- Types mismatch:
  - Update your local package build: `npm -C ../../sdks/typescript/explainability-ui run build`.

## 8) What’s inside the UI SDK

- ExplainabilityClient (HTTP Client): [client.ts](sdks/typescript/explainability-ui/src/client.ts:19)
  - `getGraph(traceId)` — GET `/v1/traces/{trace_id}/graph`
  - `getStatus(traceId)` — GET `/v1/traces/{trace_id}/status`
- HypergraphViewer (React): [HypergraphViewer.tsx](sdks/typescript/explainability-ui/src/HypergraphViewer.tsx:233)
  - Props:
    - `traceId: string`
    - `baseUrl: string`
    - `apiKey?: string`
    - `minEdgeWeight?: number` (default 0.01)
    - `grouping?: 'none' | 'supernode'` (default 'supernode')
    - `height?: number` (default 420)
    - `width?: number | '100%'` (default '100%')
    - `refreshMs?: number` (default 1000)

## 9) License

Apache-2.0