import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { select } from 'd3-selection';
import { zoom as d3zoom } from 'd3-zoom';
import {
  forceCenter,
  forceLink,
  forceManyBody,
  forceSimulation,
  Simulation,
} from 'd3-force';
import { ExplainabilityClient, HTTPError } from './client';
import type { HIFGraph, HIFIncidence, HIFNode, NodeType } from './types';

type GroupingMode = 'none' | 'supernode';

export interface HypergraphViewerProps {
  traceId: string;
  baseUrl: string;
  apiKey?: string;
  minEdgeWeight?: number; // default 0.01
  grouping?: GroupingMode; // default 'supernode'
  height?: number; // default 420
  width?: number | '100%'; // default '100%' responsive
  refreshMs?: number; // default 1000
}

type NodeExt = HIFNode & {
  x?: number;
  y?: number;
  vx?: number;
  vy?: number;
  fx?: number | null;
  fy?: number | null;
};

type LinkAgg = {
  source: string;
  target: string;
  weight: number;
  methods: Set<string>;
};

const NODE_COLORS: Record<NodeType, string> = {
  input_token: '#1f77b4',
  output_token: '#ff7f0e',
  sae_feature: '#2ca02c',
  circuit_supernode: '#9467bd',
};

const NODE_RADIUS: Record<NodeType, number> = {
  input_token: 6,
  output_token: 8,
  sae_feature: 5,
  circuit_supernode: 11,
};

function isToken(n: HIFNode): boolean {
  return n.type === 'input_token' || n.type === 'output_token';
}

function nodeTitle(n: HIFNode, membersCount?: number): string {
  const a = (k: string, v: unknown) => (v === undefined || v === null ? '' : `\n${k}: ${String(v)}`);
  const base = `${n.label ?? n.id}${a('type', n.type)}${a('layer', n.layer)}${a('activation', n.activation_strength)}`;
  return n.type === 'circuit_supernode'
    ? `${base}${a('members', membersCount ?? 0)}`
    : base;
}

function edgeTitle(e: LinkAgg): string {
  return `weight: ${e.weight.toFixed(3)}\nmethod(s): ${Array.from(e.methods).join(', ') || '-'}`;
}

/**
 * Group sae_feature nodes by prefix before ':' into circuit_supernode for visualization.
 * Only groups with size >= 2 are collapsed; singletons remain as original nodes.
 */
function applyGrouping(
  nodes: HIFNode[],
  usedFeatureIds: Set<string>,
  mode: GroupingMode
): {
  outNodes: HIFNode[];
  featureToDisplayId: Map<string, string>;
  superMembers: Map<string, string[]>;
} {
  const tokens = nodes.filter((n) => isToken(n));
  const features = nodes.filter((n) => n.type === 'sae_feature');

  const featureToDisplayId = new Map<string, string>();
  const superMembers = new Map<string, string[]>();
  const outNodes: HIFNode[] = [...tokens];

  if (mode === 'none') {
    for (const f of features) {
      if (usedFeatureIds.has(f.id)) {
        outNodes.push(f);
      }
      featureToDisplayId.set(f.id, f.id);
    }
    return { outNodes, featureToDisplayId, superMembers };
  }

  // mode === 'supernode'
  const byPrefix = new Map<string, HIFNode[]>();
  const noPrefix: HIFNode[] = [];

  for (const f of features) {
    const label = f.label ?? '';
    const idx = label.indexOf(':');
    if (idx > 0) {
      const prefix = label.slice(0, idx).trim();
      if (!byPrefix.has(prefix)) byPrefix.set(prefix, []);
      byPrefix.get(prefix)!.push(f);
    } else {
      noPrefix.push(f);
    }
  }

  // For groups with size >= 2, create supernodes
  for (const [prefix, members] of byPrefix) {
    const activeMembers = members.filter((m) => usedFeatureIds.has(m.id));
    if (activeMembers.length >= 2) {
      const superId = `super::${prefix}`;
      superMembers.set(superId, activeMembers.map((m) => m.id));
      const avgAct =
        activeMembers.reduce((s, m) => s + (m.activation_strength ?? 0), 0) /
        activeMembers.length;
      const superNode: HIFNode = {
        id: superId,
        type: 'circuit_supernode',
        label: prefix,
        layer: null,
        position: null,
        activation_strength: Number.isFinite(avgAct) ? avgAct : undefined,
      };
      outNodes.push(superNode);
      for (const m of members) {
        featureToDisplayId.set(m.id, superId);
      }
    } else {
      // Singletons stay as original nodes
      for (const m of members) {
        featureToDisplayId.set(m.id, m.id);
        if (usedFeatureIds.has(m.id)) outNodes.push(m);
      }
    }
  }

  // Features without prefix remain as-is
  for (const f of noPrefix) {
    featureToDisplayId.set(f.id, f.id);
    if (usedFeatureIds.has(f.id)) outNodes.push(f);
  }

  return { outNodes, featureToDisplayId, superMembers };
}

/**
 * Convert incidences (hyperedges) into pairwise edges for visualization.
 * Connect each feature (sae_feature or circuit_supernode-mapped) to each token present in the incidence.
 */
function incidencesToLinks(
  incidences: HIFIncidence[],
  nodeById: Map<string, HIFNode>,
  featureToDisplayId: Map<string, string>
): LinkAgg[] {
  const agg = new Map<string, LinkAgg>();

  for (const inc of incidences) {
    const presentNodes = inc.node_ids
      .map((id) => nodeById.get(id))
      .filter((n): n is HIFNode => !!n);

    const tokenIds = new Set(
      presentNodes.filter(isToken).map((n) => n.id)
    );

    // Original features only; supernodes are display artifacts
    const featureIds = new Set(
      presentNodes.filter((n) => n.type === 'sae_feature').map((n) => n.id)
    );

    // Fallback: if no tokens or no features, connect all pairs
    let pairs: Array<[string, string]> = [];
    if (tokenIds.size > 0 && featureIds.size > 0) {
      for (const f of featureIds) {
        const src = featureToDisplayId.get(f);
        if (!src) continue;
        for (const t of tokenIds) {
          pairs.push([src, t]);
        }
      }
    } else {
      // all-pairs among present nodes (skip self)
      const ids = presentNodes.map((n) => {
        if (n.type === 'sae_feature') {
          return featureToDisplayId.get(n.id) ?? n.id;
        }
        return n.id;
      });
      for (let i = 0; i < ids.length; i++) {
        for (let j = i + 1; j < ids.length; j++) {
          pairs.push([ids[i], ids[j]]);
        }
      }
    }

    const method = (inc.metadata as any)?.method as string | undefined;

    for (const [a, b] of pairs) {
      const s = a;
      const t = b;
      // normalize direction: keep feature->token if possible, else a->b
      const key = `${s}-->${t}`;
      const cur = agg.get(key);
      if (cur) {
        cur.weight += inc.weight;
        if (method) cur.methods.add(method);
      } else {
        agg.set(key, {
          source: s,
          target: t,
          weight: inc.weight,
          methods: new Set(method ? [method] : []),
        });
      }
    }
  }

  return Array.from(agg.values());
}

export function HypergraphViewer(props: HypergraphViewerProps) {
  const {
    traceId,
    baseUrl,
    apiKey,
    minEdgeWeight = 0.01,
    grouping = 'supernode',
    height = 420,
    width = '100%',
    refreshMs = 1000,
  } = props;

  const [graph, setGraph] = useState<HIFGraph | null>(null);
  const [status, setStatus] = useState<'idle' | 'loading' | 'polling' | 'ready' | 'expired' | 'error'>('idle');
  const [error, setError] = useState<string | null>(null);

  const [localMinEdgeWeight, setLocalMinEdgeWeight] = useState<number>(minEdgeWeight);
  const [localGrouping, setLocalGrouping] = useState<GroupingMode>(grouping);
  const [selectedSuper, setSelectedSuper] = useState<{ id: string; count: number } | null>(null);

  const containerRef = useRef<HTMLDivElement | null>(null);
  const svgRef = useRef<SVGSVGElement | null>(null);
  const simRef = useRef<Simulation<NodeExt, any> | null>(null);
  const [svgWidth, setSvgWidth] = useState<number>(typeof width === 'number' ? width : 640);

  // Responsive width via ResizeObserver if width === '100%'
  useEffect(() => {
    if (typeof width === 'number') {
      setSvgWidth(width);
      return;
    }
    const el = containerRef.current;
    if (!el) return;
    const ro = new ResizeObserver((entries: ResizeObserverEntry[]) => {
      for (const entry of entries) {
        const w = Math.floor(entry.contentRect.width);
        if (w > 0) setSvgWidth(w);
      }
    });
    ro.observe(el);
    // initialize
    const w0 = el.getBoundingClientRect().width;
    if (w0 > 0) setSvgWidth(Math.floor(w0));
    return () => {
      try {
        ro.disconnect();
      } catch {
        // ignore
      }
    };
  }, [width]);

  // Fetch graph with polling if 404 (not ready)
  useEffect(() => {
    let cancelled = false;
    let to: number | null = null;
    const client = new ExplainabilityClient({ baseUrl, apiKey });

    async function once() {
      if (cancelled) return;
      setStatus((s: 'idle' | 'loading' | 'polling' | 'ready' | 'expired' | 'error') => (s === 'ready' ? s : 'loading'));
      setError(null);
      try {
        const g = await client.getGraph(traceId);
        if (cancelled) return;
        setGraph(g);
        setStatus('ready');
      } catch (err: unknown) {
        if (cancelled) return;
        if (err instanceof HTTPError) {
          if (err.status === 404) {
            setStatus('polling');
            to = window.setTimeout(once, refreshMs);
          } else if (err.status === 410) {
            setStatus('expired');
          } else {
            setStatus('error');
            setError(err.message || String(err));
          }
        } else {
          setStatus('error');
          setError(String(err));
        }
      }
    }

    once();

    return () => {
      cancelled = true;
      if (to) clearTimeout(to);
    };
  }, [traceId, baseUrl, apiKey, refreshMs]);

  // Build quick lookups
  const nodeById = useMemo(() => {
    const m = new Map<string, HIFNode>();
    if (!graph) return m;
    for (const n of graph.nodes) m.set(n.id, n);
    return m;
  }, [graph]);

  const pruned = useMemo(() => {
    if (!graph) return null;

    const incidences = graph.incidences.filter(
      (e: HIFIncidence) => Number.isFinite(e.weight) && e.weight >= localMinEdgeWeight
    );

    // Determine which original feature nodes are used by pruned incidences
    const usedFeatureIds = new Set<string>();
    const usedAnyIds = new Set<string>();
    for (const inc of incidences) {
      for (const id of inc.node_ids) {
        const n = nodeById.get(id);
        if (!n) continue;
        usedAnyIds.add(id);
        if (n.type === 'sae_feature') usedFeatureIds.add(id);
      }
    }

    const { outNodes, featureToDisplayId, superMembers } = applyGrouping(graph.nodes, usedFeatureIds, localGrouping);

    // Build display node map (includes tokens + kept features + supernodes)
    const displayNodeById = new Map<string, HIFNode>();
    for (const dn of outNodes) displayNodeById.set(dn.id, dn);

    // Convert incidences -> aggregated pairwise display links
    const links = incidencesToLinks(incidences, nodeById, featureToDisplayId).filter(
      (l) => displayNodeById.has(l.source) && displayNodeById.has(l.target)
    );

    // Integrate only nodes that appear in at least one link
    const usedDisplayIds = new Set<string>();
    for (const l of links) {
      usedDisplayIds.add(l.source);
      usedDisplayIds.add(l.target);
    }
    const usedDisplayNodes = outNodes.filter((n) => usedDisplayIds.has(n.id));

    return {
      nodes: usedDisplayNodes,
      links,
      superMembers,
    };
  }, [graph, nodeById, localMinEdgeWeight, localGrouping]);

  const draw = useCallback(
    (svg: SVGSVGElement, width: number, height: number, dataset: NonNullable<typeof pruned>) => {
      // Cleanup prior sim
      if (simRef.current) {
        try {
          simRef.current.stop();
        } catch {
          // ignore
        }
        simRef.current = null;
      }

      // Clear SVG
      const root = select(svg);
      root.selectAll('*').remove();

      // Base groups
      const g = root.append('g');

      // Zoom/pan
      const zoom = d3zoom<any, any>().scaleExtent([0.2, 4]).on('zoom', (event: any) => {
        g.attr('transform', event.transform);
      });
      root.call(zoom as any);

      // Scales
      const strokeForWeight = (w: number) => Math.max(0.5, 0.5 + 4 * Math.min(1, Math.max(0, w)));

      // Prepare nodes and links arrays for simulation
      const nodes: NodeExt[] = dataset.nodes.map((n: HIFNode) => ({ ...n }));
      const nodeIndex = new Map(nodes.map((n, i) => [n.id, i]));
      const linksData = (dataset.links
        .map((l: { source: string; target: string; weight: number; methods: Set<string> }) => {
          const si = nodeIndex.get(l.source);
          const ti = nodeIndex.get(l.target);
          if (si == null || ti == null) return null;
          return {
            source: nodes[si],
            target: nodes[ti],
            weight: l.weight,
            methods: l.methods,
          };
        })
        .filter((x) => x !== null)) as Array<{ source: NodeExt; target: NodeExt; weight: number; methods: Set<string> }>;

      // Edges layer
      const edgesGroup = g.append('g').attr('stroke', '#999').attr('stroke-opacity', 0.6);

      const edgesSel = edgesGroup
        .selectAll<SVGLineElement, { source: NodeExt; target: NodeExt; weight: number; methods: Set<string> }>('line')
        .data(linksData)
        .enter()
        .append('line')
        .attr('stroke-width', (d: { source: NodeExt; target: NodeExt; weight: number; methods: Set<string> }) => strokeForWeight(d.weight));

      edgesSel
        .append('title')
        .text((d: { source: NodeExt; target: NodeExt; weight: number; methods: Set<string> }) =>
          edgeTitle({ source: d.source.id, target: d.target.id, weight: d.weight, methods: d.methods })
        );

      // Nodes layer
      const nodeSel = g
        .append('g')
        .selectAll<SVGCircleElement, NodeExt>('circle')
        .data(nodes)
        .enter()
        .append('circle')
        .attr('r', (d: NodeExt) => NODE_RADIUS[d.type])
        .attr('fill', (d: NodeExt) => NODE_COLORS[d.type])
        .attr('stroke', '#222')
        .attr('stroke-width', 0.5)
        .on('click', (_evt: MouseEvent, d: NodeExt) => {
          if (d.type === 'circuit_supernode') {
            const members = dataset.superMembers.get(d.id)?.length ?? 0;
            setSelectedSuper({ id: d.id, count: members });
          } else {
            setSelectedSuper(null);
          }
        });

      nodeSel
        .append('title')
        .text((d: NodeExt) => nodeTitle(d, dataset.superMembers.get(d.id)?.length));

      // Force simulation
      const sim = forceSimulation<NodeExt>(nodes as any)
        .force(
          'link',
          (forceLink<NodeExt, any>(linksData as any) as any)
            .id((d: any) => d.id)
            .distance((d: any) => 50 / Math.max(0.2, Math.min(1, d.weight ?? 0.5)))
            .strength(0.2)
        )
        .force('charge', forceManyBody().strength(-80))
        .force('center', forceCenter(width / 2, height / 2))
        .on('tick', () => {
          edgesSel
            .attr('x1', (d: any) => d.source.x)
            .attr('y1', (d: any) => d.source.y)
            .attr('x2', (d: any) => d.target.x)
            .attr('y2', (d: any) => d.target.y);

          nodeSel.attr('cx', (d: any) => d.x).attr('cy', (d: any) => d.y);
        });

      simRef.current = sim;
    },
    []
  );

  // Re-render graph when data or size changes
  useEffect(() => {
    if (!svgRef.current || !pruned) return;
    draw(svgRef.current, svgWidth, height, pruned);
    // Cleanup on unmount
    return () => {
      if (simRef.current) {
        try {
          simRef.current.stop();
        } catch {
          // ignore
        }
        simRef.current = null;
      }
    };
  }, [draw, pruned, svgWidth, height]);

  const onWeightChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const v = parseFloat(e.target.value);
    if (!Number.isNaN(v)) setLocalMinEdgeWeight(Math.max(0, Math.min(1, v)));
  };

  const onGroupingChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const v = e.target.value === 'none' ? 'none' : 'supernode';
    setLocalGrouping(v);
  };

  // Status banner text
  const statusText = useMemo(() => {
    switch (status) {
      case 'loading':
        return 'Loading graph...';
      case 'polling':
        return 'Graph not ready (404). Polling...';
      case 'expired':
        return 'Trace expired (410).';
      case 'error':
        return `Error loading graph${error ? `: ${error}` : ''}`;
      case 'ready':
        return '';
      default:
        return '';
    }
  }, [status, error]);

  return (
    <div ref={containerRef} style={{ width: typeof width === 'number' ? `${width}px` : '100%' }}>
      {/* Controls */}
      <div
        style={{
          display: 'flex',
          gap: 12,
          alignItems: 'center',
          fontFamily: 'system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif',
          marginBottom: 8,
        }}
      >
        <div style={{ fontSize: 12, color: '#555' }}>
          {status !== 'ready' && statusText ? <span>{statusText}</span> : null}
        </div>
        <div style={{ marginLeft: 'auto', display: 'flex', gap: 16, alignItems: 'center' }}>
          <label style={{ fontSize: 12 }}>
            Min edge weight: {localMinEdgeWeight.toFixed(2)}
            <input
              type="range"
              min={0}
              max={1}
              step={0.01}
              value={localMinEdgeWeight}
              onChange={onWeightChange}
              style={{ marginLeft: 8, verticalAlign: 'middle' }}
            />
          </label>
          <label style={{ fontSize: 12 }}>
            Grouping:
            <select
              value={localGrouping}
              onChange={onGroupingChange}
              style={{ marginLeft: 8, padding: '2px 6px', fontSize: 12 }}
            >
              <option value="supernode">supernode</option>
              <option value="none">none</option>
            </select>
          </label>
        </div>
      </div>

      {/* Graph + sidebar info */}
      <div style={{ position: 'relative', border: '1px solid #e2e2e2', borderRadius: 6 }}>
        <svg
          ref={svgRef}
          width={svgWidth}
          height={height}
          style={{ display: 'block', background: '#fafafa', borderRadius: 6 }}
        />
        {selectedSuper ? (
          <div
            style={{
              position: 'absolute',
              right: 8,
              top: 8,
              background: 'rgba(255,255,255,0.95)',
              border: '1px solid #ddd',
              borderRadius: 6,
              padding: 8,
              maxWidth: 240,
              fontFamily: 'system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif',
              fontSize: 12,
              boxShadow: '0 2px 8px rgba(0,0,0,0.08)',
            }}
          >
            <div style={{ fontWeight: 600, marginBottom: 4 }}>Supernode</div>
            <div>ID: {selectedSuper.id}</div>
            <div>Members: {selectedSuper.count}</div>
            <button
              onClick={() => setSelectedSuper(null)}
              style={{
                marginTop: 6,
                fontSize: 12,
                padding: '4px 8px',
                background: '#f0f0f0',
                border: '1px solid #ccc',
                borderRadius: 4,
                cursor: 'pointer',
              }}
            >
              Close
            </button>
          </div>
        ) : null}
      </div>

      {/* Footer info */}
      <div style={{ marginTop: 6, fontSize: 11, color: '#777' }}>
        {pruned ? (
          <span>
            Nodes: {pruned.nodes.length} • Edges: {pruned.links.length}
          </span>
        ) : null}
      </div>
    </div>
  );
}

export default HypergraphViewer;