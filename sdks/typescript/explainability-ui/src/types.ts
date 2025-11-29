/**
 * Types aligned with HIF v1 (legacy) schema as defined in:
 * - OpenAPI components schemas HIFGraph/HIFNode/HIFIncidence/HIFMeta
 *   (see ../../../../api/openapi/hypergraph-api.yaml)
 * - Legacy v1 shape in libs schema (HIFGraphLegacy)
 *   (see ../../../../libs/hif/schema.json)
 */

export type NodeType =
  | 'sae_feature'
  | 'input_token'
  | 'output_token'
  | 'circuit_supernode';

export interface HIFNode {
  id: string;
  type: NodeType;
  label?: string;
  layer?: number | null;
  position?: number | null;
  activation_strength?: number | null;
  attributes?: Record<string, unknown>;
}

export interface HIFIncidenceMetadataProvenance {
  capture_layers?: number[];
  sae_version?: string;
  model_hash?: string;
  [k: string]: unknown;
}

export interface HIFIncidenceMetadata {
  type?: string;
  method?: string;
  window?: string;
  description?: string;
  provenance?: HIFIncidenceMetadataProvenance;
  [k: string]: unknown;
}

export interface HIFIncidence {
  id: string;
  node_ids: string[];
  weight: number;
  metadata?: HIFIncidenceMetadata;
  attributes?: Record<string, unknown>;
}

export interface HIFMetaLimits {
  min_edge_weight?: number;
  max_nodes?: number;
  max_incidences?: number;
  [k: string]: unknown;
}

export interface HIFMeta {
  model_name?: string;
  model_hash?: string;
  sae_dictionary?: string;
  granularity?: 'sentence' | 'token';
  created_at?: string; // ISO date-time
  limits?: HIFMetaLimits;
  version?: string; // e.g. "hif-1"
  [k: string]: unknown;
}

export interface HIFGraph {
  'network-type': 'directed';
  nodes: HIFNode[];
  incidences: HIFIncidence[];
  meta?: HIFMeta;
}

/**
 * Trace status mirrors OpenAPI components.schemas.TraceStatus
 * (see ../../../../api/openapi/hypergraph-api.yaml)
 */
export type TraceState =
  | 'queued'
  | 'running'
  | 'partial'
  | 'complete'
  | 'expired'
  | 'canceled'
  | 'failed';

export interface TraceStatus {
  trace_id: string;
  state: TraceState;
  progress?: number; // 0-100
  stage?: string;
  updated_at?: string; // ISO date-time
  s3_key?: string | null;
  error?: string | null;
  granularity?: 'sentence' | 'token';
  featureset?: string;
}

/**
 * Runtime type guard to defensively check HIFGraph shape
 */
export function isHIFGraph(o: unknown): o is HIFGraph {
  if (typeof o !== 'object' || o === null) return false;
  const g = o as Partial<HIFGraph>;

  if (g['network-type'] !== 'directed') return false;
  if (!Array.isArray(g.nodes)) return false;
  if (!Array.isArray(g.incidences)) return false;

  // Basic node validation
  for (const n of g.nodes) {
    if (typeof (n as any)?.id !== 'string') return false;
    const t = (n as any)?.type;
    if (
      t !== 'sae_feature' &&
      t !== 'input_token' &&
      t !== 'output_token' &&
      t !== 'circuit_supernode'
    ) {
      return false;
    }
  }

  // Basic incidence validation
  for (const e of g.incidences) {
    if (typeof (e as any)?.id !== 'string') return false;
    if (!Array.isArray((e as any)?.node_ids)) return false;
    if ((e as any).node_ids.some((x: unknown) => typeof x !== 'string'))
      return false;
    if (
      typeof (e as any)?.weight !== 'number' ||
      !Number.isFinite((e as any).weight)
    )
      return false;
    if ((e as any).node_ids.length < 2) return false; // hyperedge must involve at least 2 nodes
  }

  return true;
}