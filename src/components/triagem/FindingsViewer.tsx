'use client';

import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react';

export type FindingCategory =
  | 'structure'
  | 'tooth'
  | 'pathology'
  | 'anatomy'
  | 'treatment'
  | 'other';

export type RawFinding = {
  label?: string;
  class?: string;
  name?: string;
  category?: string;
  tipo?: string;
  tipo_lesao?: string;
  confidence?: number;
  score?: number;
  prob?: number;
  confidence_score?: number;
  probability?: number;
  bbox?: number[];
  box?: number[];
  bounding_box?: number[];
  segmentation?: number[][] | number[] | string;
  polygon?: number[][] | number[];
  source_model?: string;
  sourceModel?: string;
  model?: string;
  model_type?: string;
  source?: string;
  tooth_id?: number;
  toothId?: number;
  displayLabel?: string;
  canonicalLabel?: string;
  categoryLabel?: FindingCategory;
  toothType?: string;
  toothGroup?: string;
};

type LabelMeta = {
  display: string;
  canonical: string;
  category: FindingCategory;
  toothGroup?: string;
};

export type NormalizedFinding = {
  id: string;
  label: string;
  confidence?: number;
  bbox?: number[];
  segmentation?: number[][];
  sourceModel?: string;
  displayLabel: string;
  canonicalLabel: string;
  category: FindingCategory;
  toothId?: number;
  toothType?: string;
  toothGroup?: string;
  sourceKind: 'yolo' | 'detectron' | 'other';
};

type ViewerTab = 'pathology' | 'treatment' | 'tooth';

type FindingsViewerProps = {
  imageUrl?: string | null;
  findings: RawFinding[];
  title?: string;
  subtitle?: string;
  toolbar?: ReactNode;
  className?: string;
  showList?: boolean;
  defaultTab?: ViewerTab;
  isLoading?: boolean;
  loadingLabel?: string;
  enableToothFusionPreview?: boolean;
  showModelToggles?: boolean;
  enableClickSelect?: boolean;
  enableDebug?: boolean;
  showTeethToggle?: boolean;
  structureLabel?: string;
};

const LABEL_DICTIONARY: Record<string, LabelMeta> = {
  dente: { display: 'Esmalte', canonical: 'esmalte', category: 'structure' },
  dentina: { display: 'Dentina', canonical: 'dentina', category: 'structure' },
  polpa: { display: 'Polpa', canonical: 'polpa', category: 'structure' },
  restauracao: { display: 'Restauração', canonical: 'restauracao', category: 'treatment' },
  material_restaurador: { display: 'Restauração', canonical: 'restauracao', category: 'treatment' },
  coroa: { display: 'Coroa', canonical: 'coroa', category: 'treatment' },
  ponte: { display: 'Ponte', canonical: 'ponte', category: 'treatment' },
  carie: { display: 'Cárie', canonical: 'carie', category: 'pathology' },
  decay: { display: 'Cárie', canonical: 'carie', category: 'pathology' },
  filling: { display: 'Restauração', canonical: 'restauracao', category: 'treatment' },
  periapical_lesion: {
    display: 'Lesão periapical',
    canonical: 'periapical_lesion',
    category: 'pathology',
  },
  periapicopatia: {
    display: 'Lesão periapical',
    canonical: 'periapical_lesion',
    category: 'pathology',
  },
  impactado: { display: 'Dente impactado', canonical: 'impactado', category: 'pathology' },
  resto_residual: { display: 'Raiz residual', canonical: 'raiz_residual', category: 'pathology' },
  raiz_residual: { display: 'Raiz residual', canonical: 'raiz_residual', category: 'pathology' },
  tto_endo: { display: 'Tratamento endodôntico', canonical: 'tto_endo', category: 'treatment' },
  canal_mandibular: {
    display: 'Canal mandibular',
    canonical: 'canal_mandibular',
    category: 'anatomy',
  },
  jaw: { display: 'Mandíbula', canonical: 'mandibula', category: 'anatomy' },
  mandibula: { display: 'Mandíbula', canonical: 'mandibula', category: 'anatomy' },
  maxila: { display: 'Maxila', canonical: 'maxila', category: 'anatomy' },
  molar_inf: {
    display: 'Molar inferior',
    canonical: 'molar_inf',
    category: 'tooth',
    toothGroup: 'molar',
  },
  molar_sup: {
    display: 'Molar superior',
    canonical: 'molar_sup',
    category: 'tooth',
    toothGroup: 'molar',
  },
  pre_molar_inf: {
    display: 'Pré-molar inferior',
    canonical: 'pre_molar_inf',
    category: 'tooth',
    toothGroup: 'pre_molar',
  },
  pre_molar_sup: {
    display: 'Pré-molar superior',
    canonical: 'pre_molar_sup',
    category: 'tooth',
    toothGroup: 'pre_molar',
  },
  incisivo_central_inf: {
    display: 'Incisivo central inferior',
    canonical: 'incisivo_central_inf',
    category: 'tooth',
    toothGroup: 'incisivo',
  },
  incisivo_central_sup: {
    display: 'Incisivo central superior',
    canonical: 'incisivo_central_sup',
    category: 'tooth',
    toothGroup: 'incisivo',
  },
  incisivo_lateral_inf: {
    display: 'Incisivo lateral inferior',
    canonical: 'incisivo_lateral_inf',
    category: 'tooth',
    toothGroup: 'incisivo',
  },
  incisivo_lateral_sup: {
    display: 'Incisivo lateral superior',
    canonical: 'incisivo_lateral_sup',
    category: 'tooth',
    toothGroup: 'incisivo',
  },
  canino_inf: {
    display: 'Canino inferior',
    canonical: 'canino_inf',
    category: 'tooth',
    toothGroup: 'canino',
  },
  canino_sup: {
    display: 'Canino superior',
    canonical: 'canino_sup',
    category: 'tooth',
    toothGroup: 'canino',
  },
  terceiro_molar_inf: {
    display: '3º molar inferior',
    canonical: 'terceiro_molar_inf',
    category: 'tooth',
    toothGroup: 'terceiro_molar',
  },
  terceiro_molar_sup: {
    display: '3º molar superior',
    canonical: 'terceiro_molar_sup',
    category: 'tooth',
    toothGroup: 'terceiro_molar',
  },
};

const ANATOMY_KEYWORDS = [
  'dentina',
  'esmalte',
  'polpa',
  'dente',
  'enamel',
  'dentin',
  'dentine',
  'tooth',
  'teeth',
  'pulp',
  'mandibula',
  'maxila',
  'jaw',
];

const slugLabel = (label: string) =>
  label
    .trim()
    .toLowerCase()
    .replace(/\s+/g, '_')
    .replace(/_+/g, '_');

const humanizeLabel = (label: string) =>
  label
    .replace(/[_-]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
    .replace(/^\w|\s\w/g, (match) => match.toUpperCase());

const getLabelMeta = (label: string): LabelMeta => {
  const key = slugLabel(label);
  const meta = LABEL_DICTIONARY[key];
  if (meta) {
    return meta;
  }
  return {
    display: humanizeLabel(label),
    canonical: key,
    category: ANATOMY_KEYWORDS.some((keyword) => key.includes(keyword)) ? 'anatomy' : 'other',
  };
};

const toNumber = (value: unknown) => {
  if (typeof value === 'number' && Number.isFinite(value)) {
    return value;
  }
  if (typeof value === 'string') {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : undefined;
  }
  return undefined;
};

const parseSvgPath = (path: string) => {
  const matches = path.match(/-?\d*\.?\d+/g);
  if (!matches) {
    return undefined;
  }
  const points: number[][] = [];
  for (let i = 0; i < matches.length - 1; i += 2) {
    const x = Number(matches[i]);
    const y = Number(matches[i + 1]);
    if (Number.isFinite(x) && Number.isFinite(y)) {
      points.push([x, y]);
    }
  }
  return points.length >= 3 ? points : undefined;
};

const pointsFromFlat = (flat: number[]): number[][] | undefined => {
  if (flat.length < 6) {
    return undefined;
  }
  const points: number[][] = [];
  for (let i = 0; i < flat.length - 1; i += 2) {
    const x = flat[i];
    const y = flat[i + 1];
    if (typeof x === 'number' && typeof y === 'number') {
      points.push([x, y]);
    }
  }
  return points.length >= 3 ? points : undefined;
};

const pointsFromPairs = (pairs: unknown[]): number[][] | undefined => {
  if (
    pairs.length < 3 ||
    !pairs.every(
      (point) =>
        Array.isArray(point) &&
        point.length === 2 &&
        typeof point[0] === 'number' &&
        typeof point[1] === 'number',
    )
  ) {
    return undefined;
  }
  return (pairs as number[][]).map((point) => [point[0], point[1]]);
};

const normalizeSegmentation = (
  segmentation?: RawFinding['segmentation'],
): number[][] | undefined => {
  if (!segmentation) {
    return undefined;
  }
  if (typeof segmentation === 'string') {
    const trimmed = segmentation.trim();
    if (trimmed.startsWith('[')) {
      try {
        const parsed = JSON.parse(trimmed) as RawFinding['segmentation'];
        const normalized = normalizeSegmentation(parsed);
        if (normalized) {
          return normalized;
        }
      } catch (error) {
        console.warn('Nao foi possivel ler o JSON de segmentacao.', error);
      }
    }
    return parseSvgPath(trimmed);
  }
  if (Array.isArray(segmentation)) {
    if (segmentation.length === 0) {
      return undefined;
    }
    if (Array.isArray(segmentation[0])) {
      const first = segmentation[0] as unknown[];
      if (first.length === 0) {
        return undefined;
      }
      if (Array.isArray(first[0])) {
        const polygons = (segmentation as unknown[][])
          .map((polygon) => pointsFromPairs(polygon))
          .filter((polygon): polygon is number[][] => Boolean(polygon))
          .sort((a, b) => b.length - a.length);
        return polygons[0];
      }
      if (typeof first[0] === 'number') {
        const points = pointsFromPairs(segmentation as unknown[]);
        if (points) {
          return points;
        }
        const polygons = (segmentation as unknown[])
          .filter((item): item is number[] => Array.isArray(item) && item.every((value) => typeof value === 'number'))
          .map((polygon) => pointsFromFlat(polygon))
          .filter((polygon): polygon is number[][] => Boolean(polygon))
          .sort((a, b) => b.length - a.length);
        return polygons[0];
      }
      return undefined;
    }
    if ((segmentation as unknown[]).every((value) => typeof value === 'number')) {
      return pointsFromFlat(segmentation as number[]);
    }
  }
  return undefined;
};

const getSourceKind = (finding: { sourceModel?: string; model?: string }) => {
  const source = `${finding.sourceModel ?? ''} ${finding.model ?? ''}`.toLowerCase();
  if (source.includes('yolo')) {
    return 'yolo' as const;
  }
  if (source.includes('detectron') || source.includes('rcnn') || source.includes('mask')) {
    return 'detectron' as const;
  }
  return 'other' as const;
};

const normalizeFinding = (raw: RawFinding, index: number): NormalizedFinding | null => {
  const labelCandidate =
    raw.label ?? raw.class ?? raw.name ?? raw.category ?? raw.tipo ?? raw.tipo_lesao;
  if (!labelCandidate) {
    return null;
  }
  const label = String(labelCandidate).trim();
  if (!label) {
    return null;
  }
  const confidence = toNumber(
    raw.confidence ?? raw.score ?? raw.prob ?? raw.confidence_score ?? raw.probability,
  );
  const bbox = Array.isArray(raw.bbox)
    ? raw.bbox
    : Array.isArray(raw.box)
      ? raw.box
      : Array.isArray(raw.bounding_box)
        ? raw.bounding_box
        : undefined;
  const segmentation =
    normalizeSegmentation(raw.segmentation) ?? normalizeSegmentation(raw.polygon);
  const sourceModel =
    raw.source_model ??
    raw.sourceModel ??
    raw.model ??
    raw.model_type ??
    raw.source;
  const toothId = toNumber(raw.tooth_id ?? raw.toothId);

  const meta = getLabelMeta(label);
  const displayLabel =
    typeof raw.displayLabel === 'string' && raw.displayLabel.trim()
      ? raw.displayLabel
      : meta.display;
  const canonicalLabel =
    typeof raw.canonicalLabel === 'string' && raw.canonicalLabel.trim()
      ? raw.canonicalLabel
      : meta.canonical;
  const category =
    raw.categoryLabel ??
    (typeof raw.category === 'string'
      ? (raw.category as FindingCategory)
      : undefined) ??
    meta.category;
  const toothType =
    typeof raw.toothType === 'string' && raw.toothType.trim()
      ? raw.toothType
      : category === 'tooth'
        ? displayLabel
        : undefined;
  const toothGroup = raw.toothGroup ?? meta.toothGroup;

  const sourceKind = getSourceKind({ sourceModel, model: typeof raw.model === 'string' ? raw.model : undefined });

  return {
    id: `${slugLabel(label)}-${index}`,
    label,
    confidence,
    bbox,
    segmentation,
    sourceModel,
    displayLabel,
    canonicalLabel,
    category,
    toothId,
    toothType,
    toothGroup,
    sourceKind,
  };
};

const bboxFromSegmentation = (segmentation?: number[][]) => {
  if (!segmentation || segmentation.length === 0) {
    return undefined;
  }
  let minX = Number.POSITIVE_INFINITY;
  let minY = Number.POSITIVE_INFINITY;
  let maxX = Number.NEGATIVE_INFINITY;
  let maxY = Number.NEGATIVE_INFINITY;
  segmentation.forEach(([x, y]) => {
    minX = Math.min(minX, x);
    minY = Math.min(minY, y);
    maxX = Math.max(maxX, x);
    maxY = Math.max(maxY, y);
  });
  if (!Number.isFinite(minX) || !Number.isFinite(minY)) {
    return undefined;
  }
  return [minX, minY, maxX, maxY];
};

const getFindingBbox = (finding: NormalizedFinding) => {
  if (finding.bbox?.length === 4) {
    return finding.bbox;
  }
  return bboxFromSegmentation(finding.segmentation);
};

const bboxCenter = (bbox: number[]) => ({
  cx: (bbox[0] + bbox[2]) / 2,
  cy: (bbox[1] + bbox[3]) / 2,
});

const bboxIou = (a: number[], b: number[]) => {
  const x1 = Math.max(a[0], b[0]);
  const y1 = Math.max(a[1], b[1]);
  const x2 = Math.min(a[2], b[2]);
  const y2 = Math.min(a[3], b[3]);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  if (inter <= 0) {
    return 0;
  }
  const areaA = Math.max(0, a[2] - a[0]) * Math.max(0, a[3] - a[1]);
  const areaB = Math.max(0, b[2] - b[0]) * Math.max(0, b[3] - b[1]);
  const denom = areaA + areaB - inter;
  return denom > 0 ? inter / denom : 0;
};

const attachToothMatches = (findings: NormalizedFinding[]) => {
  type ToothCandidate = {
    bbox: number[];
    cx: number;
    cy: number;
    toothLabel: string;
    toothGroup?: string;
  };

  type ToothMatch = {
    toothLabel: string;
    toothGroup?: string;
    score: number;
    dist: number;
  };

  const toothCandidates: ToothCandidate[] = findings
    .filter(
      (finding) =>
        finding.category === 'tooth' &&
        finding.sourceKind === 'yolo' &&
        getFindingBbox(finding),
    )
    .map((finding) => {
      const bbox = getFindingBbox(finding) as number[];
      const { cx, cy } = bboxCenter(bbox);
      return {
        bbox,
        cx,
        cy,
        toothLabel: finding.displayLabel,
        toothGroup: finding.toothGroup,
      };
    });

  if (toothCandidates.length === 0) {
    return findings;
  }

  return findings.map((finding) => {
    if (finding.category === 'tooth' || finding.category === 'anatomy' || finding.category === 'other') {
      return finding;
    }
    const bbox = getFindingBbox(finding);
    if (!bbox) {
      return finding;
    }
    let best: ToothMatch | null = null;
    const center = bboxCenter(bbox);
    toothCandidates.forEach((candidate) => {
      const iou = bboxIou(bbox, candidate.bbox);
      const dist =
        (candidate.cx - center.cx) * (candidate.cx - center.cx) +
        (candidate.cy - center.cy) * (candidate.cy - center.cy);
      if (!best || iou > best.score || (iou === best.score && dist < best.dist)) {
        best = { toothLabel: candidate.toothLabel, toothGroup: candidate.toothGroup, score: iou, dist };
      }
    });
    if (!best) {
      return finding;
    }
    const matched = best as ToothMatch;
    return { ...finding, toothType: matched.toothLabel, toothGroup: matched.toothGroup };
  });
};

const applyCariesDepth = (findings: NormalizedFinding[]) => {
  const dentinaBoxes = findings
    .filter((finding) => finding.canonicalLabel === 'dentina')
    .map((finding) => getFindingBbox(finding))
    .filter((bbox): bbox is number[] => Boolean(bbox));
  const polpaBoxes = findings
    .filter((finding) => finding.canonicalLabel === 'polpa')
    .map((finding) => getFindingBbox(finding))
    .filter((bbox): bbox is number[] => Boolean(bbox));

  return findings.map((finding) => {
    if (finding.canonicalLabel !== 'carie' || finding.sourceKind !== 'detectron') {
      return finding;
    }
    const bbox = getFindingBbox(finding);
    if (!bbox) {
      return finding;
    }
    const maxPolpa = polpaBoxes.reduce((max, box) => Math.max(max, bboxIou(bbox, box)), 0);
    const maxDentina = dentinaBoxes.reduce((max, box) => Math.max(max, bboxIou(bbox, box)), 0);
    let canonical = 'carie_esmalte';
    let display = 'Cárie em esmalte';
    if (maxPolpa >= 0.02) {
      canonical = 'carie_profunda';
      display = 'Cárie profunda';
    } else if (maxDentina >= 0.02) {
      canonical = 'carie_dentina';
      display = 'Cárie em dentina';
    }
    return { ...finding, canonicalLabel: canonical, displayLabel: display };
  });
};

export const normalizeFindings = (findings: RawFinding[]) => {
  const normalized = findings
    .map((finding, index) => normalizeFinding(finding, index))
    .filter((item): item is NormalizedFinding => Boolean(item));
  return applyCariesDepth(attachToothMatches(normalized));
};

const polygonArea = (points: number[][]) => {
  if (points.length < 3) {
    return 0;
  }
  let area = 0;
  for (let i = 0; i < points.length; i += 1) {
    const [x1, y1] = points[i];
    const [x2, y2] = points[(i + 1) % points.length];
    area += x1 * y2 - x2 * y1;
  }
  return Math.abs(area) / 2;
};

const pruneNearbyPoints = (points: number[][], minDistance: number) => {
  if (points.length < 2) {
    return points;
  }
  const minDistSq = minDistance * minDistance;
  const cleaned: number[][] = [points[0]];
  for (let i = 1; i < points.length; i += 1) {
    const last = cleaned[cleaned.length - 1];
    const current = points[i];
    const dx = current[0] - last[0];
    const dy = current[1] - last[1];
    if (dx * dx + dy * dy >= minDistSq) {
      cleaned.push(current);
    }
  }
  if (cleaned.length > 2) {
    const first = cleaned[0];
    const last = cleaned[cleaned.length - 1];
    const dx = first[0] - last[0];
    const dy = first[1] - last[1];
    if (dx * dx + dy * dy < minDistSq) {
      cleaned.pop();
    }
  }
  return cleaned;
};

const smoothPolygon = (points: number[][], iterations: number) => {
  let current = points;
  for (let iter = 0; iter < iterations; iter += 1) {
    const next = current.map((point, index) => {
      const prev = current[(index - 1 + current.length) % current.length];
      const nextPoint = current[(index + 1) % current.length];
      return [
        (prev[0] + point[0] + nextPoint[0]) / 3,
        (prev[1] + point[1] + nextPoint[1]) / 3,
      ];
    });
    current = next;
  }
  return current;
};

const cleanSegmentation = (
  points: number[][],
  {
    minArea,
    minPoints,
    minDistance,
    smoothIterations,
  }: { minArea: number; minPoints: number; minDistance: number; smoothIterations: number },
) => {
  if (points.length < minPoints) {
    return undefined;
  }
  const pruned = pruneNearbyPoints(points, minDistance);
  if (pruned.length < minPoints) {
    return undefined;
  }
  if (polygonArea(pruned) < minArea) {
    return undefined;
  }
  return smoothPolygon(pruned, smoothIterations);
};

const getFindingStyle = (finding: NormalizedFinding) => {
  const canonical = (finding.canonicalLabel ?? finding.label).toLowerCase();
  if (finding.category === 'pathology' || canonical.includes('carie') || canonical.includes('lesao')) {
    return { stroke: '#f87171', fill: '#f87171' };
  }
  if (finding.category === 'treatment') {
    return { stroke: '#fbbf24', fill: '#fbbf24' };
  }
  if (canonical === 'esmalte') {
    return { stroke: '#38bdf8', fill: '#38bdf8' };
  }
  if (canonical === 'dentina') {
    return { stroke: '#f59e0b', fill: '#f59e0b' };
  }
  if (canonical === 'polpa') {
    return { stroke: '#009739', fill: '#009739' };
  }
  if (finding.category === 'structure') {
    return { stroke: '#38bdf8', fill: '#38bdf8' };
  }
  if (finding.category === 'tooth') {
    return { stroke: '#4ade80', fill: '#4ade80' };
  }
  return { stroke: '#94a3b8', fill: '#94a3b8' };
};

const getGroupKey = (finding: NormalizedFinding) => {
  if (finding.category === 'tooth') {
    return `tooth:${finding.id}`;
  }
  return `${finding.category}:${finding.canonicalLabel ?? slugLabel(finding.label)}`;
};

const hexToRgba = (hex: string, alpha: number) => {
  const normalized = hex.replace('#', '');
  if (normalized.length === 3) {
    const r = parseInt(normalized[0] + normalized[0], 16);
    const g = parseInt(normalized[1] + normalized[1], 16);
    const b = parseInt(normalized[2] + normalized[2], 16);
    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
  }
  if (normalized.length === 6) {
    const r = parseInt(normalized.slice(0, 2), 16);
    const g = parseInt(normalized.slice(2, 4), 16);
    const b = parseInt(normalized.slice(4, 6), 16);
    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
  }
  return `rgba(148, 163, 184, ${alpha})`;
};

const toggleButtonClass = (active: boolean) =>
  [
    'rounded-full border px-3 py-1 text-[11px] font-semibold uppercase tracking-wider transition-all',
    active
      ? 'bg-brand-blue text-white border-brand-blue/40 shadow-sm'
      : 'border-brand-blue/30 text-brand-blue hover:bg-white/40 dark:border-white/20 dark:text-white/70 dark:hover:text-white dark:hover:bg-white/10',
  ].join(' ');

const tabButtonClass = (active: boolean) =>
  [
    'px-3 py-2 text-[11px] font-semibold uppercase tracking-wider rounded-full transition-all',
    active
      ? 'bg-white/70 text-brand-blue shadow-sm dark:bg-white/10 dark:text-white'
      : 'text-brand-blue/70 hover:text-brand-blue dark:text-white/60 dark:hover:text-white',
  ].join(' ');

const drawRoundedRect = (
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  width: number,
  height: number,
  radius: number,
) => {
  const r = Math.min(radius, width / 2, height / 2);
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + width - r, y);
  ctx.quadraticCurveTo(x + width, y, x + width, y + r);
  ctx.lineTo(x + width, y + height - r);
  ctx.quadraticCurveTo(x + width, y + height, x + width - r, y + height);
  ctx.lineTo(x + r, y + height);
  ctx.quadraticCurveTo(x, y + height, x, y + height - r);
  ctx.lineTo(x, y + r);
  ctx.quadraticCurveTo(x, y, x + r, y);
  ctx.closePath();
  ctx.stroke();
  ctx.fill();
};

const getImageRenderMetrics = (img: HTMLImageElement) => {
  const boxWidth = img.clientWidth;
  const boxHeight = img.clientHeight;
  const naturalWidth = img.naturalWidth || boxWidth;
  const naturalHeight = img.naturalHeight || boxHeight;
  if (!boxWidth || !boxHeight || !naturalWidth || !naturalHeight) {
    return null;
  }
  const imageRatio = naturalWidth / naturalHeight;
  const boxRatio = boxWidth / boxHeight;
  let renderWidth = boxWidth;
  let renderHeight = boxHeight;
  let offsetX = 0;
  let offsetY = 0;

  if (Math.abs(imageRatio - boxRatio) > 0.001) {
    if (imageRatio > boxRatio) {
      renderWidth = boxWidth;
      renderHeight = boxWidth / imageRatio;
      offsetY = (boxHeight - renderHeight) / 2;
    } else {
      renderHeight = boxHeight;
      renderWidth = boxHeight * imageRatio;
      offsetX = (boxWidth - renderWidth) / 2;
    }
  }

  return {
    boxWidth,
    boxHeight,
    naturalWidth,
    naturalHeight,
    renderWidth,
    renderHeight,
    offsetX,
    offsetY,
  };
};

type DebugSummary = {
  topLabels: { label: string; count: number }[];
  pathologyBySource: Record<string, number>;
  treatmentBySource: Record<string, number>;
  carieBySource: Record<string, number>;
  restauracaoBySource: Record<string, number>;
};

export default function FindingsViewer({
  imageUrl,
  findings,
  title = 'Visualizador interativo',
  subtitle = 'Achados integrados dos modelos com overlay e filtros inteligentes.',
  toolbar,
  className = '',
  showList = true,
  defaultTab = 'pathology',
  isLoading = false,
  loadingLabel = 'Processando analise...',
  enableToothFusionPreview = false,
  showModelToggles = false,
  enableClickSelect = false,
  enableDebug = false,
  showTeethToggle = true,
  structureLabel = 'Estruturas',
}: FindingsViewerProps) {
  const [activeTab, setActiveTab] = useState<ViewerTab>(defaultTab);
  const [selectedGroup, setSelectedGroup] = useState<string | null>(null);
  const [selectedFindingId, setSelectedFindingId] = useState<string | null>(null);
  const [showYolo, setShowYolo] = useState(true);
  const [showDetectron, setShowDetectron] = useState(true);
  const [showOther, setShowOther] = useState(true);
  const [showStructures, setShowStructures] = useState(true);
  const [showTeeth, setShowTeeth] = useState(true);
  const [showPathology, setShowPathology] = useState(true);
  const [showTreatments, setShowTreatments] = useState(true);
  const [showLabels, setShowLabels] = useState(false);
  const [imageReady, setImageReady] = useState(false);
  const [renderMetrics, setRenderMetrics] = useState<ReturnType<typeof getImageRenderMetrics> | null>(null);
  const [zoomStyle, setZoomStyle] = useState({ scale: 1, tx: 0, ty: 0 });
  const [debugEnabled, setDebugEnabled] = useState(enableDebug);
  const [debugInfo, setDebugInfo] = useState<Record<string, unknown> | null>(null);
  const [debugCopyStatus, setDebugCopyStatus] = useState<string | null>(null);
  const [coordTransform, setCoordTransform] = useState({
    yolo: { x: 1, y: 1, dx: 0, dy: 0 },
    detectron: { x: 1, y: 1, dx: 0, dy: 0 },
    other: { x: 1, y: 1, dx: 0, dy: 0 },
  });
  const imageRef = useRef<HTMLImageElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const stageRef = useRef<HTMLDivElement | null>(null);
  const highlightUntilRef = useRef(0);

  const normalizedFindings = useMemo(() => normalizeFindings(findings), [findings]);

  useEffect(() => {
    if (enableDebug) {
      setDebugEnabled(true);
      return;
    }
    const params = new URLSearchParams(window.location.search);
    const debugParam = params.get('debug');
    if (debugParam === '1' || debugParam === 'true' || params.has('debug') || params.has('debug1')) {
      setDebugEnabled(true);
    }
  }, [enableDebug]);

  useEffect(() => {
    const img = imageRef.current;
    if (!imageReady || !img) {
      setRenderMetrics(null);
      return;
    }
    const updateMetrics = () => {
      setRenderMetrics(getImageRenderMetrics(img));
    };
    updateMetrics();
    window.addEventListener('resize', updateMetrics);
    return () => window.removeEventListener('resize', updateMetrics);
  }, [imageReady, imageUrl]);

  useEffect(() => {
    setImageReady(false);
  }, [imageUrl]);

  useEffect(() => {
    const img = imageRef.current;
    if (!img || !imageUrl) {
      return;
    }
    const handleLoad = () => setImageReady(true);
    const handleError = () => setImageReady(false);
    if (img.complete && img.naturalWidth > 0) {
      setImageReady(true);
      return;
    }
    img.addEventListener('load', handleLoad);
    img.addEventListener('error', handleError);
    return () => {
      img.removeEventListener('load', handleLoad);
      img.removeEventListener('error', handleError);
    };
  }, [imageUrl]);

  useEffect(() => {
    setSelectedGroup(null);
  }, [activeTab, findings]);

  useEffect(() => {
    if (selectedGroup || selectedFindingId) {
      highlightUntilRef.current = performance.now() + 600;
    } else {
      highlightUntilRef.current = 0;
    }
  }, [selectedGroup, selectedFindingId]);

  useEffect(() => {
    const img = imageRef.current;
    if (!imageReady || !img || !img.naturalWidth || !img.naturalHeight) {
      setCoordTransform({
        yolo: { x: 1, y: 1, dx: 0, dy: 0 },
        detectron: { x: 1, y: 1, dx: 0, dy: 0 },
        other: { x: 1, y: 1, dx: 0, dy: 0 },
      });
      return;
    }
    const width = img.naturalWidth;
    const height = img.naturalHeight;
    const calcScale = (items: NormalizedFinding[]) => {
      let maxX = 0;
      let maxY = 0;
      items.forEach((finding) => {
        const bbox = getFindingBbox(finding);
        if (bbox) {
          maxX = Math.max(maxX, bbox[2]);
          maxY = Math.max(maxY, bbox[3]);
        }
        if (finding.segmentation?.length) {
          finding.segmentation.forEach(([x, y]) => {
            maxX = Math.max(maxX, x);
            maxY = Math.max(maxY, y);
          });
        }
      });
      if (maxX <= 0 || maxY <= 0) {
        return { x: 1, y: 1, dx: 0, dy: 0 };
      }
      if (maxX <= 1.5 && maxY <= 1.5) {
        return { x: width, y: height, dx: 0, dy: 0 };
      }
      return { x: 1, y: 1, dx: 0, dy: 0 };
    };

    setCoordTransform({
      yolo: calcScale(normalizedFindings.filter((finding) => finding.sourceKind === 'yolo')),
      detectron: calcScale(normalizedFindings.filter((finding) => finding.sourceKind === 'detectron')),
      other: calcScale(normalizedFindings.filter((finding) => finding.sourceKind === 'other')),
    });
  }, [imageReady, normalizedFindings]);

  const getScaleForFinding = (finding: NormalizedFinding) => {
    if (finding.sourceKind === 'yolo') {
      return coordTransform.yolo;
    }
    if (finding.sourceKind === 'detectron') {
      return coordTransform.detectron;
    }
    return coordTransform.other;
  };

  const getScaledBbox = (finding: NormalizedFinding) => {
    const bbox = getFindingBbox(finding);
    if (!bbox) {
      return undefined;
    }
    const scale = getScaleForFinding(finding);
    if (scale.x === 1 && scale.y === 1 && scale.dx === 0 && scale.dy === 0) {
      return bbox;
    }
    return [
      bbox[0] * scale.x + scale.dx,
      bbox[1] * scale.y + scale.dy,
      bbox[2] * scale.x + scale.dx,
      bbox[3] * scale.y + scale.dy,
    ];
  };

  const getScaledSegmentation = (finding: NormalizedFinding) => {
    if (!finding.segmentation?.length) {
      return undefined;
    }
    const scale = getScaleForFinding(finding);
    if (scale.x === 1 && scale.y === 1 && scale.dx === 0 && scale.dy === 0) {
      return finding.segmentation;
    }
    return finding.segmentation.map(([x, y]) => [x * scale.x + scale.dx, y * scale.y + scale.dy]);
  };

  const filteredFindings = useMemo(() => {
    return normalizedFindings.filter((finding) => {
      if (finding.sourceKind === 'yolo' && !showYolo) {
        return false;
      }
      if (finding.sourceKind === 'detectron' && !showDetectron) {
        return false;
      }
      if (finding.sourceKind === 'other' && !showOther) {
        return false;
      }
      if (!showStructures && (finding.category === 'structure' || finding.category === 'anatomy')) {
        return false;
      }
      if (!showTeeth && finding.category === 'tooth') {
        return false;
      }
      if (!showPathology && finding.category === 'pathology') {
        return false;
      }
      if (!showTreatments && finding.category === 'treatment') {
        return false;
      }
      return true;
    });
  }, [
    normalizedFindings,
    showYolo,
    showDetectron,
    showOther,
    showStructures,
    showTeeth,
    showPathology,
    showTreatments,
  ]);

  useEffect(() => {
    if (!debugEnabled) {
      setDebugInfo(null);
      return;
    }
    const updateDebug = () => {
      const img = imageRef.current;
      const canvas = canvasRef.current;
      const stage = stageRef.current;
      const metrics = img ? getImageRenderMetrics(img) : null;
      const bySource = { yolo: 0, detectron: 0, other: 0 };
      const byCategory: Record<string, number> = {};
      const labelCounts = new Map<string, number>();
      const pathologyBySource = { yolo: 0, detectron: 0, other: 0 };
      const treatmentBySource = { yolo: 0, detectron: 0, other: 0 };
      const carieBySource = { yolo: 0, detectron: 0, other: 0 };
      const restauracaoBySource = { yolo: 0, detectron: 0, other: 0 };
      let bboxCount = 0;
      let segmentationCount = 0;
      normalizedFindings.forEach((finding) => {
        bySource[finding.sourceKind] += 1;
        byCategory[finding.category] = (byCategory[finding.category] ?? 0) + 1;
        const labelKey = finding.displayLabel ?? finding.label;
        labelCounts.set(labelKey, (labelCounts.get(labelKey) ?? 0) + 1);
        if (finding.category === 'pathology') {
          pathologyBySource[finding.sourceKind] += 1;
        }
        if (finding.category === 'treatment') {
          treatmentBySource[finding.sourceKind] += 1;
        }
        const canonical = (finding.canonicalLabel ?? finding.label).toLowerCase();
        if (canonical.includes('carie')) {
          carieBySource[finding.sourceKind] += 1;
        }
        if (canonical.includes('restauracao')) {
          restauracaoBySource[finding.sourceKind] += 1;
        }
        if (finding.segmentation?.length) {
          segmentationCount += 1;
        }
        if (finding.bbox?.length === 4 || getFindingBbox(finding)) {
          bboxCount += 1;
        }
      });
      const topLabels = Array.from(labelCounts.entries())
        .sort((a, b) => b[1] - a[1])
        .slice(0, 8)
        .map(([label, count]) => ({ label, count }));
      const sample = filteredFindings[0] ?? normalizedFindings[0];
      setDebugInfo({
        imageReady,
        imageUrl: imageUrl ?? null,
        image: img
          ? {
              naturalWidth: img.naturalWidth,
              naturalHeight: img.naturalHeight,
              clientWidth: img.clientWidth,
              clientHeight: img.clientHeight,
            }
          : null,
        stage: stage ? { width: stage.clientWidth, height: stage.clientHeight } : null,
        canvas: canvas ? { width: canvas.width, height: canvas.height } : null,
        metrics,
        zoom: zoomStyle,
        coordTransform,
        filters: {
          showYolo,
          showDetectron,
          showOther,
          showStructures,
          showTeeth,
          showPathology,
          showTreatments,
        },
        counts: {
          total: normalizedFindings.length,
          filtered: filteredFindings.length,
          bboxCount,
          segmentationCount,
          bySource,
          byCategory,
        },
        summary: {
          topLabels,
          pathologyBySource,
          treatmentBySource,
          carieBySource,
          restauracaoBySource,
        },
        sample: sample
          ? {
              label: sample.displayLabel ?? sample.label,
              sourceKind: sample.sourceKind,
              category: sample.category,
              bbox: getFindingBbox(sample),
              segmentationPoints: sample.segmentation?.length ?? 0,
            }
          : null,
      });
    };
    updateDebug();
    const handleResize = () => updateDebug();
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [
    debugEnabled,
    imageReady,
    imageUrl,
    normalizedFindings,
    filteredFindings,
    zoomStyle,
    coordTransform,
    showYolo,
    showDetectron,
    showOther,
    showStructures,
    showTeeth,
    showPathology,
    showTreatments,
  ]);

  const handleCopyDebug = async () => {
    if (!debugInfo) {
      setDebugCopyStatus('Sem dados.');
      window.setTimeout(() => setDebugCopyStatus(null), 2000);
      return;
    }
    const payload = {
      generatedAt: new Date().toISOString(),
      ...debugInfo,
    };
    const text = JSON.stringify(payload, null, 2);
    try {
      await navigator.clipboard.writeText(text);
      setDebugCopyStatus('Debug copiado.');
    } catch (error) {
      const textarea = document.createElement('textarea');
      textarea.value = text;
      textarea.style.position = 'fixed';
      textarea.style.opacity = '0';
      document.body.appendChild(textarea);
      textarea.focus();
      textarea.select();
      const ok = document.execCommand('copy');
      document.body.removeChild(textarea);
      setDebugCopyStatus(ok ? 'Debug copiado.' : 'Falha ao copiar.');
    }
    window.setTimeout(() => setDebugCopyStatus(null), 2000);
  };

  const tabItems = useMemo(() => {
    const targetCategory =
      activeTab === 'pathology'
        ? 'pathology'
        : activeTab === 'treatment'
          ? 'treatment'
          : 'tooth';
    const items = filteredFindings.filter((finding) => finding.category === targetCategory);

    if (targetCategory === 'tooth') {
      const sorted = [...items].sort((a, b) => {
        const aBox = getScaledBbox(a);
        const bBox = getScaledBbox(b);
        const aX = aBox ? (aBox[0] + aBox[2]) / 2 : 0;
        const bX = bBox ? (bBox[0] + bBox[2]) / 2 : 0;
        return aX - bX;
      });
      const totals = new Map<string, number>();
      const counters = new Map<string, number>();
      sorted.forEach((finding) => {
        const key = finding.displayLabel;
        totals.set(key, (totals.get(key) ?? 0) + 1);
      });
      return sorted.map((finding) => {
        const key = finding.displayLabel;
        const order = (counters.get(key) ?? 0) + 1;
        counters.set(key, order);
        const total = totals.get(key) ?? 1;
        const suffix = total > 1 ? ` ${order}/${total}` : '';
        return {
          key: getGroupKey(finding),
          label: `${finding.displayLabel}${suffix}`,
          count: 1,
          confidence: finding.confidence,
        };
      });
    }

    const groups = new Map<
      string,
      { key: string; label: string; count: number; confidence?: number }
    >();
    items.forEach((finding) => {
      const key = getGroupKey(finding);
      const entry = groups.get(key) ?? {
        key,
        label: finding.displayLabel,
        count: 0,
        confidence: finding.confidence,
      };
      entry.count += 1;
      if (finding.confidence != null) {
        entry.confidence =
          entry.confidence == null ? finding.confidence : Math.max(entry.confidence, finding.confidence);
      }
      groups.set(key, entry);
    });
    return Array.from(groups.values()).sort((a, b) => {
      if (b.count !== a.count) {
        return b.count - a.count;
      }
      return a.label.localeCompare(b.label);
    });
  }, [filteredFindings, activeTab, coordTransform]);

  const selectedTarget = useMemo(() => {
    if (!enableToothFusionPreview) {
      return null;
    }
    if (selectedFindingId) {
      return filteredFindings.find((finding) => finding.id === selectedFindingId) ?? null;
    }
    if (!selectedGroup) {
      return null;
    }
    const candidates = filteredFindings.filter(
      (finding) => finding.category === 'tooth' && getGroupKey(finding) === selectedGroup,
    );
    if (candidates.length === 0) {
      return null;
    }
    return candidates.reduce((best, current) => {
      const bestBox = getScaledBbox(best);
      const currentBox = getScaledBbox(current);
      const bestArea = bestBox ? (bestBox[2] - bestBox[0]) * (bestBox[3] - bestBox[1]) : 0;
      const currentArea = currentBox ? (currentBox[2] - currentBox[0]) * (currentBox[3] - currentBox[1]) : 0;
      if (currentArea !== bestArea) {
        return currentArea > bestArea ? current : best;
      }
      const bestScore = best.confidence ?? 0;
      const currentScore = current.confidence ?? 0;
      return currentScore > bestScore ? current : best;
    }, candidates[0]);
  }, [enableToothFusionPreview, filteredFindings, selectedFindingId, selectedGroup, coordTransform]);

  const clickableFindings = useMemo(() => {
    const teeth = filteredFindings.filter(
      (finding) => finding.category === 'tooth' && getScaledBbox(finding),
    );
    if (teeth.length > 0) {
      return teeth;
    }
    return filteredFindings.filter(
      (finding) =>
        finding.category === 'structure' &&
        finding.canonicalLabel === 'esmalte' &&
        getScaledBbox(finding),
    );
  }, [filteredFindings, coordTransform]);

  const handleStageClick = (event: React.MouseEvent<HTMLDivElement>) => {
    if (!enableClickSelect || !imageReady) {
      return;
    }
    const stage = stageRef.current;
    const img = imageRef.current;
    if (!stage || !img) {
      return;
    }
    const stageRect = stage.getBoundingClientRect();
    const clickX = event.clientX - stageRect.left;
    const clickY = event.clientY - stageRect.top;

    const metrics = getImageRenderMetrics(img);
    if (!metrics) {
      return;
    }

    const offsetX = img.offsetLeft + metrics.offsetX;
    const offsetY = img.offsetTop + metrics.offsetY;
    const scaleX = metrics.renderWidth / metrics.naturalWidth;
    const scaleY = metrics.renderHeight / metrics.naturalHeight;
    const displayX = (clickX - offsetX - zoomStyle.tx) / zoomStyle.scale;
    const displayY = (clickY - offsetY - zoomStyle.ty) / zoomStyle.scale;
    const imageX = displayX / scaleX;
    const imageY = displayY / scaleY;

    if (clickableFindings.length === 0) {
      setSelectedGroup(null);
      setSelectedFindingId(null);
      return;
    }

    const inside = clickableFindings.filter((finding) => {
      const box = getScaledBbox(finding);
      if (!box) {
        return false;
      }
      return imageX >= box[0] && imageX <= box[2] && imageY >= box[1] && imageY <= box[3];
    });

    let chosen = inside[0];
    if (!chosen) {
      const sorted = [...clickableFindings].sort((a, b) => {
        const aBox = getScaledBbox(a);
        const bBox = getScaledBbox(b);
        if (!aBox || !bBox) {
          return 0;
        }
        const aCenter = bboxCenter(aBox);
        const bCenter = bboxCenter(bBox);
        const aDist = (aCenter.cx - imageX) ** 2 + (aCenter.cy - imageY) ** 2;
        const bDist = (bCenter.cx - imageX) ** 2 + (bCenter.cy - imageY) ** 2;
        return aDist - bDist;
      });
      chosen = sorted[0];
    }

    if (chosen) {
      setSelectedFindingId((prev) => (prev === chosen.id ? null : chosen.id));
      if (showList) {
        const nextKey = getGroupKey(chosen);
        setSelectedGroup((prev) => (prev === nextKey ? null : nextKey));
      } else {
        setSelectedGroup(null);
      }
    } else {
      setSelectedGroup(null);
      setSelectedFindingId(null);
    }
  };

  useEffect(() => {
    const stage = stageRef.current;
    const img = imageRef.current;
    if (!imageReady || !stage || !img || !selectedTarget) {
      setZoomStyle({ scale: 1, tx: 0, ty: 0 });
      return;
    }

    const updateZoom = () => {
      const toothBox = getScaledBbox(selectedTarget);
      if (!toothBox) {
        setZoomStyle({ scale: 1, tx: 0, ty: 0 });
        return;
      }

      const metrics = getImageRenderMetrics(img);
      if (!metrics) {
        setZoomStyle({ scale: 1, tx: 0, ty: 0 });
        return;
      }

      const offsetX = img.offsetLeft + metrics.offsetX;
      const offsetY = img.offsetTop + metrics.offsetY;
      const scaleX = metrics.renderWidth / metrics.naturalWidth;
      const scaleY = metrics.renderHeight / metrics.naturalHeight;
      const boxX = toothBox[0] * scaleX;
      const boxY = toothBox[1] * scaleY;
      const boxW = (toothBox[2] - toothBox[0]) * scaleX;
      const boxH = (toothBox[3] - toothBox[1]) * scaleY;
      if (boxW <= 0 || boxH <= 0) {
        setZoomStyle({ scale: 1, tx: 0, ty: 0 });
        return;
      }

      const stageW = stage.clientWidth;
      const stageH = stage.clientHeight;
      const targetW = stageW * 0.6;
      const targetH = stageH * 0.6;
      const scale = Math.min(3, Math.max(1, Math.min(targetW / boxW, targetH / boxH)));
      const centerX = boxX + boxW / 2;
      const centerY = boxY + boxH / 2;
      const tx = stageW / 2 - offsetX - centerX * scale;
      const ty = stageH / 2 - offsetY - centerY * scale;
      setZoomStyle({ scale, tx, ty });
    };

    updateZoom();
    const handleResize = () => updateZoom();
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [imageReady, selectedTarget]);

  useEffect(() => {
    const canvas = canvasRef.current;
    const img = imageRef.current;
    if (!canvas || !img || !imageReady) {
      return;
    }

    const draw = () => {
      const metrics = getImageRenderMetrics(img);
      if (!metrics) {
        return;
      }
      const offsetX = img.offsetLeft + metrics.offsetX;
      const offsetY = img.offsetTop + metrics.offsetY;
      canvas.style.left = `${offsetX}px`;
      canvas.style.top = `${offsetY}px`;
      canvas.style.width = `${metrics.renderWidth}px`;
      canvas.style.height = `${metrics.renderHeight}px`;
      canvas.width = metrics.renderWidth;
      canvas.height = metrics.renderHeight;
      const ctx = canvas.getContext('2d');
      if (!ctx) {
        return;
      }
      ctx.clearRect(0, 0, metrics.renderWidth, metrics.renderHeight);
      const scaleX = metrics.renderWidth / metrics.naturalWidth;
      const scaleY = metrics.renderHeight / metrics.naturalHeight;

      filteredFindings.forEach((finding) => {
        const style = getFindingStyle(finding);
        const fusionMatch =
          selectedTarget?.category === 'tooth' &&
          finding.category === 'structure' &&
          (() => {
            const toothBox = getScaledBbox(selectedTarget);
            const findingBox = getScaledBbox(finding);
            if (!toothBox || !findingBox) {
              return false;
            }
            const iou = bboxIou(toothBox, findingBox);
            if (iou >= 0.01) {
              return true;
            }
            const center = bboxCenter(findingBox);
            return (
              center.cx >= toothBox[0] &&
              center.cx <= toothBox[2] &&
              center.cy >= toothBox[1] &&
              center.cy <= toothBox[3]
            );
          })();
        const isActive = selectedFindingId
          ? finding.id === selectedFindingId || fusionMatch
          : selectedGroup && (getGroupKey(finding) === selectedGroup || fusionMatch);
        const isDimmed = (selectedGroup || selectedFindingId) && !isActive;
        const shouldShowLabel =
          showLabels ||
          (selectedFindingId ? finding.id === selectedFindingId : selectedGroup && getGroupKey(finding) === selectedGroup);
        const hideYoloBBox =
          finding.sourceKind === 'yolo' &&
          (finding.category === 'tooth' || finding.category === 'anatomy');
        const highlightRemaining = highlightUntilRef.current - performance.now();
        const highlightFactor = isActive ? Math.max(0, Math.min(1, highlightRemaining / 500)) : 0;
        ctx.globalAlpha = isDimmed ? 0.15 : 1;
        ctx.strokeStyle = style.stroke;
        ctx.fillStyle = hexToRgba(style.fill, highlightFactor * 0.4);
        ctx.lineWidth = isActive ? 3.5 : 2;
        ctx.shadowColor = isActive ? style.stroke : 'transparent';
        ctx.shadowBlur = isActive ? 8 : 0;
        ctx.lineJoin = 'round';
        ctx.lineCap = 'round';

        const allowYoloSegmentation = finding.sourceKind === 'yolo' && finding.category === 'anatomy';
        let segmentation = finding.segmentation?.length ? getScaledSegmentation(finding) : undefined;
        if (segmentation && allowYoloSegmentation) {
          const imageArea = (img.naturalWidth || 1) * (img.naturalHeight || 1);
          const minArea = Math.max(800, imageArea * 0.0005);
          segmentation =
            cleanSegmentation(segmentation, {
              minArea,
              minPoints: 16,
              minDistance: 2,
              smoothIterations: 2,
            }) ?? undefined;
        }

        if (segmentation?.length) {
          ctx.beginPath();
          segmentation.forEach((point, index) => {
            const x = point[0] * scaleX;
            const y = point[1] * scaleY;
            if (index === 0) {
              ctx.moveTo(x, y);
            } else {
              ctx.lineTo(x, y);
            }
          });
          ctx.closePath();
          ctx.stroke();
          if (highlightFactor > 0) {
            ctx.fill();
          }
        } else if (finding.bbox?.length === 4 && !hideYoloBBox) {
          const scaledBox = getScaledBbox(finding);
          if (!scaledBox) {
            ctx.globalAlpha = 1;
            ctx.shadowBlur = 0;
            return;
          }
          const [x1, y1, x2, y2] = scaledBox;
          const width = (x2 - x1) * scaleX;
          const height = (y2 - y1) * scaleY;
          const radius = Math.max(2, Math.min(10, Math.min(width, height) * 0.12));
          drawRoundedRect(ctx, x1 * scaleX, y1 * scaleY, width, height, radius);
        }

        if (shouldShowLabel) {
          const label = finding.displayLabel ?? finding.label;
          const sourceShort =
            finding.sourceKind === 'detectron' ? 'D' : finding.sourceKind === 'yolo' ? 'Y' : 'O';
          const toothText = finding.toothId != null ? `·${finding.toothId}` : finding.toothType ? `·${finding.toothType}` : '';
          const labelText = `${label}${toothText}·${sourceShort}`;

          let labelX = 8;
          let labelY = 14;
          const bboxForLabel = getScaledBbox(finding);
          if (bboxForLabel?.length === 4) {
            const [x1, y1] = bboxForLabel;
            labelX = x1 * scaleX + 4;
            labelY = y1 * scaleY + 14;
          } else if (segmentation?.length) {
            const [x, y] = segmentation[0];
            labelX = x * scaleX + 4;
            labelY = y * scaleY + 14;
          }

          ctx.font = '11px ui-sans-serif, system-ui, -apple-system, sans-serif';
          const metrics = ctx.measureText(labelText);
          const padding = 3;
          const textWidth = metrics.width + padding * 2;
          const textHeight = 14;
          ctx.fillStyle = 'rgba(15, 23, 42, 0.65)';
          ctx.fillRect(labelX, labelY - textHeight, textWidth, textHeight);
          ctx.fillStyle = '#f8fafc';
          ctx.fillText(labelText, labelX + padding, labelY - 3);
        }
        ctx.globalAlpha = 1;
        ctx.shadowBlur = 0;
      });

      if (debugEnabled) {
        ctx.save();
        ctx.strokeStyle = 'rgba(248, 113, 113, 0.9)';
        ctx.lineWidth = 1;
        ctx.strokeRect(0.5, 0.5, metrics.renderWidth - 1, metrics.renderHeight - 1);
        ctx.restore();
      }
    };

    draw();
    let raf = 0;
    const animate = () => {
      if (highlightUntilRef.current > performance.now()) {
        draw();
        raf = requestAnimationFrame(animate);
      }
    };
    if (highlightUntilRef.current > performance.now()) {
      raf = requestAnimationFrame(animate);
    }
    const handleResize = () => draw();
    window.addEventListener('resize', handleResize);
    return () => {
      window.removeEventListener('resize', handleResize);
      if (raf) {
        cancelAnimationFrame(raf);
      }
    };
  }, [
    filteredFindings,
    imageReady,
    selectedGroup,
    selectedFindingId,
    showLabels,
    selectedTarget,
    coordTransform,
  ]);

  const emptyState = !imageUrl;
  const debugSummary = debugInfo?.summary as DebugSummary | undefined;

  return (
    <div
      className={[
        'rounded-3xl border border-white/20 bg-white/60 p-6 shadow-xl dark:border-white/10 dark:bg-white/5',
        className,
      ].join(' ')}
    >
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <p className="text-xs uppercase tracking-[0.3em] text-brand-blue/60 dark:text-white/60">
            Visualizador
          </p>
          <h2 className="mt-2 text-xl font-semibold text-brand-blue dark:text-white">{title}</h2>
          <p className="mt-1 text-xs text-slate-500 dark:text-slate-400">{subtitle}</p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          {toolbar}
        </div>
      </div>

      <div className="mt-4 flex flex-wrap items-center gap-2 text-[11px] uppercase tracking-wider text-brand-blue/70 dark:text-white/70">
        {showModelToggles ? (
          <>
            <span className="text-[10px] font-semibold text-brand-blue/60 dark:text-white/50">
              Modelos
            </span>
            <button type="button" className={toggleButtonClass(showYolo)} onClick={() => setShowYolo((prev) => !prev)}>
              YOLO
            </button>
            <button
              type="button"
              className={toggleButtonClass(showDetectron)}
              onClick={() => setShowDetectron((prev) => !prev)}
            >
              Detectron
            </button>
            <button type="button" className={toggleButtonClass(showOther)} onClick={() => setShowOther((prev) => !prev)}>
              Outros
            </button>
          </>
        ) : null}
        <span className={`${showModelToggles ? 'ml-2' : ''} text-[10px] font-semibold text-brand-blue/60 dark:text-white/50`}>
          Exibir
        </span>
        <button
          type="button"
          className={toggleButtonClass(showPathology)}
          onClick={() => setShowPathology((prev) => !prev)}
        >
          Patologias
        </button>
        <button
          type="button"
          className={toggleButtonClass(showTreatments)}
          onClick={() => setShowTreatments((prev) => !prev)}
        >
          Tratamentos
        </button>
        {showTeethToggle ? (
          <button
            type="button"
            className={toggleButtonClass(showTeeth)}
            onClick={() => setShowTeeth((prev) => !prev)}
          >
            Dentes
          </button>
        ) : null}
        <button
          type="button"
          className={toggleButtonClass(showStructures)}
          onClick={() => setShowStructures((prev) => !prev)}
        >
          {structureLabel}
        </button>
        <button
          type="button"
          className={toggleButtonClass(showLabels)}
          onClick={() => setShowLabels((prev) => !prev)}
        >
          Labels
        </button>
      </div>

      <div className={`mt-4 grid gap-4 ${showList ? 'lg:grid-cols-[1.5fr_1fr]' : ''}`}>
        <div
          ref={stageRef}
          onClick={handleStageClick}
          className={[
            'relative overflow-hidden rounded-2xl border border-white/40 bg-white/70 dark:border-white/10 dark:bg-slate-900/60 min-h-[360px] lg:h-[560px] flex items-center justify-center',
            enableClickSelect ? 'cursor-zoom-in' : '',
          ].join(' ')}
        >
          {imageUrl ? (
            <>
              <img
                ref={imageRef}
                src={imageUrl}
                alt="Radiografia"
                onLoad={() => setImageReady(true)}
                className="max-h-[560px] w-full object-contain image-raw"
                style={{
                  transform: `translate(${zoomStyle.tx}px, ${zoomStyle.ty}px) scale(${zoomStyle.scale})`,
                  transformOrigin: renderMetrics
                    ? `${renderMetrics.offsetX}px ${renderMetrics.offsetY}px`
                    : 'top left',
                }}
              />
              <canvas
                ref={canvasRef}
                className="absolute pointer-events-none"
                style={{
                  transform: `translate(${zoomStyle.tx}px, ${zoomStyle.ty}px) scale(${zoomStyle.scale})`,
                  transformOrigin: 'top left',
                }}
              />
              {selectedTarget ? (
                <div className="absolute left-4 top-4 rounded-2xl border border-white/30 bg-white/80 px-3 py-2 text-[11px] font-semibold text-brand-blue shadow-sm dark:border-white/10 dark:bg-slate-900/70 dark:text-white">
                  {selectedTarget.displayLabel}
                  {selectedTarget.confidence != null ? (
                    <span className="ml-2 text-[10px] uppercase tracking-wider text-slate-500 dark:text-slate-400">
                      {(selectedTarget.confidence * 100).toFixed(1)}%
                    </span>
                  ) : null}
                </div>
              ) : null}
              {enableClickSelect && (selectedGroup || selectedFindingId) ? (
                <button
                  type="button"
                  onClick={(event) => {
                    event.stopPropagation();
                    setSelectedGroup(null);
                    setSelectedFindingId(null);
                  }}
                  className="absolute right-4 top-4 rounded-full border border-white/40 bg-white/80 px-3 py-1 text-[10px] font-semibold uppercase tracking-wider text-brand-blue shadow-sm transition hover:bg-white dark:border-white/10 dark:bg-slate-900/70 dark:text-white dark:hover:bg-slate-900"
                >
                  Voltar ao zoom
                </button>
              ) : null}
              {isLoading ? (
                <div className="absolute inset-0 flex items-center justify-center bg-white/70 dark:bg-slate-900/70">
                  <div className="flex items-center gap-3 rounded-full bg-white/80 px-4 py-2 text-xs font-semibold uppercase tracking-wider text-brand-blue shadow-lg dark:bg-white/10 dark:text-white">
                    <span className="h-4 w-4 animate-spin rounded-full border-2 border-brand-green border-t-transparent"></span>
                    {loadingLabel}
                  </div>
                </div>
              ) : null}
              {debugEnabled ? (
                <div className="absolute bottom-4 left-4 max-w-[340px] rounded-2xl border border-white/40 bg-white/90 px-3 py-2 text-[10px] text-slate-700 shadow-lg dark:border-white/10 dark:bg-slate-900/80 dark:text-slate-200">
                  <div className="flex items-center justify-between gap-2">
                    <span className="text-[11px] font-semibold uppercase tracking-wider text-brand-blue dark:text-white">
                      Debug
                    </span>
                    <button
                      type="button"
                      onClick={handleCopyDebug}
                      className="rounded-full border border-brand-blue/40 px-2 py-0.5 text-[10px] font-semibold text-brand-blue hover:bg-brand-blue/10 dark:border-white/20 dark:text-white/80 dark:hover:bg-white/10"
                    >
                      Copiar
                    </button>
                  </div>
                  <div className="mt-2 space-y-1 font-mono text-[10px] leading-4">
                    <div>imageReady: {debugInfo?.imageReady ? 'true' : 'false'}</div>
                    <div>
                      image: {debugInfo?.image && typeof debugInfo.image === 'object'
                        ? `${(debugInfo.image as { clientWidth: number; clientHeight: number }).clientWidth}x${(debugInfo.image as { clientWidth: number; clientHeight: number }).clientHeight}`
                        : 'n/a'}
                    </div>
                    <div>
                      natural: {debugInfo?.image && typeof debugInfo.image === 'object'
                        ? `${(debugInfo.image as { naturalWidth: number; naturalHeight: number }).naturalWidth}x${(debugInfo.image as { naturalWidth: number; naturalHeight: number }).naturalHeight}`
                        : 'n/a'}
                    </div>
                    <div>
                      render: {debugInfo?.metrics && typeof debugInfo.metrics === 'object'
                        ? `${(debugInfo.metrics as { renderWidth: number; renderHeight: number }).renderWidth}x${(debugInfo.metrics as { renderWidth: number; renderHeight: number }).renderHeight}`
                        : 'n/a'}
                    </div>
                    <div>
                      offset: {debugInfo?.metrics && typeof debugInfo.metrics === 'object'
                        ? `${(debugInfo.metrics as { offsetX: number; offsetY: number }).offsetX.toFixed(1)},${(debugInfo.metrics as { offsetX: number; offsetY: number }).offsetY.toFixed(1)}`
                        : 'n/a'}
                    </div>
                    <div>
                      canvas: {debugInfo?.canvas && typeof debugInfo.canvas === 'object'
                        ? `${(debugInfo.canvas as { width: number; height: number }).width}x${(debugInfo.canvas as { width: number; height: number }).height}`
                        : 'n/a'}
                    </div>
                    <div>
                      findings: {debugInfo?.counts && typeof debugInfo.counts === 'object'
                        ? `${(debugInfo.counts as { filtered: number; total: number }).filtered}/${(debugInfo.counts as { filtered: number; total: number }).total}`
                        : 'n/a'}
                    </div>
                    <div>
                      seg/bbox: {debugInfo?.counts && typeof debugInfo.counts === 'object'
                        ? `${(debugInfo.counts as { segmentationCount: number; bboxCount: number }).segmentationCount}/${(debugInfo.counts as { segmentationCount: number; bboxCount: number }).bboxCount}`
                        : 'n/a'}
                    </div>
                    {debugSummary ? (
                      <div>
                        carie Y/D/O: {debugSummary.carieBySource.yolo}/{debugSummary.carieBySource.detectron}/{debugSummary.carieBySource.other}
                      </div>
                    ) : null}
                    {debugSummary ? (
                      <div>
                        restauracao Y/D/O: {debugSummary.restauracaoBySource.yolo}/{debugSummary.restauracaoBySource.detectron}/{debugSummary.restauracaoBySource.other}
                      </div>
                    ) : null}
                    {debugInfo?.sample && typeof debugInfo.sample === 'object' ? (
                      <div>
                        sample: {(debugInfo.sample as { label: string }).label}
                      </div>
                    ) : null}
                    {debugCopyStatus ? (
                      <div className="text-[10px] uppercase tracking-wider text-brand-green">
                        {debugCopyStatus}
                      </div>
                    ) : null}
                  </div>
                </div>
              ) : null}
            </>
          ) : (
            <div className="text-center text-xs text-slate-500 dark:text-slate-400">
              Nenhuma radiografia carregada.
            </div>
          )}
        </div>

        {showList ? (
          <div className="rounded-2xl border border-white/30 bg-white/70 p-4 text-sm text-slate-600 dark:border-white/10 dark:bg-slate-900/40 dark:text-slate-200">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div>
                <p className="text-xs font-semibold uppercase tracking-wider text-brand-blue dark:text-white">
                  Lista de achados
                </p>
                <p className="mt-1 text-[11px] text-slate-500 dark:text-slate-400">
                  Total visivel: {filteredFindings.length}
                </p>
              </div>
              <div className="flex items-center gap-1 rounded-full bg-white/70 p-1 dark:bg-white/10">
                <button
                  type="button"
                  onClick={() => setActiveTab('pathology')}
                  className={tabButtonClass(activeTab === 'pathology')}
                >
                  Patologias
                </button>
                <button
                  type="button"
                  onClick={() => setActiveTab('treatment')}
                  className={tabButtonClass(activeTab === 'treatment')}
                >
                  Tratamentos
                </button>
                <button
                  type="button"
                  onClick={() => setActiveTab('tooth')}
                  className={tabButtonClass(activeTab === 'tooth')}
                >
                  Dentes
                </button>
              </div>
            </div>

            <div className="mt-4 max-h-[420px] lg:max-h-[560px] overflow-y-auto pr-1 space-y-2">
              {tabItems.length === 0 ? (
                <div className="rounded-xl border border-white/30 bg-white/60 p-3 text-xs text-slate-500 dark:border-white/10 dark:bg-slate-900/40 dark:text-slate-300">
                  {emptyState
                    ? 'Carregue uma radiografia para listar os achados.'
                    : 'Sem achados visiveis para este filtro.'}
                </div>
              ) : (
                tabItems.map((item) => (
                  <button
                    key={item.key}
                    type="button"
                    onClick={() => {
                      setSelectedFindingId(null);
                      setSelectedGroup((prev) => (prev === item.key ? null : item.key));
                    }}
                    className={[
                      'w-full text-left rounded-xl border px-3 py-2 transition-all',
                      selectedGroup === item.key
                        ? 'border-brand-green/60 bg-brand-green/10 text-brand-blue dark:text-white'
                        : 'border-white/40 bg-white/60 text-slate-600 hover:border-brand-blue/40 dark:border-white/10 dark:bg-slate-900/30 dark:text-slate-200',
                    ].join(' ')}
                  >
                    <div className="flex items-center justify-between">
                      <span className="font-semibold">{item.label}</span>
                      <span className="text-[10px] uppercase tracking-wider text-slate-500 dark:text-slate-400">
                        {item.count}x
                      </span>
                    </div>
                    {item.confidence != null ? (
                      <div className="mt-1 text-[10px] uppercase tracking-wider text-slate-500 dark:text-slate-400">
                        Confiança máx: {(item.confidence * 100).toFixed(1)}%
                      </div>
                    ) : null}
                  </button>
                ))
              )}
            </div>
          </div>
        ) : null}
      </div>
    </div>
  );
}
