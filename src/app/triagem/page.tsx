'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import {
  ArrowLeft,
  BarChart3,
  CheckCircle2,
  FileSearch,
  Filter,
  LogOut,
  Moon,
  Sun,
  UploadCloud,
  UserRound,
  X,
} from 'lucide-react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { useSupabaseUser } from '@/hooks/use-supabase-user';
import { supabase } from '@/lib/supabase/client';
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover';

type ModelType = 'yolo' | 'mask_rcnn' | 'combined';

type FindingCategory =
  | 'structure'
  | 'tooth'
  | 'pathology'
  | 'anatomy'
  | 'treatment'
  | 'other';

type LabelMeta = {
  display: string;
  canonical: string;
  category: FindingCategory;
  toothGroup?: string;
};

type Finding = {
  label: string;
  confidence?: number;
  bbox?: number[];
  segmentation?: number[][];
  model?: string;
  sourceModel?: string;
  toothId?: number;
  displayLabel?: string;
  canonicalLabel?: string;
  category?: FindingCategory;
  toothType?: string;
  toothGroup?: string;
  depth?: 'Esmalte' | 'Dentina' | 'Polpa';
};

type CaseStatus = 'pending' | 'analyzing' | 'done' | 'error';

type TriageCase = {
  id: string;
  name: string;
  file?: File | null;
  previewUrl?: string | null;
  status: CaseStatus;
  findings: Finding[];
  error?: string;
  modelType: ModelType;
  source: 'upload' | 'json';
};

type CaseInsight = {
  severity: 'Alta' | 'Media' | 'Baixa' | 'Aguardando';
  score: number;
  needs: string[];
  topFindings: string[];
  visibleFindings: Finding[];
  normalizedFindings: Finding[];
};

type TriageCaseWithInsights = TriageCase & {
  insights: CaseInsight;
  displayName: string;
  caseNumber: number;
  patientName: string;
  clinicalProfile: ClinicalProfile;
};

type ClinicalProfile = {
  name: string;
  age: number;
  gender: 'Feminino' | 'Masculino' | 'Outros';
  complaint: string;
  symptoms: string[];
  alerts: string[];
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
  tto_endo: {
    display: 'Tratamento endodôntico',
    canonical: 'tto_endo',
    category: 'treatment',
  },
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

const STRUCTURE_CATEGORIES = new Set<FindingCategory>(['structure', 'anatomy']);

const PRIORITY_RULES: Record<string, { weight: number; need: string }> = {
  periapical_lesion: { weight: 3.4, need: 'Endodontia' },
  carie_profunda: { weight: 3.2, need: 'Endodontia' },
  raiz_residual: { weight: 3.0, need: 'Cirurgia' },
  impactado: { weight: 2.6, need: 'Cirurgia' },
  carie_dentina: { weight: 2.2, need: 'Restauradora' },
  carie_esmalte: { weight: 1.4, need: 'Restauradora' },
  carie: { weight: 1.8, need: 'Restauradora' },
  tto_endo: { weight: 1.4, need: 'Endodontia' },
  restauracao: { weight: 0.6, need: 'Reabilitação' },
};

const TOOTH_EXPECTATIONS: Record<string, number> = {
  incisivo: 8,
  canino: 4,
  pre_molar: 8,
  molar: 8,
  terceiro_molar: 4,
};

const TOOTH_GROUP_LABELS: Record<string, string> = {
  incisivo: 'Incisivos',
  canino: 'Caninos',
  pre_molar: 'Pré-molares',
  molar: 'Molares',
  terceiro_molar: '3º molares',
  outros: 'Outros',
};

const SEVERITY_RULES = [
  {
    keywords: ['abscesso', 'cisto', 'fratura', 'lesao', 'lesão', 'reabsorcao', 'granuloma', 'tumor'],
    weight: 3,
    need: 'Cirurgia',
  },
  {
    keywords: ['carie', 'cárie', 'periodont', 'perda ossea', 'periapical', 'infeccao', 'infecção'],
    weight: 2,
    need: 'Periodontal',
  },
  {
    keywords: ['canal', 'endodont', 'polpa', 'apice', 'ápice', 'raiz'],
    weight: 1.5,
    need: 'Endodontia',
  },
  {
    keywords: ['restaur', 'obtur', 'protese', 'prótese', 'implante'],
    weight: 1,
    need: 'Reabilitação',
  },
  {
    keywords: ['inclus', 'retido', 'impactado', 'erupcao', 'erupção'],
    weight: 1.5,
    need: 'Cirurgia',
  },
];

const isAnatomyLabel = (label: string) => {
  const normalized = label.toLowerCase();
  return ANATOMY_KEYWORDS.some((keyword) => normalized.includes(keyword));
};

const coerceModelType = (value: unknown, fallback: ModelType): ModelType => {
  if (value === 'yolo') {
    return 'yolo';
  }
  if (value === 'mask_rcnn') {
    return 'mask_rcnn';
  }
  if (value === 'combined') {
    return 'combined';
  }
  return fallback;
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

const hashSeed = (value: string) => {
  let hash = 0;
  for (let i = 0; i < value.length; i += 1) {
    hash = (hash << 5) - hash + value.charCodeAt(i);
    hash |= 0;
  }
  return Math.abs(hash);
};

const pickUnique = (items: string[], seed: number, count: number) => {
  const chosen = new Set<string>();
  let i = 0;
  while (chosen.size < Math.min(count, items.length) && i < items.length * 3) {
    const idx = (seed + i * 7) % items.length;
    chosen.add(items[idx]);
    i += 1;
  }
  return Array.from(chosen);
};

const pickPatientName = (seed: number) => (seed % 2 === 0 ? 'Maria da Silva' : 'João da Silva');

const buildClinicalProfile = (caseItem: { name: string; insights: CaseInsight }): ClinicalProfile => {
  const seed = hashSeed(caseItem.name);
  const name = pickPatientName(seed);
  const age = 18 + (seed % 60);
  const genders: ClinicalProfile['gender'][] = ['Feminino', 'Masculino', 'Outros'];
  const gender = genders[seed % genders.length];

  const complaints = [
    'Rotina / check-up',
    'Dor localizada',
    'Sensibilidade ao frio',
    'Dente fraturado',
    'Sangramento gengival',
    'Inchaço facial',
    'Reavaliação endodôntica',
  ];
  const routineOnly = seed % 5 === 0;
  const complaint = routineOnly
    ? complaints[0]
    : seed % 10 < 6
      ? complaints[0]
      : complaints[1 + (seed % (complaints.length - 1))];

  const symptomsPool = [
    'Dor espontânea',
    'Sensibilidade ao frio',
    'Sensibilidade ao quente',
    'Sangramento gengival',
    'Halitose',
    'Mobilidade dental',
    'Fístula',
    'Inchaço local',
    'Dor à mastigação',
  ];
  const alertsPool = ['Inchaço facial', 'Dor intensa', 'Febre relatada'];

  const baseCount =
    caseItem.insights.severity === 'Alta'
      ? 3
      : caseItem.insights.severity === 'Media'
        ? 2
        : 1;

  const symptoms = routineOnly ? [] : pickUnique(symptomsPool, seed, baseCount);
  const alerts =
    caseItem.insights.severity === 'Alta' && !routineOnly
      ? pickUnique(alertsPool, seed + 11, 2)
      : [];

  return {
    name,
    age,
    gender,
    complaint,
    symptoms,
    alerts,
  };
};

const buildCaseDisplayName = (index: number) => `Paciente ${index + 1}`;

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
    category: isAnatomyLabel(label) ? 'anatomy' : 'other',
  };
};

const formatToothTag = (label: string) =>
  label
    .replace('inferior', 'inf')
    .replace('superior', 'sup')
    .replace('Pré-molar', 'Pré-mol.')
    .replace('Incisivo', 'Inc.')
    .replace('Molar', 'Molar');

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

const getFindingBbox = (finding: Finding) => {
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

const attachToothMatches = (findings: Finding[]) => {
  const toothCandidates = findings
    .filter(
      (finding) =>
        finding.category === 'tooth' && getSourceKind(finding) === 'yolo' && getFindingBbox(finding),
    )
    .map((finding) => {
      const bbox = getFindingBbox(finding) as number[];
      const { cx, cy } = bboxCenter(bbox);
      return {
        bbox,
        cx,
        cy,
        toothLabel: finding.displayLabel ?? finding.label,
        toothGroup: finding.toothGroup,
      };
    });

  if (toothCandidates.length === 0) {
    return findings;
  }

  return findings.map((finding) => {
    if (!['structure', 'pathology', 'treatment'].includes(finding.category ?? '')) {
      return finding;
    }
    const bbox = getFindingBbox(finding);
    if (!bbox) {
      return finding;
    }
    let best = null as null | { toothLabel: string; toothGroup?: string; score: number; dist: number };
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
    if (best.score < 0.02) {
      return { ...finding, toothType: best.toothLabel, toothGroup: best.toothGroup };
    }
    return { ...finding, toothType: best.toothLabel, toothGroup: best.toothGroup };
  });
};

const applyCariesDepth = (findings: Finding[], normalizeLabels: boolean) => {
  const dentinaBoxes = findings
    .filter((finding) => finding.canonicalLabel === 'dentina')
    .map((finding) => getFindingBbox(finding))
    .filter((bbox): bbox is number[] => Boolean(bbox));
  const polpaBoxes = findings
    .filter((finding) => finding.canonicalLabel === 'polpa')
    .map((finding) => getFindingBbox(finding))
    .filter((bbox): bbox is number[] => Boolean(bbox));

  return findings.map((finding) => {
    if (finding.canonicalLabel !== 'carie' || getSourceKind(finding) !== 'detectron') {
      return finding;
    }
    const bbox = getFindingBbox(finding);
    if (!bbox) {
      return finding;
    }
    const maxPolpa = polpaBoxes.reduce((max, box) => Math.max(max, bboxIou(bbox, box)), 0);
    const maxDentina = dentinaBoxes.reduce((max, box) => Math.max(max, bboxIou(bbox, box)), 0);
    let depth: Finding['depth'] = 'Esmalte';
    let canonical = 'carie_esmalte';
    let display = 'Cárie em esmalte';
    if (maxPolpa >= 0.02) {
      depth = 'Polpa';
      canonical = 'carie_profunda';
      display = 'Cárie profunda';
    } else if (maxDentina >= 0.02) {
      depth = 'Dentina';
      canonical = 'carie_dentina';
      display = 'Cárie em dentina';
    }
    return {
      ...finding,
      depth,
      canonicalLabel: canonical,
      displayLabel: normalizeLabels ? display : finding.displayLabel,
    };
  });
};

const enrichFindings = (findings: Finding[], normalizeLabels: boolean) => {
  const enriched = findings.map((finding) => {
    const meta = getLabelMeta(finding.label);
    return {
      ...finding,
      displayLabel: normalizeLabels ? meta.display : finding.label,
      canonicalLabel: meta.canonical,
      category: meta.category,
      toothGroup: meta.toothGroup,
      toothType: meta.category === 'tooth' ? meta.display : finding.toothType,
    };
  });
  const withTeeth = attachToothMatches(enriched);
  return applyCariesDepth(withTeeth, normalizeLabels);
};

const normalizeFinding = (raw: Record<string, unknown>): Finding | null => {
  const labelCandidate =
    raw.label ?? raw.class ?? raw.name ?? raw.category ?? raw.tipo ?? raw.tipo_lesao;
  if (labelCandidate == null) {
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
    ? (raw.bbox as number[])
    : Array.isArray(raw.box)
      ? (raw.box as number[])
      : Array.isArray(raw.bounding_box)
        ? (raw.bounding_box as number[])
        : undefined;
  const segmentation = Array.isArray(raw.segmentation)
    ? (raw.segmentation as number[][])
    : Array.isArray(raw.polygon)
      ? (raw.polygon as number[][])
      : typeof raw.segmentation === 'string'
        ? parseSvgPath(raw.segmentation)
        : undefined;
  const sourceModel =
    (raw.source_model as string | undefined) ??
    (raw.sourceModel as string | undefined);
  const model =
    sourceModel ??
    (raw.model as string | undefined) ??
    (raw.model_type as string | undefined) ??
    (raw.source as string | undefined);
  const toothId = toNumber((raw.tooth_id ?? raw.toothId) as unknown);

  return {
    label,
    confidence,
    bbox,
    segmentation,
    model,
    sourceModel,
    toothId,
  };
};

const extractCaseArray = (payload: unknown) => {
  if (Array.isArray(payload)) {
    return payload;
  }
  if (payload && typeof payload === 'object') {
    const container = payload as Record<string, unknown>;
    const candidates = ['cases', 'images', 'items', 'results', 'data', 'samples'];
    for (const key of candidates) {
      if (Array.isArray(container[key])) {
        return container[key] as unknown[];
      }
    }
    if (Array.isArray(container.findings)) {
      return [payload];
    }
  }
  return [];
};

const normalizeCaseFromJson = (
  raw: Record<string, unknown>,
  index: number,
  fallbackModel: ModelType,
  imageBaseUrl?: string,
): TriageCase => {
  const findingsRaw =
    raw.findings ??
    raw.results ??
    raw.detections ??
    (raw.analysis as Record<string, unknown> | undefined)?.findings ??
    raw.predictions;
  const findingsArray = Array.isArray(findingsRaw) ? findingsRaw : [];
  const findings = findingsArray
    .map((item) => (item && typeof item === 'object' ? normalizeFinding(item as Record<string, unknown>) : null))
    .filter((item): item is Finding => Boolean(item));

  const nameCandidate =
    raw.name ?? raw.filename ?? raw.file ?? raw.id ?? raw.uid ?? `Caso ${index + 1}`;
  const modelType = coerceModelType(raw.modelType ?? raw.model_type ?? raw.model, fallbackModel);
  const previewUrl =
    (raw.previewUrl as string | undefined) ??
    (raw.preview_url as string | undefined) ??
    (raw.imageUrl as string | undefined) ??
    (raw.image_url as string | undefined) ??
    (raw.image as string | undefined);
  const baseUrl = imageBaseUrl
    ? imageBaseUrl.endsWith('/') ? imageBaseUrl : `${imageBaseUrl}/`
    : undefined;
  const resolvedPreview =
    typeof previewUrl === 'string'
      ? previewUrl
      : baseUrl
        ? `${baseUrl}${encodeURIComponent(String(nameCandidate))}`
        : null;

  return {
    id: `${String(nameCandidate)}-${index}-${Math.random().toString(36).slice(2, 8)}`,
    name: String(nameCandidate),
    file: null,
    previewUrl: resolvedPreview,
    status: 'done',
    findings,
    modelType,
    source: 'json',
  };
};

const isHiddenStructure = (finding: Finding) => {
  if (finding.category) {
    return STRUCTURE_CATEGORIES.has(finding.category) || finding.category === 'tooth';
  }
  return isAnatomyLabel(finding.label);
};

const scoreSeverity = (score: number) => {
  if (score >= 3.5) {
    return 'Alta';
  }
  if (score >= 1.6) {
    return 'Media';
  }
  return 'Baixa';
};

const toPercent = (value: number, max: number) => {
  if (max <= 0) {
    return 0;
  }
  return Math.round((value / max) * 100);
};

const getFindingStyle = (finding: Finding) => {
  const label = (finding.canonicalLabel ?? finding.label).toLowerCase();
  if (finding.category === 'pathology' || label.includes('carie') || label.includes('lesao')) {
    return { stroke: '#f87171', fill: 'rgba(248, 113, 113, 0.2)' };
  }
  if (finding.category === 'treatment') {
    return { stroke: '#fbbf24', fill: 'rgba(251, 191, 36, 0.2)' };
  }
  if (finding.category === 'structure') {
    return { stroke: '#38bdf8', fill: 'rgba(56, 189, 248, 0.18)' };
  }
  if (finding.category === 'tooth') {
    return { stroke: '#4ade80', fill: 'rgba(74, 222, 128, 0.2)' };
  }
  return { stroke: '#94a3b8', fill: 'rgba(148, 163, 184, 0.2)' };
};

const getSourceKind = (finding: Finding) => {
  const source = (finding.sourceModel ?? finding.model ?? '').toLowerCase();
  if (source.includes('yolo')) {
    return 'yolo';
  }
  if (source.includes('detectron')) {
    return 'detectron';
  }
  return 'other';
};

const formatModelType = (value: ModelType) => {
  if (value === 'combined') {
    return 'Combinado';
  }
  if (value === 'mask_rcnn') {
    return 'Mask R-CNN';
  }
  return 'YOLOv11';
};

export default function TriagemPage() {
  const router = useRouter();
  const { user, isLoading } = useSupabaseUser();
  const [tokenChecked, setTokenChecked] = useState(false);
  const [hasTokenAccess, setHasTokenAccess] = useState(false);
  const [isDarkMode, setIsDarkMode] = useState(false);
  const [modelType, setModelType] = useState<ModelType>('yolo');
  const [hideAnatomy, setHideAnatomy] = useState(true);
  const [normalizeToothLabel, setNormalizeToothLabel] = useState(true);
  const [cases, setCases] = useState<TriageCase[]>([]);
  const [isBatchAnalyzing, setIsBatchAnalyzing] = useState(false);
  const [batchError, setBatchError] = useState<string | null>(null);
  const [batchProgress, setBatchProgress] = useState({ completed: 0, total: 0 });
  const [summary, setSummary] = useState('');
  const [summaryError, setSummaryError] = useState<string | null>(null);
  const [isSummaryLoading, setIsSummaryLoading] = useState(false);
  const [importError, setImportError] = useState<string | null>(null);
  const [isSampleLoading, setIsSampleLoading] = useState(false);
  const [isAutoBatchLoading, setIsAutoBatchLoading] = useState(false);
  const [uploadPassword, setUploadPassword] = useState('');
  const [selectedCaseId, setSelectedCaseId] = useState<string | null>(null);
  const [isPreviewReady, setIsPreviewReady] = useState(false);
  const [showDetectron, setShowDetectron] = useState(true);
  const [showYolo, setShowYolo] = useState(true);
  const [showOther, setShowOther] = useState(true);
  const [showOverlayLabels, setShowOverlayLabels] = useState(true);
  const casesRef = useRef<TriageCase[]>([]);
  const imageMapRef = useRef<string[]>([]);
  const previewImageRef = useRef<HTMLImageElement | null>(null);
  const previewCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const isVisitor = hasTokenAccess && !user;
  const displayName = user?.user_metadata?.full_name || user?.email?.split('@')[0] || 'Usuario';
  const profileHref = isVisitor ? '/perfil/visitante' : '/perfil';
  const profileLabel = isVisitor ? 'Perfil (Visitante)' : 'Perfil';
  const baseApiUrl = process.env.NEXT_PUBLIC_MODAL_API_URL ?? '';
  const sampleUrl = process.env.NEXT_PUBLIC_TRIAGEM_SAMPLE_URL ?? '/triagem_sample.json';
  const autoBatchUrl =
    process.env.NEXT_PUBLIC_TRIAGEM_BATCH_URL ?? '/batch_test/triagem_batch_results.json';
  const autoBatchImageBase =
    process.env.NEXT_PUBLIC_TRIAGEM_BATCH_IMAGE_BASE ?? '/batch_test/';
  const uploadAttempted = uploadPassword.trim().length > 0;
  const uploadUnlocked = false;

  useEffect(() => {
    let isActive = true;
    const verifyTokenAccess = async () => {
      try {
        const response = await fetch('/api/token-session', {
          method: 'GET',
          cache: 'no-store',
        });
        if (!isActive) {
          return;
        }
        if (!response.ok) {
          localStorage.removeItem('tokenAuth');
          setHasTokenAccess(false);
          return;
        }
        setHasTokenAccess(true);
      } catch (error) {
        console.error(error);
        if (isActive) {
          setHasTokenAccess(false);
        }
      } finally {
        if (isActive) {
          setTokenChecked(true);
        }
      }
    };
    verifyTokenAccess();
    return () => {
      isActive = false;
    };
  }, []);

  useEffect(() => {
    if (isLoading || !tokenChecked) {
      return;
    }
    if (!user && !hasTokenAccess) {
      router.replace('/');
    }
  }, [isLoading, tokenChecked, user, hasTokenAccess, router]);

  useEffect(() => {
    const savedTheme = localStorage.getItem('theme');
    const systemDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    const dark = savedTheme === 'dark' || (!savedTheme && systemDark);
    setIsDarkMode(dark);
    if (dark) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, []);

  useEffect(() => {
    setCases((prev) =>
      prev.map((caseItem) =>
        caseItem.status === 'pending' ? { ...caseItem, modelType } : caseItem,
      ),
    );
  }, [modelType]);

  useEffect(() => {
    casesRef.current = cases;
  }, [cases]);

  useEffect(() => {
    return () => {
      casesRef.current.forEach((caseItem) => {
        revokeObjectUrl(caseItem.previewUrl);
      });
      imageMapRef.current.forEach((url) => revokeObjectUrl(url));
    };
  }, []);

  const toggleTheme = () => {
    const nextDark = !isDarkMode;
    setIsDarkMode(nextDark);
    if (nextDark) {
      document.documentElement.classList.add('dark');
      localStorage.setItem('theme', 'dark');
    } else {
      document.documentElement.classList.remove('dark');
      localStorage.setItem('theme', 'light');
    }
  };

  const revokeObjectUrl = (url?: string | null) => {
    if (url && url.startsWith('blob:')) {
      URL.revokeObjectURL(url);
    }
  };

  const handleLogout = async () => {
    try {
      await supabase.auth.signOut();
    } catch (error) {
      console.error(error);
    } finally {
      try {
        await fetch('/api/token-session', { method: 'DELETE' });
      } catch (error) {
        console.error(error);
      }
      localStorage.removeItem('tokenAuth');
      router.push('/');
    }
  };

  const buildInsights = (caseItem: TriageCase): CaseInsight => {
    if (caseItem.status !== 'done') {
      return {
        severity: 'Aguardando',
        score: 0,
        needs: [],
        topFindings: [],
        visibleFindings: [],
        normalizedFindings: [],
      };
    }

    const normalizedFindings = enrichFindings(caseItem.findings, normalizeToothLabel);
    const visibleFindings = hideAnatomy
      ? normalizedFindings.filter((finding) => !isHiddenStructure(finding))
      : normalizedFindings;

    const triageFindings = visibleFindings.filter(
      (finding) => finding.category === 'pathology' || finding.category === 'treatment',
    );

    if (triageFindings.length === 0) {
      return {
        severity: 'Baixa',
        score: 0,
        needs: [],
        topFindings: [],
        visibleFindings,
        normalizedFindings,
      };
    }

    const needs = new Set<string>();
    const counts = new Map<string, number>();
    let score = 0;

    triageFindings.forEach((finding) => {
      const labelText = (finding.displayLabel ?? finding.label).toLowerCase();
      const canonical = finding.canonicalLabel ?? slugLabel(finding.label);
      const confidence = finding.confidence ?? 0.65;
      let matched = false;

      const rule = PRIORITY_RULES[canonical];
      if (rule) {
        matched = true;
        score += rule.weight * confidence;
        needs.add(rule.need);
      }

      if (!matched) {
        SEVERITY_RULES.forEach((fallback) => {
          if (fallback.keywords.some((keyword) => labelText.includes(keyword))) {
            matched = true;
            score += fallback.weight * confidence;
            needs.add(fallback.need);
          }
        });
      }

      if (!matched) {
        score += 0.2 * confidence;
      }

      const displayLabel = finding.displayLabel ?? finding.label;
      counts.set(displayLabel, (counts.get(displayLabel) ?? 0) + 1);
    });

    const topFindings = Array.from(counts.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, 3)
      .map(([label]) => label);

    return {
      severity: scoreSeverity(score),
      score,
      needs: Array.from(needs).slice(0, 3),
      topFindings,
      visibleFindings,
      normalizedFindings,
    };
  };

  const casesWithInsights = useMemo<TriageCaseWithInsights[]>(() => {
    return cases.map((caseItem, index) => {
      const insights = buildInsights(caseItem);
      return {
        ...caseItem,
        insights,
        displayName: buildCaseDisplayName(index),
        caseNumber: index + 1,
        patientName: pickPatientName(hashSeed(caseItem.name)),
        clinicalProfile: buildClinicalProfile({ name: caseItem.name, insights }),
      };
    });
  }, [cases, hideAnatomy, normalizeToothLabel]);

  const selectedCase = useMemo(
    () => casesWithInsights.find((caseItem) => caseItem.id === selectedCaseId) ?? null,
    [casesWithInsights, selectedCaseId],
  );
  const selectedClinicalProfile = useMemo(() => selectedCase?.clinicalProfile ?? null, [selectedCase]);

  useEffect(() => {
    if (!selectedCase && casesWithInsights.length > 0) {
      setSelectedCaseId(casesWithInsights[0].id);
    }
  }, [casesWithInsights, selectedCase]);

  useEffect(() => {
    setIsPreviewReady(false);
  }, [selectedCase?.previewUrl]);

  const totals = useMemo(() => {
    const analyzed = casesWithInsights.filter((caseItem) => caseItem.status === 'done');
    const pathologyCounts = new Map<string, number>();
    const toothGroupCounts = new Map<string, number>();
    const missingCounts: Record<string, number> = Object.fromEntries(
      Object.keys(TOOTH_EXPECTATIONS).map((key) => [key, 0]),
    );
    const severityCounts = {
      Alta: analyzed.filter((caseItem) => caseItem.insights.severity === 'Alta').length,
      Media: analyzed.filter((caseItem) => caseItem.insights.severity === 'Media').length,
      Baixa: analyzed.filter((caseItem) => caseItem.insights.severity === 'Baixa').length,
    };
    const needsCounts = new Map<string, number>();

    analyzed.forEach((caseItem) => {
      const perCaseToothCounts = new Map<string, number>();
      caseItem.insights.needs.forEach((need) => {
        needsCounts.set(need, (needsCounts.get(need) ?? 0) + 1);
      });
      caseItem.insights.normalizedFindings.forEach((finding) => {
        if (finding.category === 'tooth') {
          const group = finding.toothGroup ?? 'outros';
          toothGroupCounts.set(group, (toothGroupCounts.get(group) ?? 0) + 1);
          perCaseToothCounts.set(group, (perCaseToothCounts.get(group) ?? 0) + 1);
        }
        if (finding.category === 'pathology' || finding.category === 'treatment') {
          const label = finding.displayLabel ?? finding.label;
          pathologyCounts.set(label, (pathologyCounts.get(label) ?? 0) + 1);
        }
      });

      Object.entries(TOOTH_EXPECTATIONS).forEach(([group, expected]) => {
        const found = perCaseToothCounts.get(group) ?? 0;
        missingCounts[group] += Math.max(expected - found, 0);
      });
    });

    const topNeeds = Array.from(needsCounts.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, 5);

    const topLabels = Array.from(pathologyCounts.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, 6);

    const totalFindings = Array.from(pathologyCounts.values()).reduce((sum, value) => sum + value, 0);

    return {
      analyzed,
      totalFindings,
      severityCounts,
      topNeeds,
      topLabels,
      toothGroupCounts,
      missingCounts,
    };
  }, [casesWithInsights]);

  const priorityCases = useMemo(() => {
    const order = { Alta: 3, Media: 2, Baixa: 1, Aguardando: 0 };
    return casesWithInsights
      .filter((caseItem) => caseItem.status === 'done')
      .sort((a, b) => {
        const severityDelta = order[b.insights.severity] - order[a.insights.severity];
        if (severityDelta !== 0) {
          return severityDelta;
        }
        return b.insights.score - a.insights.score;
      })
      .slice(0, 8);
  }, [casesWithInsights]);

  const summaryContext = useMemo(() => {
    if (totals.analyzed.length === 0) {
      return '';
    }
    const needsText = totals.topNeeds.length
      ? totals.topNeeds.map(([need, count]) => `${need}: ${count}`).join(', ')
      : 'Sem necessidades definidas';
    const findingsText = totals.topLabels.length
      ? totals.topLabels.map(([label, count]) => `${label} (${count})`).join(', ')
      : 'Sem achados relevantes';
    const toothText = Array.from(totals.toothGroupCounts.entries())
      .map(([group, count]) => `${TOOTH_GROUP_LABELS[group] ?? group}: ${count}`)
      .join(', ');
    const missingText = Object.entries(totals.missingCounts)
      .map(([group, count]) => `${TOOTH_GROUP_LABELS[group] ?? group}: ${count}`)
      .join(', ');
    return [
      `Total de casos analisados: ${totals.analyzed.length}`,
      `Achados relevantes (sem estruturas): ${totals.totalFindings}`,
      `Severidade alta: ${totals.severityCounts.Alta}`,
      `Severidade media: ${totals.severityCounts.Media}`,
      `Severidade baixa: ${totals.severityCounts.Baixa}`,
      `Dentição detectada: ${toothText || 'Sem dados'}`,
      `Estimativa de ausentes: ${missingText || 'Sem dados'}`,
      `Necessidades mais comuns: ${needsText}`,
      `Top achados: ${findingsText}`,
    ].join('\n');
  }, [totals]);

  const handleFileSelection = (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(event.target.files ?? []);
    if (files.length === 0) {
      return;
    }
    const nextCases = files.map((file) => ({
      id: `${file.name}-${file.lastModified}-${Math.random().toString(36).slice(2, 8)}`,
      name: file.name,
      file,
      previewUrl: URL.createObjectURL(file),
      status: 'pending' as CaseStatus,
      findings: [],
      modelType,
      source: 'upload' as const,
    }));
    setCases((prev) => [...prev, ...nextCases]);
    setBatchError(null);
    setImportError(null);
    setSummary('');
    setSummaryError(null);
    event.target.value = '';
  };

  const ingestJsonPayload = (
    payload: unknown,
    options?: { imageBaseUrl?: string; replace?: boolean },
  ) => {
    const rawCases = extractCaseArray(payload);
    if (rawCases.length === 0) {
      throw new Error('Nenhum caso encontrado no JSON.');
    }
    const normalizedCases = rawCases.map((raw, index) =>
      normalizeCaseFromJson(raw as Record<string, unknown>, index, modelType, options?.imageBaseUrl),
    );
    setCases((prev) => {
      if (options?.replace) {
        prev.forEach((caseItem) => revokeObjectUrl(caseItem.previewUrl));
        imageMapRef.current.forEach((url) => revokeObjectUrl(url));
        imageMapRef.current = [];
        return normalizedCases;
      }
      return [...prev, ...normalizedCases];
    });
    setBatchError(null);
    setImportError(null);
    setSummary('');
    setSummaryError(null);
  };

  const handleJsonImport = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }
    setImportError(null);
    try {
      const text = await file.text();
      const payload = JSON.parse(text) as unknown;
      ingestJsonPayload(payload);
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Falha ao importar JSON.';
      setImportError(message);
    } finally {
      event.target.value = '';
    }
  };

  const handleImageMapping = (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(event.target.files ?? []);
    if (files.length === 0) {
      return;
    }
    const targetNames = new Set(
      cases
        .filter((caseItem) => !caseItem.previewUrl)
        .map((caseItem) => caseItem.name.toLowerCase()),
    );
    if (targetNames.size === 0) {
      event.target.value = '';
      return;
    }

    const urlMap = new Map<string, string>();
    files.forEach((file) => {
      const key = file.name.toLowerCase();
      if (!targetNames.has(key)) {
        return;
      }
      const url = URL.createObjectURL(file);
      imageMapRef.current.push(url);
      urlMap.set(key, url);
    });

    if (urlMap.size > 0) {
      setCases((prev) =>
        prev.map((caseItem) => {
          if (caseItem.previewUrl) {
            return caseItem;
          }
          const url = urlMap.get(caseItem.name.toLowerCase());
          return url ? { ...caseItem, previewUrl: url } : caseItem;
        }),
      );
    }
    event.target.value = '';
  };

  const handleSampleImport = async () => {
    setImportError(null);
    setIsSampleLoading(true);
    try {
      const response = await fetch(sampleUrl, { cache: 'no-store' });
      if (!response.ok) {
        throw new Error('Nao foi possivel carregar o JSON local.');
      }
      const payload = await response.json();
      ingestJsonPayload(payload);
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Falha ao carregar JSON local.';
      setImportError(message);
    } finally {
      setIsSampleLoading(false);
    }
  };

  const handleAutoBatchImport = async () => {
    setImportError(null);
    setIsAutoBatchLoading(true);
    try {
      const response = await fetch(autoBatchUrl, { cache: 'no-store' });
      if (!response.ok) {
        throw new Error('Nao foi possivel carregar o lote local.');
      }
      const payload = await response.json();
      ingestJsonPayload(payload, { imageBaseUrl: autoBatchImageBase, replace: true });
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Falha ao carregar lote local.';
      setImportError(message);
    } finally {
      setIsAutoBatchLoading(false);
    }
  };

  const removeCase = (id: string) => {
    setCases((prev) => {
      const target = prev.find((caseItem) => caseItem.id === id);
      revokeObjectUrl(target?.previewUrl);
      return prev.filter((caseItem) => caseItem.id !== id);
    });
  };

  const clearBatch = () => {
    setCases((prev) => {
      prev.forEach((caseItem) => {
        revokeObjectUrl(caseItem.previewUrl);
      });
      return [];
    });
    imageMapRef.current.forEach((url) => revokeObjectUrl(url));
    imageMapRef.current = [];
    setBatchError(null);
    setImportError(null);
    setSummary('');
    setSummaryError(null);
    setSelectedCaseId(null);
  };

  const analyzeCase = async (caseItem: TriageCase) => {
    if (!caseItem.file) {
      setCases((prev) =>
        prev.map((item) =>
          item.id === caseItem.id
            ? { ...item, status: 'error', error: 'Sem arquivo local para reanalise.' }
            : item,
        ),
      );
      return false;
    }
    if (!baseApiUrl) {
      setBatchError('Configure NEXT_PUBLIC_MODAL_API_URL para usar o backend.');
      return false;
    }
    setCases((prev) =>
      prev.map((item) =>
        item.id === caseItem.id ? { ...item, status: 'analyzing', error: undefined } : item,
      ),
    );
    try {
      const formData = new FormData();
      formData.append('file', caseItem.file);
      formData.append('model_type', caseItem.modelType);

      const response = await fetch(`${baseApiUrl}/analyze`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const text = await response.text();
        throw new Error(text || 'Falha ao analisar.');
      }

      const data = await response.json();
      const nextFindings = Array.isArray(data.findings) ? data.findings : [];

      setCases((prev) =>
        prev.map((item) =>
          item.id === caseItem.id
            ? { ...item, status: 'done', findings: nextFindings, error: undefined }
            : item,
        ),
      );
      return true;
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Falha ao analisar.';
      setCases((prev) =>
        prev.map((item) =>
          item.id === caseItem.id ? { ...item, status: 'error', error: message } : item,
        ),
      );
      return false;
    }
  };

  const handleAnalyzeBatch = async () => {
    if (cases.length === 0) {
      setBatchError('Adicione radiografias antes de analisar.');
      return;
    }
    if (!baseApiUrl) {
      setBatchError('Configure NEXT_PUBLIC_MODAL_API_URL para usar o backend.');
      return;
    }
    const pendingCases = cases.filter((caseItem) => caseItem.status !== 'done' && caseItem.file);
    if (pendingCases.length === 0) {
      setBatchError(
        cases.some((caseItem) => !caseItem.file)
          ? 'Somente casos importados. Envie novas radiografias para analisar.'
          : 'Todas as radiografias ja foram analisadas.',
      );
      return;
    }
    setIsBatchAnalyzing(true);
    setBatchError(null);
    setBatchProgress({ completed: 0, total: pendingCases.length });
    let completed = 0;
    for (const caseItem of pendingCases) {
      await analyzeCase(caseItem);
      completed += 1;
      setBatchProgress({ completed, total: pendingCases.length });
    }
    setIsBatchAnalyzing(false);
  };

  const handleOpenCase = (caseItem: TriageCaseWithInsights) => {
    try {
      const payload = {
        id: caseItem.id,
        name: caseItem.name,
        displayName: caseItem.displayName,
        caseNumber: caseItem.caseNumber,
        patientName: caseItem.patientName,
        previewUrl: caseItem.previewUrl,
        modelType: caseItem.modelType,
        insights: caseItem.insights,
        clinicalProfile: caseItem.clinicalProfile,
        createdAt: new Date().toISOString(),
      };
      localStorage.setItem(`triagem-case-${caseItem.caseNumber}`, JSON.stringify(payload));
    } catch (error) {
      console.error(error);
    }
    router.push(`/triagem/caso/${caseItem.caseNumber}`);
  };

  const handleGenerateSummary = async () => {
    if (!summaryContext) {
      setSummaryError('Analise ao menos um caso para gerar o resumo.');
      return;
    }
    if (!baseApiUrl) {
      setSummaryError('Configure NEXT_PUBLIC_MODAL_API_URL para usar o backend.');
      return;
    }
    setSummary('');
    setSummaryError(null);
    setIsSummaryLoading(true);
    try {
      const payload = {
        context: summaryContext,
        history: [
          {
            role: 'user',
            text: 'Gere um resumo epidemiologico curto com prioridades clinicas, riscos e recomendacoes.',
          },
        ],
      };
      const response = await fetch(`${baseApiUrl}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (!response.ok) {
        const text = await response.text();
        throw new Error(text || 'Falha ao gerar resumo.');
      }
      const data = await response.json();
      setSummary(data.response || 'Resumo indisponivel.');
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Falha ao gerar resumo.';
      setSummaryError(message);
    } finally {
      setIsSummaryLoading(false);
    }
  };

  useEffect(() => {
    const canvas = previewCanvasRef.current;
    const img = previewImageRef.current;
    if (!canvas || !img || !selectedCase || !isPreviewReady) {
      return;
    }

    const drawFindings = () => {
      const displayWidth = img.clientWidth;
      const displayHeight = img.clientHeight;
      if (displayWidth === 0 || displayHeight === 0) {
        return;
      }
      canvas.width = displayWidth;
      canvas.height = displayHeight;

      const ctx = canvas.getContext('2d');
      if (!ctx) {
        return;
      }
      ctx.clearRect(0, 0, displayWidth, displayHeight);

      const scaleX = displayWidth / (img.naturalWidth || displayWidth);
      const scaleY = displayHeight / (img.naturalHeight || displayHeight);
      const visibleFindings = selectedCase.insights.visibleFindings.filter((finding) => {
        const kind = getSourceKind(finding);
        if (kind === 'yolo') {
          return showYolo;
        }
        if (kind === 'detectron') {
          return showDetectron;
        }
        return showOther;
      });

      visibleFindings.forEach((finding) => {
        const style = getFindingStyle(finding);
        ctx.strokeStyle = style.stroke;
        ctx.fillStyle = style.fill;
        ctx.lineWidth = 2;

        if (finding.segmentation?.length) {
          ctx.beginPath();
          finding.segmentation.forEach((point, index) => {
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
          ctx.fill();
        } else if (finding.bbox?.length === 4) {
          const [x1, y1, x2, y2] = finding.bbox;
          const width = (x2 - x1) * scaleX;
          const height = (y2 - y1) * scaleY;
          ctx.strokeRect(x1 * scaleX, y1 * scaleY, width, height);
          ctx.fillRect(x1 * scaleX, y1 * scaleY, width, height);
        }

        if (showOverlayLabels) {
          let labelX = 8;
          let labelY = 14;
          if (finding.bbox?.length === 4) {
            const [x1, y1] = finding.bbox;
            labelX = x1 * scaleX + 4;
            labelY = y1 * scaleY + 14;
          } else if (finding.segmentation?.length) {
            const [x, y] = finding.segmentation[0];
            labelX = x * scaleX + 4;
            labelY = y * scaleY + 14;
          }

          const sourceKind = getSourceKind(finding);
          const sourceShort = sourceKind === 'detectron' ? 'D' : sourceKind === 'yolo' ? 'Y' : 'O';
          const toothText = finding.toothId != null
            ? `·${finding.toothId}`
            : finding.toothType
              ? `·${formatToothTag(finding.toothType)}`
              : '';
          const displayLabel = finding.displayLabel ?? finding.label;
          const labelText = `${displayLabel}${toothText}·${sourceShort}`;

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
      });
    };

    drawFindings();
    const handleResize = () => drawFindings();
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [selectedCase, isPreviewReady, hideAnatomy, normalizeToothLabel, showDetectron, showYolo, showOther, showOverlayLabels]);

  if (isLoading || !tokenChecked) {
    return (
      <div className="flex h-screen w-full items-center justify-center">
        <div className="h-8 w-8 animate-spin rounded-full border-4 border-brand-green border-t-transparent"></div>
      </div>
    );
  }

  return (
    <div className="min-h-screen relative overflow-hidden transition-colors duration-500 text-gray-900 dark:text-gray-100 bg-background">
      <header className="absolute left-6 right-6 top-6 z-20 flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <button
            type="button"
            onClick={() => router.push('/dashboard')}
            className="flex h-10 w-10 items-center justify-center rounded-full glass-panel text-brand-blue dark:text-brand-yellow shadow-lg hover:scale-105 transition-transform"
            title="Voltar"
          >
            <ArrowLeft className="h-5 w-5" />
          </button>
          <img
            src={
              isDarkMode
                ? 'https://i.ibb.co/yBWfdYwN/Captura-de-Tela-2026-01-08-s-13-17-59-removebg-preview.png'
                : 'https://i.ibb.co/B5Lsvm4M/Captura-de-Tela-2026-01-08-s-12-59-47-removebg-preview.png'
            }
            alt="radiologIA"
            className="h-9 w-auto drop-shadow-sm"
          />
        </div>

        <div className="flex flex-wrap items-center gap-3">
          <button
            onClick={toggleTheme}
            className="p-2.5 rounded-full glass-panel shadow-lg hover:scale-105 transition-transform text-brand-blue dark:text-brand-yellow"
            title="Alternar tema"
          >
            {isDarkMode ? <Sun className="h-5 w-5" /> : <Moon className="h-5 w-5" />}
          </button>

          <Popover>
            <PopoverTrigger asChild>
              <button
                className="flex items-center gap-2 rounded-full glass-panel border border-brand-blue/20 dark:border-white/15 px-3 py-2 text-xs uppercase tracking-wider font-semibold text-brand-blue/80 dark:text-white/80"
                title="Perfil"
              >
                <span className="flex h-8 w-8 items-center justify-center rounded-full bg-brand-blue/10 text-brand-blue dark:bg-white/10 dark:text-white">
                  <UserRound className="h-4 w-4" />
                </span>
                {isVisitor ? 'Visitante' : displayName}
              </button>
            </PopoverTrigger>
            <PopoverContent
              align="end"
              className="w-64 rounded-2xl border border-white/20 bg-white/80 dark:bg-slate-900/80 text-brand-blue dark:text-white backdrop-blur-lg shadow-xl p-3"
            >
              <div className="px-2 py-2">
                <p className="text-sm font-semibold">{isVisitor ? 'Visitante' : displayName}</p>
                <p className="text-xs text-brand-blue/70 dark:text-white/70">
                  {isVisitor ? 'Acesso via token' : user?.email}
                </p>
              </div>
              <div className="my-2 h-px bg-brand-blue/10 dark:bg-white/10"></div>
              <Link
                href={profileHref}
                className="flex items-center gap-2 rounded-lg px-3 py-2 text-sm hover:bg-brand-blue/5 dark:hover:bg-white/5 transition-colors"
              >
                <UserRound className="h-4 w-4" />
                {profileLabel}
              </Link>
              <button
                onClick={handleLogout}
                className="mt-2 w-full flex items-center gap-2 rounded-lg px-3 py-2 text-sm text-white bg-brand-green shadow-sm hover:shadow-md transition-all"
              >
                <LogOut className="h-4 w-4" />
                Sair
              </button>
            </PopoverContent>
          </Popover>
        </div>
      </header>

      <div className="absolute inset-0 z-0 flex items-center justify-center pointer-events-none">
        <img
          src="https://i.ibb.co/9HFVnY4x/Gemini-Generated-Image-oc1jgfoc1jgfoc1j-Photoroom.png"
          alt=""
          aria-hidden="true"
          className="w-[70vw] max-w-5xl opacity-90"
        />
      </div>

      <main className="relative z-10 min-h-screen pt-28 px-4 pb-10">
        <div className="mx-auto flex w-full max-w-6xl flex-col gap-6">
          <section className="rounded-3xl border border-white/20 dark:border-white/10 bg-white/50 dark:bg-white/5 backdrop-blur-xl p-6 shadow-xl">
            <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
              <div>
                <p className="text-xs uppercase tracking-[0.3em] text-brand-blue/60 dark:text-white/60">
                  Triagem Epidemiológica
                </p>
                <h1 className="mt-2 text-2xl font-semibold text-brand-blue dark:text-white">
                  Panorama clínico para priorizar comunidades remotas
                </h1>
                <p className="mt-2 text-sm text-slate-600 dark:text-slate-300 max-w-2xl">
                  Carregue grandes lotes de radiografias, identifique necessidades críticas e gere
                  uma fila de prioridade baseada em gravidade.
                </p>
              </div>
              <div className="flex flex-wrap items-center gap-2">
                <button
                  type="button"
                  onClick={clearBatch}
                  className="rounded-full border border-brand-blue/30 dark:border-white/20 px-4 py-2 text-xs font-semibold uppercase tracking-wider text-brand-blue dark:text-white hover:bg-white/40 dark:hover:bg-white/10 transition-all"
                >
                  Limpar
                </button>
              </div>
            </div>
          </section>

          <div className="grid gap-6 xl:grid-cols-[1.05fr_1.5fr]">
            <section className="rounded-3xl border border-white/20 dark:border-white/10 bg-white/45 dark:bg-white/5 backdrop-blur-xl p-6 shadow-xl">
              <div className="flex items-center justify-between">
                <div>
                  <h2 className="text-sm font-semibold uppercase tracking-wider text-brand-blue dark:text-white">
                    Entrada do lote
                  </h2>
                  <p className="mt-1 text-xs text-slate-600 dark:text-slate-300">
                    Adicione radiografias e acompanhe o processamento.
                  </p>
                </div>
                <button
                  type="button"
                  onClick={() => {
                    setHideAnatomy((prev) => !prev);
                  }}
                  className="flex items-center gap-2 rounded-full border border-brand-blue/30 dark:border-white/20 px-3 py-1 text-[11px] font-semibold uppercase tracking-wider text-brand-blue dark:text-white hover:bg-white/40 dark:hover:bg-white/10 transition-all"
                >
                  <Filter className="h-3 w-3" />
                  {hideAnatomy ? 'Ocultando estruturas' : 'Mostrando estruturas'}
                </button>
              </div>

              <div className="mt-4 rounded-2xl border border-dashed border-white/30 dark:border-white/15 bg-white/40 dark:bg-white/5 p-6 text-center">
                <UploadCloud className="mx-auto h-7 w-7 text-brand-blue/60 dark:text-white/60" />
                <p className="mt-2 text-sm text-slate-600 dark:text-slate-300">
                  Este modo roda apenas o lote já analisado (50 radiografias).
                </p>
                <p className="mt-1 text-xs text-slate-500 dark:text-slate-400">
                  Os dados foram gerados pelos mesmos modelos para evitar uso desnecessário de GPU.
                </p>
                <button
                  type="button"
                  onClick={handleAutoBatchImport}
                  disabled={isAutoBatchLoading}
                  className="mt-4 inline-flex items-center gap-2 rounded-full bg-brand-green px-4 py-2 text-xs font-semibold text-white shadow-md transition-all disabled:opacity-60 disabled:cursor-not-allowed"
                >
                  <BarChart3 className="h-4 w-4" />
                  {isAutoBatchLoading ? 'Carregando lote 50...' : 'Usar lote 50 (local)'}
                </button>
              </div>

              <div className="mt-4 rounded-2xl border border-white/30 dark:border-white/10 bg-white/40 dark:bg-white/5 p-5 text-left">
                <p className="text-xs font-semibold uppercase tracking-wider text-brand-blue dark:text-white">
                  Upload protegido
                </p>
                <p className="mt-1 text-xs text-slate-500 dark:text-slate-400">
                  Para evitar consumo desnecessário de GPU, digite uma senha para liberar o upload
                  manual. Caso queira apenas visualizar, use o lote já analisado acima.
                </p>
                <div className="mt-3 flex flex-wrap items-center gap-3">
                  <input
                    type="password"
                    value={uploadPassword}
                    onChange={(event) => setUploadPassword(event.target.value)}
                    placeholder="Senha de acesso"
                    className="flex-1 rounded-full border border-brand-blue/30 bg-white/70 px-4 py-2 text-xs text-brand-blue shadow-sm outline-none focus:border-brand-blue/60 dark:border-white/20 dark:bg-slate-900/50 dark:text-white"
                  />
                  <span
                    className={[
                      'rounded-full px-3 py-1 text-[10px] font-semibold uppercase tracking-wider',
                      uploadAttempted
                        ? 'bg-red-200 text-red-600'
                        : 'bg-brand-yellow/20 text-brand-blue',
                    ].join(' ')}
                  >
                    {uploadAttempted ? 'Senha inválida' : 'Acesso restrito'}
                  </span>
                </div>
                <div className="mt-3 flex flex-wrap items-center gap-2">
                  <label
                    className={[
                      'inline-flex items-center gap-2 rounded-full px-4 py-2 text-xs font-semibold shadow-sm transition-all',
                      uploadUnlocked
                        ? 'cursor-pointer bg-brand-blue text-white'
                        : 'cursor-not-allowed bg-slate-200 text-slate-400',
                    ].join(' ')}
                  >
                    <UploadCloud className="h-4 w-4" />
                    Enviar radiografias
                    <input
                      type="file"
                      accept="image/*"
                      multiple
                      onChange={handleFileSelection}
                      className="hidden"
                      disabled={!uploadUnlocked}
                    />
                  </label>
                  <button
                    type="button"
                    onClick={handleAnalyzeBatch}
                    disabled={!uploadUnlocked || isBatchAnalyzing || cases.length === 0}
                    className="rounded-full bg-brand-green px-4 py-2 text-xs font-semibold text-white shadow-md transition-all disabled:opacity-60 disabled:cursor-not-allowed"
                  >
                    {isBatchAnalyzing
                      ? `Analisando ${batchProgress.completed}/${batchProgress.total}`
                      : 'Analisar lote'}
                  </button>
                </div>
                <p className="mt-2 text-[11px] text-slate-500 dark:text-slate-400">
                  Acesso protegido para evitar uso desnecessário de GPU.
                </p>
              </div>

              <div className="mt-4 flex flex-wrap items-center gap-2 text-[11px] uppercase tracking-wider text-slate-500 dark:text-slate-400">
                <button
                  type="button"
                  onClick={() => setNormalizeToothLabel((prev) => !prev)}
                  className="rounded-full border border-brand-blue/30 dark:border-white/20 px-3 py-1 font-semibold text-brand-blue dark:text-white hover:bg-white/40 dark:hover:bg-white/10 transition-all"
                >
                  {normalizeToothLabel ? 'Rótulos normalizados' : 'Rótulos originais'}
                </button>
                <span>
                  {cases.length} radiografias carregadas
                </span>
                <span className="text-slate-400 dark:text-slate-500">
                  Lote local vinculado automaticamente.
                </span>
                {batchError ? (
                  <span className="text-red-500 dark:text-red-300">{batchError}</span>
                ) : null}
                {importError ? (
                  <span className="text-red-500 dark:text-red-300">{importError}</span>
                ) : null}
              </div>

              <div className="mt-4 max-h-64 space-y-3 overflow-y-auto pr-2">
                {casesWithInsights.length === 0 ? (
                  <div className="rounded-2xl border border-white/30 dark:border-white/10 bg-white/30 dark:bg-white/5 p-4 text-sm text-slate-600 dark:text-slate-300">
                    Nenhuma radiografia no lote ainda.
                  </div>
                ) : (
                  casesWithInsights.map((caseItem) => (
                    <div
                      key={caseItem.id}
                      className={[
                        'flex items-center gap-3 rounded-2xl border bg-white/30 dark:bg-white/5 p-3 transition-all cursor-pointer',
                        selectedCaseId === caseItem.id
                          ? 'border-brand-green/60 ring-1 ring-brand-green/30'
                          : 'border-white/30 dark:border-white/10 hover:border-brand-blue/30',
                      ].join(' ')}
                      onClick={() => handleOpenCase(caseItem)}
                    >
                      <div className="h-12 w-12 overflow-hidden rounded-xl bg-white/60 dark:bg-slate-900/60 flex items-center justify-center">
                        {caseItem.previewUrl ? (
                          <img
                            src={caseItem.previewUrl}
                            alt={caseItem.displayName}
                            className="h-full w-full object-cover"
                          />
                        ) : (
                          <FileSearch className="h-5 w-5 text-brand-blue/60 dark:text-white/60" />
                        )}
                      </div>
                      <div className="flex-1">
                        <div className="flex items-start justify-between gap-2">
                          <div>
                        <p className="text-sm font-semibold text-brand-blue dark:text-white">
                          {caseItem.displayName}
                        </p>
                            <p className="text-[11px] uppercase tracking-wider text-slate-500 dark:text-slate-400">
                              {caseItem.status === 'done'
                                ? `${
                                    caseItem.insights.normalizedFindings.filter(
                                      (finding) =>
                                        finding.category === 'pathology' ||
                                        finding.category === 'treatment',
                                    ).length
                                  } achados`
                                : caseItem.status === 'analyzing'
                                  ? 'Analisando'
                                  : caseItem.status === 'error'
                                    ? 'Erro'
                                    : 'Pendente'}
                            </p>
                          </div>
                          <span
                            className={[
                              'rounded-full px-2 py-1 text-[10px] font-semibold uppercase tracking-wider',
                              caseItem.status === 'done'
                                ? 'bg-brand-green/20 text-brand-green'
                                : caseItem.status === 'analyzing'
                                  ? 'bg-brand-yellow/20 text-brand-blue'
                                  : caseItem.status === 'error'
                                    ? 'bg-red-200 text-red-600'
                                    : 'bg-slate-200 text-slate-600',
                            ].join(' ')}
                          >
                            {caseItem.status === 'done'
                              ? caseItem.insights.severity
                              : caseItem.status === 'analyzing'
                                ? 'Processando'
                                : caseItem.status === 'error'
                                  ? 'Falhou'
                                  : 'Pendente'}
                          </span>
                        </div>
                        <p className="mt-1 text-[11px] uppercase tracking-wider text-slate-400 dark:text-slate-500">
                          {caseItem.patientName}
                        </p>
                        {caseItem.status === 'done' && caseItem.insights.topFindings.length > 0 ? (
                          <p className="mt-1 text-xs text-slate-600 dark:text-slate-300">
                            {caseItem.insights.topFindings.join(', ')}
                          </p>
                        ) : null}
                        {caseItem.status === 'error' && caseItem.error ? (
                          <p className="mt-1 text-xs text-red-500 dark:text-red-300">{caseItem.error}</p>
                        ) : null}
                      </div>
                      <div className="flex flex-col items-end gap-2">
                        <button
                          type="button"
                          onClick={(event) => {
                            event.stopPropagation();
                            analyzeCase(caseItem);
                          }}
                          disabled={isBatchAnalyzing || caseItem.status === 'analyzing' || !caseItem.file}
                          className="rounded-full border border-brand-blue/30 dark:border-white/20 px-3 py-1 text-[10px] font-semibold uppercase tracking-wider text-brand-blue dark:text-white hover:bg-white/40 dark:hover:bg-white/10 transition-all disabled:opacity-50"
                        >
                          {caseItem.status === 'done' ? 'Reanalisar' : 'Analisar'}
                        </button>
                        <button
                          type="button"
                          onClick={(event) => {
                            event.stopPropagation();
                            removeCase(caseItem.id);
                          }}
                          className="text-slate-400 hover:text-red-500 transition-colors"
                          title="Remover"
                        >
                          <X className="h-4 w-4" />
                        </button>
                      </div>
                    </div>
                  ))
                )}
              </div>

              <div className="mt-6 rounded-2xl border border-white/30 dark:border-white/10 bg-white/40 dark:bg-white/5 p-4">
                <div className="flex items-center justify-between gap-3">
                  <div>
                    <h3 className="text-xs font-semibold uppercase tracking-wider text-brand-blue dark:text-white">
                      Visualizacao do caso
                    </h3>
                    <p className="mt-1 text-[11px] text-slate-500 dark:text-slate-400">
                      Clique em um caso para ver os achados sobrepostos.
                    </p>
                  </div>
                  {selectedCase ? (
                    <span className="rounded-full bg-brand-blue/10 px-3 py-1 text-[10px] font-semibold uppercase tracking-wider text-brand-blue dark:bg-white/10 dark:text-white">
                      {selectedCase.displayName}
                    </span>
                  ) : null}
                </div>

                <div className="mt-3 relative overflow-hidden rounded-2xl border border-white/40 dark:border-white/10 bg-white/60 dark:bg-slate-900/60 flex items-center justify-center min-h-[320px]">
                  {selectedCase?.previewUrl ? (
                    <>
                      <img
                        ref={previewImageRef}
                        src={selectedCase.previewUrl}
                        alt={selectedCase.displayName}
                        onLoad={() => setIsPreviewReady(true)}
                        className="max-h-[480px] w-full object-contain"
                      />
                      <canvas
                        ref={previewCanvasRef}
                        className="absolute inset-0 h-full w-full pointer-events-none"
                      />
                    </>
                  ) : (
                    <div className="text-center text-xs text-slate-500 dark:text-slate-400">
                      {selectedCase
                        ? 'Vincule a imagem local para visualizar os achados.'
                        : 'Selecione um caso para visualizar.'}
                    </div>
                  )}
                </div>

                {selectedCase ? (
                  <div className="mt-3 flex flex-wrap items-center gap-2">
                    <button
                      type="button"
                      onClick={() => setShowYolo((prev) => !prev)}
                      className={[
                        'rounded-full border px-3 py-1 text-[10px] font-semibold uppercase tracking-wider transition-all',
                        showYolo
                          ? 'bg-brand-green/20 border-brand-green/40 text-brand-green'
                          : 'border-brand-blue/30 text-brand-blue dark:text-white/70',
                      ].join(' ')}
                    >
                      YOLO
                    </button>
                    <button
                      type="button"
                      onClick={() => setShowDetectron((prev) => !prev)}
                      className={[
                        'rounded-full border px-3 py-1 text-[10px] font-semibold uppercase tracking-wider transition-all',
                        showDetectron
                          ? 'bg-sky-200/40 border-sky-200 text-sky-700'
                          : 'border-brand-blue/30 text-brand-blue dark:text-white/70',
                      ].join(' ')}
                    >
                      Detectron
                    </button>
                    <button
                      type="button"
                      onClick={() => setShowOther((prev) => !prev)}
                      className={[
                        'rounded-full border px-3 py-1 text-[10px] font-semibold uppercase tracking-wider transition-all',
                        showOther
                          ? 'bg-brand-yellow/20 border-brand-yellow/50 text-brand-blue'
                          : 'border-brand-blue/30 text-brand-blue dark:text-white/70',
                      ].join(' ')}
                    >
                      Outros
                    </button>
                    <button
                      type="button"
                      onClick={() => setShowOverlayLabels((prev) => !prev)}
                      className={[
                        'rounded-full border px-3 py-1 text-[10px] font-semibold uppercase tracking-wider transition-all',
                        showOverlayLabels
                          ? 'bg-white/70 border-white/40 text-brand-blue'
                          : 'border-brand-blue/30 text-brand-blue dark:text-white/70',
                      ].join(' ')}
                    >
                      Rotulos
                    </button>
                  </div>
                ) : null}

                {selectedCase ? (
                  <div className="mt-3 flex flex-wrap items-center gap-2 text-[11px] uppercase tracking-wider text-slate-500 dark:text-slate-400">
                    <span>
                      Achados:{' '}
                      {
                        selectedCase.insights.normalizedFindings.filter(
                          (finding) =>
                            finding.category === 'pathology' || finding.category === 'treatment',
                        ).length
                      }
                    </span>
                    <span>
                      Dentes mapeados:{' '}
                      {selectedCase.insights.visibleFindings.filter(
                        (f) => f.toothId != null || f.toothType,
                      ).length}
                    </span>
                    <span>
                      Modelo: {formatModelType(selectedCase.modelType)}
                    </span>
                  </div>
                ) : null}

                {selectedCase && selectedClinicalProfile ? (
                  <div className="mt-4 rounded-2xl border border-white/30 dark:border-white/10 bg-white/40 dark:bg-white/5 p-4 text-xs text-slate-600 dark:text-slate-300">
                    <div className="flex items-center justify-between gap-3">
                      <p className="text-[11px] font-semibold uppercase tracking-wider text-brand-blue dark:text-white">
                        Ficha clínica completa (fictícia)
                      </p>
                      <span className="rounded-full bg-brand-yellow/20 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wider text-brand-blue">
                        Demonstração
                      </span>
                    </div>
                    <div className="mt-3 rounded-xl border border-white/40 bg-white/60 px-3 py-2 text-sm font-semibold text-brand-blue dark:border-white/10 dark:bg-slate-900/40 dark:text-white">
                      {selectedClinicalProfile.name} · {selectedCase.displayName}
                    </div>
                    <div className="mt-3 grid gap-3 sm:grid-cols-2">
                      <div>
                        <p className="text-[10px] uppercase tracking-wider text-slate-500 dark:text-slate-400">
                          Idade
                        </p>
                        <p className="mt-1 text-sm font-semibold text-brand-blue dark:text-white">
                          {selectedClinicalProfile.age} anos
                        </p>
                      </div>
                      <div>
                        <p className="text-[10px] uppercase tracking-wider text-slate-500 dark:text-slate-400">
                          Gênero
                        </p>
                        <p className="mt-1 text-sm font-semibold text-brand-blue dark:text-white">
                          {selectedClinicalProfile.gender}
                        </p>
                      </div>
                    </div>
                    <div className="mt-3">
                      <p className="text-[10px] uppercase tracking-wider text-slate-500 dark:text-slate-400">
                        Queixa principal
                      </p>
                      <p className="mt-1 text-sm font-semibold text-brand-blue dark:text-white">
                        {selectedClinicalProfile.complaint}
                      </p>
                    </div>
                    <div className="mt-3">
                      <p className="text-[10px] uppercase tracking-wider text-slate-500 dark:text-slate-400">
                        Sintomas relatados
                      </p>
                      <div className="mt-2 flex flex-wrap gap-2">
                        {selectedClinicalProfile.symptoms.length === 0 ? (
                          <span className="text-[11px] text-slate-500 dark:text-slate-400">
                            Sem sintomas relatados
                          </span>
                        ) : (
                          selectedClinicalProfile.symptoms.map((symptom) => (
                            <span
                              key={symptom}
                              className="rounded-full bg-white/70 px-2 py-0.5 text-[10px] font-semibold text-brand-blue dark:bg-white/10 dark:text-white"
                            >
                              {symptom}
                            </span>
                          ))
                        )}
                      </div>
                    </div>
                    {selectedClinicalProfile.alerts.length > 0 ? (
                      <div className="mt-3">
                        <p className="text-[10px] uppercase tracking-wider text-slate-500 dark:text-slate-400">
                          Sinais de alerta
                        </p>
                        <div className="mt-2 flex flex-wrap gap-2">
                          {selectedClinicalProfile.alerts.map((alert) => (
                            <span
                              key={alert}
                              className="rounded-full bg-red-200 px-2 py-0.5 text-[10px] font-semibold text-red-700"
                            >
                              {alert}
                            </span>
                          ))}
                        </div>
                      </div>
                    ) : null}
                  </div>
                ) : null}
              </div>
            </section>

            <section className="rounded-3xl border border-white/20 dark:border-white/10 bg-white/45 dark:bg-white/5 backdrop-blur-xl p-6 shadow-xl">
              <div className="flex flex-wrap items-center justify-between gap-3">
                <div>
                  <h2 className="text-sm font-semibold uppercase tracking-wider text-brand-blue dark:text-white">
                    Visão epidemiológica
                  </h2>
                  <p className="mt-1 text-xs text-slate-600 dark:text-slate-300">
                    Indicadores principais para tomada de decisão.
                  </p>
                </div>
                <button
                  type="button"
                  onClick={handleGenerateSummary}
                  disabled={isSummaryLoading || totals.analyzed.length === 0}
                  className="flex items-center gap-2 rounded-full bg-brand-green px-4 py-2 text-xs font-semibold text-white shadow-md transition-all disabled:opacity-60 disabled:cursor-not-allowed"
                >
                  <FileSearch className="h-4 w-4" />
                  {isSummaryLoading ? 'Gerando resumo...' : 'Resumo IA'}
                </button>
              </div>

              <div className="mt-5 grid gap-4 sm:grid-cols-3">
                <div className="rounded-2xl border border-white/30 dark:border-white/10 bg-white/40 dark:bg-white/5 p-4">
                  <p className="text-xs uppercase tracking-wider text-slate-500 dark:text-slate-400">
                    Casos analisados
                  </p>
                  <p className="mt-2 text-2xl font-semibold text-brand-blue dark:text-white">
                    {totals.analyzed.length}
                  </p>
                </div>
                <div className="rounded-2xl border border-white/30 dark:border-white/10 bg-white/40 dark:bg-white/5 p-4">
                  <p className="text-xs uppercase tracking-wider text-slate-500 dark:text-slate-400">
                    Achados relevantes
                  </p>
                  <p className="mt-2 text-2xl font-semibold text-brand-blue dark:text-white">
                    {totals.totalFindings}
                  </p>
                </div>
                <div className="rounded-2xl border border-white/30 dark:border-white/10 bg-white/40 dark:bg-white/5 p-4">
                  <p className="text-xs uppercase tracking-wider text-slate-500 dark:text-slate-400">
                    Alta prioridade
                  </p>
                  <p className="mt-2 text-2xl font-semibold text-brand-blue dark:text-white">
                    {totals.severityCounts.Alta}
                  </p>
                </div>
              </div>

              <div className="mt-6 grid gap-6 lg:grid-cols-2">
                <div className="rounded-2xl border border-white/30 dark:border-white/10 bg-white/40 dark:bg-white/5 p-4">
                  <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-brand-blue dark:text-white">
                    <BarChart3 className="h-4 w-4" />
                    Severidade clínica
                  </div>
                  <div className="mt-4 space-y-3 text-xs text-slate-600 dark:text-slate-300">
                    {[
                      { label: 'Alta', value: totals.severityCounts.Alta, color: 'bg-red-400' },
                      { label: 'Media', value: totals.severityCounts.Media, color: 'bg-brand-yellow' },
                      { label: 'Baixa', value: totals.severityCounts.Baixa, color: 'bg-brand-green' },
                    ].map((item) => (
                      <div key={item.label} className="flex items-center gap-3">
                        <span className="w-12">{item.label}</span>
                        <div className="h-2 flex-1 overflow-hidden rounded-full bg-white/50 dark:bg-white/10">
                          <div
                            className={`h-full ${item.color}`}
                            style={{
                              width: `${toPercent(
                                item.value,
                                Math.max(
                                  totals.severityCounts.Alta,
                                  totals.severityCounts.Media,
                                  totals.severityCounts.Baixa,
                                  1,
                                ),
                              )}%`,
                            }}
                          ></div>
                        </div>
                        <span className="w-6 text-right">{item.value}</span>
                      </div>
                    ))}
                  </div>
                </div>
                <div className="rounded-2xl border border-white/30 dark:border-white/10 bg-white/40 dark:bg-white/5 p-4">
                  <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-brand-blue dark:text-white">
                    <CheckCircle2 className="h-4 w-4" />
                    Necessidades encontradas
                  </div>
                  <div className="mt-4 space-y-3 text-xs text-slate-600 dark:text-slate-300">
                    {totals.topNeeds.length === 0 ? (
                      <p className="text-xs text-slate-500 dark:text-slate-400">
                        Sem dados suficientes ainda.
                      </p>
                    ) : (
                      totals.topNeeds.map(([need, count]) => (
                        <div key={need} className="flex items-center gap-3">
                          <span className="flex-1">{need}</span>
                          <span className="rounded-full bg-brand-blue/10 px-2 py-0.5 text-[10px] font-semibold text-brand-blue dark:bg-white/10 dark:text-white">
                            {count}
                          </span>
                        </div>
                      ))
                    )}
                  </div>
                </div>
              </div>

              <div className="mt-6 rounded-2xl border border-white/30 dark:border-white/10 bg-white/40 dark:bg-white/5 p-4">
                <div className="flex items-center justify-between gap-3">
                  <div>
                    <p className="text-xs font-semibold uppercase tracking-wider text-brand-blue dark:text-white">
                      Top achados
                    </p>
                    <p className="mt-1 text-[11px] text-slate-500 dark:text-slate-400">
                      Distribuição das principais suspeitas.
                    </p>
                  </div>
                </div>
                <div className="mt-4 space-y-3 text-xs text-slate-600 dark:text-slate-300">
                  {totals.topLabels.length === 0 ? (
                    <p className="text-xs text-slate-500 dark:text-slate-400">
                      Aguarde as primeiras analises.
                    </p>
                  ) : (
                    totals.topLabels.map(([label, count]) => (
                      <div key={label} className="flex items-center gap-3">
                        <span className="w-28 truncate">{label}</span>
                        <div className="h-2 flex-1 overflow-hidden rounded-full bg-white/50 dark:bg-white/10">
                          <div
                            className="h-full bg-brand-blue"
                            style={{
                              width: `${toPercent(
                                count,
                                Math.max(...totals.topLabels.map(([, value]) => value), 1),
                              )}%`,
                            }}
                          ></div>
                        </div>
                        <span className="w-6 text-right">{count}</span>
                      </div>
                    ))
                  )}
                </div>
              </div>

              <div className="mt-6 rounded-2xl border border-white/30 dark:border-white/10 bg-white/40 dark:bg-white/5 p-4">
                <div className="flex items-center justify-between gap-3">
                  <div>
                    <p className="text-xs font-semibold uppercase tracking-wider text-brand-blue dark:text-white">
                      Distribuição dentária
                    </p>
                    <p className="mt-1 text-[11px] text-slate-500 dark:text-slate-400">
                      Estimativa de dentes presentes e ausentes.
                    </p>
                  </div>
                  <span className="rounded-full bg-brand-blue/10 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wider text-brand-blue dark:bg-white/10 dark:text-white">
                    Estimativa
                  </span>
                </div>
                <div className="mt-4 grid gap-4 text-xs text-slate-600 dark:text-slate-300 sm:grid-cols-2">
                  <div>
                    <p className="text-[11px] font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400">
                      Dentes detectados
                    </p>
                    <div className="mt-2 space-y-2">
                      {Array.from(totals.toothGroupCounts.entries()).length === 0 ? (
                        <p className="text-xs text-slate-500 dark:text-slate-400">
                          Sem dados suficientes.
                        </p>
                      ) : (
                        Array.from(totals.toothGroupCounts.entries())
                          .sort((a, b) => b[1] - a[1])
                          .map(([group, count]) => (
                            <div key={group} className="flex items-center justify-between">
                              <span>{TOOTH_GROUP_LABELS[group] ?? group}</span>
                              <span className="rounded-full bg-white/70 px-2 py-0.5 text-[10px] font-semibold text-brand-blue dark:bg-white/10 dark:text-white">
                                {count}
                              </span>
                            </div>
                          ))
                      )}
                    </div>
                  </div>
                  <div>
                    <p className="text-[11px] font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400">
                      Ausentes (estimado)
                    </p>
                    <div className="mt-2 space-y-2">
                      {Object.keys(totals.missingCounts).length === 0 ? (
                        <p className="text-xs text-slate-500 dark:text-slate-400">
                          Sem dados suficientes.
                        </p>
                      ) : (
                        Object.entries(totals.missingCounts).map(([group, count]) => (
                          <div key={group} className="flex items-center justify-between">
                            <span>{TOOTH_GROUP_LABELS[group] ?? group}</span>
                            <span className="rounded-full bg-brand-yellow/20 px-2 py-0.5 text-[10px] font-semibold text-brand-blue">
                              {count}
                            </span>
                          </div>
                        ))
                      )}
                    </div>
                  </div>
                </div>
              </div>

              <div className="mt-6 rounded-2xl border border-white/30 dark:border-white/10 bg-white/40 dark:bg-white/5 p-4">
                <div className="flex items-center justify-between gap-3">
                  <p className="text-xs font-semibold uppercase tracking-wider text-brand-blue dark:text-white">
                    Resumo IA
                  </p>
                  <span className="rounded-full bg-brand-green/20 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wider text-brand-green">
                    IA clinica
                  </span>
                </div>
                {summary ? (
                  <p className="mt-3 text-sm text-slate-700 dark:text-slate-200 whitespace-pre-line">
                    {summary}
                  </p>
                ) : (
                  <p className="mt-3 text-xs text-slate-500 dark:text-slate-400">
                    Gere um resumo para destacar riscos coletivos, prioridades e planejamento de
                    recursos.
                  </p>
                )}
                {summaryError ? (
                  <p className="mt-3 text-xs text-red-500 dark:text-red-300">{summaryError}</p>
                ) : null}
              </div>
            </section>
          </div>

          <section className="rounded-3xl border border-white/20 dark:border-white/10 bg-white/45 dark:bg-white/5 backdrop-blur-xl p-6 shadow-xl">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div>
                <h2 className="text-sm font-semibold uppercase tracking-wider text-brand-blue dark:text-white">
                  Prioridade clínica
                </h2>
                <p className="mt-1 text-xs text-slate-600 dark:text-slate-300">
                  Lista ordenada para visitas e encaminhamentos imediatos.
                </p>
              </div>
              <span className="rounded-full border border-brand-blue/30 dark:border-white/20 px-3 py-1 text-[11px] font-semibold uppercase tracking-wider text-brand-blue dark:text-white">
                {priorityCases.length} casos prioritários
              </span>
            </div>

            <div className="mt-4 grid gap-3 lg:grid-cols-2">
              {priorityCases.length === 0 ? (
                <div className="rounded-2xl border border-white/30 dark:border-white/10 bg-white/30 dark:bg-white/5 p-4 text-sm text-slate-600 dark:text-slate-300">
                  Nenhum caso prioritário ainda.
                </div>
              ) : (
                priorityCases.map((caseItem) => (
                  <div
                    key={caseItem.id}
                    className="flex items-center gap-3 rounded-2xl border border-white/30 dark:border-white/10 bg-white/30 dark:bg-white/5 p-4"
                  >
                    <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-brand-blue/10 text-brand-blue dark:bg-white/10 dark:text-white">
                      <BarChart3 className="h-5 w-5" />
                    </div>
                    <div className="flex-1">
                      <p className="text-sm font-semibold text-brand-blue dark:text-white">
                        {caseItem.displayName}
                      </p>
                      <p className="mt-1 text-xs text-slate-600 dark:text-slate-300">
                        {caseItem.insights.topFindings.length > 0
                          ? caseItem.insights.topFindings.join(', ')
                          : 'Sem achados relevantes'}
                      </p>
                      <div className="mt-2 flex flex-wrap items-center gap-2 text-[10px] uppercase tracking-wider text-slate-500 dark:text-slate-400">
                        {caseItem.insights.needs.length > 0 ? (
                          caseItem.insights.needs.map((need) => (
                            <span
                              key={need}
                              className="rounded-full bg-brand-blue/10 px-2 py-0.5 text-brand-blue dark:bg-white/10 dark:text-white"
                            >
                              {need}
                            </span>
                          ))
                        ) : (
                          <span>Sem necessidades</span>
                        )}
                      </div>
                    </div>
                    <span
                      className={[
                        'rounded-full px-3 py-1 text-[10px] font-semibold uppercase tracking-wider',
                        caseItem.insights.severity === 'Alta'
                          ? 'bg-red-200 text-red-600'
                          : caseItem.insights.severity === 'Media'
                            ? 'bg-brand-yellow/30 text-brand-blue'
                            : 'bg-brand-green/20 text-brand-green',
                      ].join(' ')}
                    >
                      {caseItem.insights.severity}
                    </span>
                  </div>
                ))
              )}
            </div>
          </section>
        </div>
      </main>
    </div>
  );
}
