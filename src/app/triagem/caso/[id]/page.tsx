'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import { ArrowLeft, FileSearch } from 'lucide-react';
import { useParams, useRouter } from 'next/navigation';

type FindingCategory =
  | 'structure'
  | 'tooth'
  | 'pathology'
  | 'anatomy'
  | 'treatment'
  | 'other';

type Finding = {
  label: string;
  confidence?: number;
  bbox?: number[];
  segmentation?: number[][];
  sourceModel?: string;
  displayLabel?: string;
  canonicalLabel?: string;
  category?: FindingCategory;
  toothId?: number;
  toothType?: string;
  toothGroup?: string;
};

type CaseInsight = {
  severity: 'Alta' | 'Media' | 'Baixa' | 'Aguardando';
  score: number;
  needs: string[];
  topFindings: string[];
  visibleFindings: Finding[];
  normalizedFindings: Finding[];
};

type LabelMeta = {
  display: string;
  canonical: string;
  category: FindingCategory;
  toothGroup?: string;
};

type ClinicalProfile = {
  name: string;
  age: number;
  gender: 'Feminino' | 'Masculino' | 'Outros';
  complaint: string;
  symptoms: string[];
  alerts: string[];
};

type StoredCase = {
  id: string;
  name: string;
  displayName: string;
  caseNumber: number;
  patientName: string;
  previewUrl?: string | null;
  modelType: string;
  insights: CaseInsight;
  clinicalProfile: ClinicalProfile;
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

const STRUCTURE_CATEGORIES = new Set<FindingCategory>(['structure', 'anatomy', 'tooth']);

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

const buildClinicalProfile = (seed: number, severity: CaseInsight['severity']): ClinicalProfile => {
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
  const baseCount = severity === 'Alta' ? 3 : severity === 'Media' ? 2 : 1;
  const symptoms = routineOnly ? [] : pickUnique(symptomsPool, seed, baseCount);
  const alerts = severity === 'Alta' && !routineOnly ? pickUnique(alertsPool, seed + 11, 2) : [];

  return {
    name,
    age,
    gender,
    complaint,
    symptoms,
    alerts,
  };
};

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

const normalizeFinding = (raw: Record<string, unknown>): Finding | null => {
  const labelCandidate = raw.label ?? raw.class ?? raw.name ?? raw.category ?? raw.tipo ?? raw.tipo_lesao;
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
  const toothId = toNumber((raw.tooth_id ?? raw.toothId) as unknown);

  return {
    label,
    confidence,
    bbox,
    segmentation,
    sourceModel,
    toothId,
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
    return { ...finding, toothType: best.toothLabel, toothGroup: best.toothGroup };
  });
};

const applyCariesDepth = (findings: Finding[]) => {
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

const enrichFindings = (findings: Finding[]) => {
  const enriched = findings.map((finding) => {
    const meta = getLabelMeta(finding.label);
    return {
      ...finding,
      displayLabel: meta.display,
      canonicalLabel: meta.canonical,
      category: meta.category,
      toothGroup: meta.toothGroup,
      toothType: meta.category === 'tooth' ? meta.display : finding.toothType,
    };
  });
  return applyCariesDepth(attachToothMatches(enriched));
};

const isHiddenStructure = (finding: Finding) => {
  if (finding.category) {
    return STRUCTURE_CATEGORIES.has(finding.category);
  }
  return ANATOMY_KEYWORDS.some((keyword) => finding.label.toLowerCase().includes(keyword));
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

const buildInsights = (findings: Finding[]): CaseInsight => {
  const normalizedFindings = enrichFindings(findings);
  const visibleFindings = normalizedFindings.filter((finding) => !isHiddenStructure(finding));
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
  imageBaseUrl?: string,
) => {
  const findingsRaw = raw.findings ?? raw.results ?? raw.detections ?? raw.predictions;
  const findingsArray = Array.isArray(findingsRaw) ? findingsRaw : [];
  const findings = findingsArray
    .map((item) => (item && typeof item === 'object' ? normalizeFinding(item as Record<string, unknown>) : null))
    .filter((item): item is Finding => Boolean(item));
  const nameCandidate = raw.name ?? raw.filename ?? raw.file ?? raw.id ?? `Caso ${index + 1}`;
  const previewUrl =
    (raw.previewUrl as string | undefined) ??
    (raw.preview_url as string | undefined) ??
    (raw.imageUrl as string | undefined) ??
    (raw.image_url as string | undefined) ??
    (raw.image as string | undefined);
  const baseUrl = imageBaseUrl ? (imageBaseUrl.endsWith('/') ? imageBaseUrl : `${imageBaseUrl}/`) : '';
  const resolvedPreview =
    typeof previewUrl === 'string'
      ? previewUrl
      : baseUrl
        ? `${baseUrl}${encodeURIComponent(String(nameCandidate))}`
        : null;
  return {
    id: `${String(nameCandidate)}-${index}`,
    name: String(nameCandidate),
    previewUrl: resolvedPreview,
    findings,
    modelType: (raw.modelType ?? raw.model_type ?? raw.model ?? 'combined') as string,
  };
};

const buildCaseDisplayName = (index: number) => `Paciente ${index + 1}`;

const getSourceKind = (finding: Finding) => {
  const source = (finding.sourceModel ?? '').toLowerCase();
  if (source.includes('yolo')) {
    return 'yolo';
  }
  if (source.includes('detectron')) {
    return 'detectron';
  }
  return 'other';
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

export default function TriagemCasoPage() {
  const router = useRouter();
  const params = useParams();
  const caseId = Array.isArray(params?.id) ? params?.id[0] : params?.id;
  const batchUrl =
    process.env.NEXT_PUBLIC_TRIAGEM_BATCH_URL ?? '/batch_test/triagem_batch_results.json';
  const batchImageBase =
    process.env.NEXT_PUBLIC_TRIAGEM_BATCH_IMAGE_BASE ?? '/batch_test/';
  const [caseData, setCaseData] = useState<StoredCase | null>(null);
  const [loadState, setLoadState] = useState<'idle' | 'loading' | 'error'>('idle');
  const [loadError, setLoadError] = useState<string | null>(null);
  const [isPreviewReady, setIsPreviewReady] = useState(false);
  const previewImageRef = useRef<HTMLImageElement | null>(null);
  const previewCanvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    if (!caseId) {
      return;
    }
    let isActive = true;
    const loadCase = async () => {
      const caseNumber = Number(caseId);
      if (!Number.isFinite(caseNumber) || caseNumber <= 0) {
        setLoadState('error');
        setLoadError('ID de caso invalido.');
        return;
      }

      setLoadState('loading');
      setLoadError(null);
      const storageKey = `triagem-case-${caseNumber}`;
      const raw = localStorage.getItem(storageKey);
      if (raw) {
        try {
          const parsed = JSON.parse(raw) as StoredCase;
          if (isActive) {
            setCaseData(parsed);
            setLoadState('idle');
          }
          return;
        } catch (error) {
          console.error(error);
        }
      }

      try {
        const response = await fetch(batchUrl, { cache: 'no-store' });
        if (!response.ok) {
          throw new Error('Nao foi possivel carregar o lote local.');
        }
        const payload = await response.json();
        const cases = extractCaseArray(payload);
        const index = caseNumber - 1;
        if (!cases[index] || typeof cases[index] !== 'object') {
          throw new Error('Caso nao encontrado no lote local.');
        }
        const normalized = normalizeCaseFromJson(
          cases[index] as Record<string, unknown>,
          index,
          batchImageBase,
        );
        const insights = buildInsights(normalized.findings);
        const seed = hashSeed(`${normalized.name}-${index}`);
        const displayName = buildCaseDisplayName(index);
        const patientName = pickPatientName(seed);
        const clinicalProfile = buildClinicalProfile(seed, insights.severity);
        const payloadCase: StoredCase = {
          id: normalized.id,
          name: normalized.name,
          displayName,
          caseNumber,
          patientName,
          previewUrl: normalized.previewUrl,
          modelType: normalized.modelType,
          insights,
          clinicalProfile,
        };
        localStorage.setItem(storageKey, JSON.stringify(payloadCase));
        if (isActive) {
          setCaseData(payloadCase);
          setLoadState('idle');
        }
      } catch (error) {
        if (!isActive) {
          return;
        }
        const message =
          error instanceof Error ? error.message : 'Nao foi possivel carregar o caso.';
        setLoadError(message);
        setLoadState('error');
      }
    };
    loadCase();
    return () => {
      isActive = false;
    };
  }, [caseId, batchUrl, batchImageBase]);

  const findings = useMemo(() => {
    if (!caseData) {
      return [];
    }
    return caseData.insights.normalizedFindings ?? [];
  }, [caseData]);

  const visibleFindings = useMemo(() => {
    return findings.filter(
      (finding) => finding.category === 'pathology' || finding.category === 'treatment',
    );
  }, [findings]);

  useEffect(() => {
    const canvas = previewCanvasRef.current;
    const img = previewImageRef.current;
    if (!canvas || !img || !caseData || !isPreviewReady) {
      return;
    }

    const draw = () => {
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

      findings.forEach((finding) => {
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

        const label = finding.displayLabel ?? finding.label;
        const sourceKind = getSourceKind(finding);
        const sourceShort = sourceKind === 'detectron' ? 'D' : sourceKind === 'yolo' ? 'Y' : 'O';
        const toothText = finding.toothId != null ? `·${finding.toothId}` : finding.toothType ? `·${finding.toothType}` : '';
        const labelText = `${label}${toothText}·${sourceShort}`;

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

        ctx.font = '11px ui-sans-serif, system-ui, -apple-system, sans-serif';
        const metrics = ctx.measureText(labelText);
        const padding = 3;
        const textWidth = metrics.width + padding * 2;
        const textHeight = 14;
        ctx.fillStyle = 'rgba(15, 23, 42, 0.65)';
        ctx.fillRect(labelX, labelY - textHeight, textWidth, textHeight);
        ctx.fillStyle = '#f8fafc';
        ctx.fillText(labelText, labelX + padding, labelY - 3);
      });
    };

    draw();
    const handleResize = () => draw();
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [caseData, findings, isPreviewReady]);

  if (!caseData) {
    return (
      <div className="min-h-screen bg-background text-gray-900 dark:text-gray-100">
        <header className="p-6">
          <button
            type="button"
            onClick={() => router.push('/triagem')}
            className="flex h-10 w-10 items-center justify-center rounded-full glass-panel text-brand-blue dark:text-brand-yellow shadow-lg"
          >
            <ArrowLeft className="h-5 w-5" />
          </button>
        </header>
        <main className="mx-auto max-w-3xl px-4 pb-10">
          <div className="rounded-3xl border border-white/20 bg-white/70 p-6 text-sm text-slate-600 dark:border-white/10 dark:bg-white/5 dark:text-slate-200">
            {loadState === 'loading'
              ? 'Carregando ficha clinica...'
              : loadError || 'Abra este caso pelo painel de triagem para carregar a ficha completa.'}
          </div>
        </main>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background text-gray-900 dark:text-gray-100">
      <header className="p-6 flex items-center justify-between gap-4">
        <button
          type="button"
          onClick={() => router.push('/triagem')}
          className="flex h-10 w-10 items-center justify-center rounded-full glass-panel text-brand-blue dark:text-brand-yellow shadow-lg"
        >
          <ArrowLeft className="h-5 w-5" />
        </button>
        <div className="flex flex-col text-right">
          <p className="text-xs uppercase tracking-[0.3em] text-brand-blue/60 dark:text-white/60">
            Caso clínico
          </p>
          <h1 className="text-lg font-semibold text-brand-blue dark:text-white">
            {caseData.displayName} · {caseData.patientName}
          </h1>
        </div>
      </header>

      <main className="mx-auto flex w-full max-w-6xl flex-col gap-6 px-4 pb-10">
        <section className="rounded-3xl border border-white/20 bg-white/60 p-6 shadow-xl dark:border-white/10 dark:bg-white/5">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div>
              <p className="text-xs uppercase tracking-[0.3em] text-brand-blue/60 dark:text-white/60">
                Radiografia panorâmica
              </p>
              <h2 className="mt-2 text-xl font-semibold text-brand-blue dark:text-white">
                Diagnóstico visual completo
              </h2>
            </div>
            <span className="rounded-full bg-brand-yellow/20 px-3 py-1 text-[10px] font-semibold uppercase tracking-wider text-brand-blue">
              Dados fictícios
            </span>
          </div>
          <div className="mt-4 relative overflow-hidden rounded-2xl border border-white/40 bg-white/60 dark:border-white/10 dark:bg-slate-900/60 min-h-[420px] flex items-center justify-center">
            {caseData.previewUrl ? (
              <>
                <img
                  ref={previewImageRef}
                  src={caseData.previewUrl}
                  alt={caseData.displayName}
                  onLoad={() => setIsPreviewReady(true)}
                  className="max-h-[560px] w-full object-contain"
                />
                <canvas ref={previewCanvasRef} className="absolute inset-0 h-full w-full pointer-events-none" />
              </>
            ) : (
              <div className="text-xs text-slate-500 dark:text-slate-400">
                Imagem não disponível para este caso.
              </div>
            )}
          </div>
        </section>

        <div className="grid gap-6 lg:grid-cols-[1.2fr_1fr]">
          <section className="rounded-3xl border border-white/20 bg-white/60 p-6 shadow-xl dark:border-white/10 dark:bg-white/5">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-xs font-semibold uppercase tracking-wider text-brand-blue dark:text-white">
                  Ficha clínica completa (fictícia)
                </p>
                <p className="mt-1 text-xs text-slate-500 dark:text-slate-400">
                  Dados simulados apenas para demonstração.
                </p>
              </div>
              <span className="rounded-full bg-brand-green/20 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wider text-brand-green">
                {caseData.insights.severity}
              </span>
            </div>

            <div className="mt-4 rounded-2xl border border-white/30 bg-white/70 p-4 text-sm text-brand-blue dark:border-white/10 dark:bg-slate-900/40 dark:text-white">
              {caseData.patientName} · {caseData.displayName}
            </div>

            <div className="mt-4 grid gap-4 sm:grid-cols-2 text-sm">
              <div>
                <p className="text-[11px] uppercase tracking-wider text-slate-500 dark:text-slate-400">
                  Idade
                </p>
                <p className="mt-1 font-semibold">{caseData.clinicalProfile.age} anos</p>
              </div>
              <div>
                <p className="text-[11px] uppercase tracking-wider text-slate-500 dark:text-slate-400">
                  Gênero
                </p>
                <p className="mt-1 font-semibold">{caseData.clinicalProfile.gender}</p>
              </div>
            </div>

            <div className="mt-4">
              <p className="text-[11px] uppercase tracking-wider text-slate-500 dark:text-slate-400">
                Queixa principal
              </p>
              <p className="mt-1 text-sm font-semibold">{caseData.clinicalProfile.complaint}</p>
            </div>

            <div className="mt-4">
              <p className="text-[11px] uppercase tracking-wider text-slate-500 dark:text-slate-400">
                Sintomas relatados
              </p>
              <div className="mt-2 flex flex-wrap gap-2">
                {caseData.clinicalProfile.symptoms.length === 0 ? (
                  <span className="text-xs text-slate-500 dark:text-slate-400">
                    Sem sintomas relatados
                  </span>
                ) : (
                  caseData.clinicalProfile.symptoms.map((symptom) => (
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

            {caseData.clinicalProfile.alerts.length > 0 ? (
              <div className="mt-4">
                <p className="text-[11px] uppercase tracking-wider text-slate-500 dark:text-slate-400">
                  Sinais de alerta
                </p>
                <div className="mt-2 flex flex-wrap gap-2">
                  {caseData.clinicalProfile.alerts.map((alert) => (
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
          </section>

          <section className="rounded-3xl border border-white/20 bg-white/60 p-6 shadow-xl dark:border-white/10 dark:bg-white/5">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-xs font-semibold uppercase tracking-wider text-brand-blue dark:text-white">
                  Achados principais
                </p>
                <p className="mt-1 text-xs text-slate-500 dark:text-slate-400">
                  Resumo das principais suspeitas clínicas.
                </p>
              </div>
              <span className="rounded-full bg-brand-blue/10 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wider text-brand-blue dark:bg-white/10 dark:text-white">
                {visibleFindings.length} achados
              </span>
            </div>

            <div className="mt-4 space-y-3 text-xs text-slate-600 dark:text-slate-300">
              {visibleFindings.length === 0 ? (
                <p className="text-xs text-slate-500 dark:text-slate-400">
                  Nenhum achado clínico relevante.
                </p>
              ) : (
                visibleFindings.slice(0, 12).map((finding, index) => (
                  <div
                    key={`${finding.label}-${index}`}
                    className="flex items-center justify-between rounded-xl border border-white/40 bg-white/70 px-3 py-2 dark:border-white/10 dark:bg-slate-900/40"
                  >
                    <span className="font-semibold">{finding.displayLabel ?? finding.label}</span>
                    <span className="text-[10px] uppercase tracking-wider text-slate-500 dark:text-slate-400">
                      {finding.toothId ? `FDI ${finding.toothId}` : finding.toothType ?? 'Sem dente'}
                    </span>
                  </div>
                ))
              )}
            </div>

            <div className="mt-5 rounded-2xl border border-white/30 bg-white/70 p-4 text-xs text-slate-600 dark:border-white/10 dark:bg-slate-900/40 dark:text-slate-200">
              <div className="flex items-center gap-2 text-[11px] font-semibold uppercase tracking-wider text-brand-blue dark:text-white">
                <FileSearch className="h-4 w-4" />
                Prioridade clínica
              </div>
              <p className="mt-2 text-sm font-semibold">
                {caseData.insights.severity} · Score {caseData.insights.score.toFixed(1)}
              </p>
              <p className="mt-1 text-xs text-slate-500 dark:text-slate-400">
                Necessidades principais: {caseData.insights.needs.join(', ') || 'Não definido'}
              </p>
            </div>
          </section>
        </div>
      </main>
    </div>
  );
}
