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

type ModelType = 'yolo' | 'mask_rcnn';

type Finding = {
  label: string;
  confidence?: number;
  bbox?: number[];
  segmentation?: number[][];
  model?: string;
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
];

const SEVERITY_RULES = [
  {
    keywords: ['abscesso', 'cisto', 'fratura', 'lesao', 'reabsorcao', 'granuloma', 'tumor'],
    weight: 3,
    need: 'Cirurgia',
  },
  {
    keywords: ['carie', 'periodont', 'perda ossea', 'periapical', 'infeccao'],
    weight: 2,
    need: 'Periodontal',
  },
  {
    keywords: ['canal', 'endodont', 'polpa', 'apice', 'raiz'],
    weight: 1.5,
    need: 'Endodontia',
  },
  {
    keywords: ['restaur', 'obtur', 'protese', 'implante'],
    weight: 1,
    need: 'Reabilitacao',
  },
  {
    keywords: ['inclus', 'retido', 'impactado', 'erupcao'],
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
      : undefined;
  const model =
    (raw.model as string | undefined) ??
    (raw.model_type as string | undefined) ??
    (raw.source as string | undefined);

  return {
    label,
    confidence,
    bbox,
    segmentation,
    model,
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

  return {
    id: `${String(nameCandidate)}-${index}-${Math.random().toString(36).slice(2, 8)}`,
    name: String(nameCandidate),
    file: null,
    previewUrl: typeof previewUrl === 'string' ? previewUrl : null,
    status: 'done',
    findings,
    modelType,
    source: 'json',
  };
};

const normalizeLabel = (label: string, modelType: ModelType, normalizeTooth: boolean) => {
  if (!normalizeTooth || modelType !== 'mask_rcnn') {
    return label.trim();
  }
  return label.replace(/esmalte/gi, 'dente').trim();
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

export default function TriagemPage() {
  const router = useRouter();
  const { user, isLoading } = useSupabaseUser();
  const [tokenChecked, setTokenChecked] = useState(false);
  const [hasTokenAccess, setHasTokenAccess] = useState(false);
  const [isDarkMode, setIsDarkMode] = useState(false);
  const [modelType, setModelType] = useState<ModelType>('mask_rcnn');
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
  const casesRef = useRef<TriageCase[]>([]);
  const isVisitor = hasTokenAccess && !user;
  const displayName = user?.user_metadata?.full_name || user?.email?.split('@')[0] || 'Usuario';
  const profileHref = isVisitor ? '/perfil/visitante' : '/perfil';
  const profileLabel = isVisitor ? 'Perfil (Visitante)' : 'Perfil';
  const baseApiUrl = process.env.NEXT_PUBLIC_MODAL_API_URL ?? '';

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
        if (caseItem.previewUrl) {
          URL.revokeObjectURL(caseItem.previewUrl);
        }
      });
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

    const normalizedFindings = caseItem.findings.map((finding) => ({
      ...finding,
      label: normalizeLabel(finding.label, caseItem.modelType, normalizeToothLabel),
    }));

    const visibleFindings = hideAnatomy
      ? normalizedFindings.filter((finding) => !isAnatomyLabel(finding.label))
      : normalizedFindings;

    if (visibleFindings.length === 0) {
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

    visibleFindings.forEach((finding) => {
      const label = finding.label.toLowerCase();
      const confidence = finding.confidence ?? 0.65;
      let matched = false;

      SEVERITY_RULES.forEach((rule) => {
        if (rule.keywords.some((keyword) => label.includes(keyword))) {
          matched = true;
          score += rule.weight * confidence;
          needs.add(rule.need);
        }
      });

      if (!matched) {
        score += 0.2 * confidence;
      }

      counts.set(finding.label, (counts.get(finding.label) ?? 0) + 1);
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

  const casesWithInsights = useMemo(() => {
    return cases.map((caseItem) => ({
      ...caseItem,
      insights: buildInsights(caseItem),
    }));
  }, [cases, hideAnatomy, normalizeToothLabel]);

  const totals = useMemo(() => {
    const analyzed = casesWithInsights.filter((caseItem) => caseItem.status === 'done');
    const totalFindings = analyzed.reduce(
      (sum, caseItem) => sum + caseItem.insights.visibleFindings.length,
      0,
    );
    const severityCounts = {
      Alta: analyzed.filter((caseItem) => caseItem.insights.severity === 'Alta').length,
      Media: analyzed.filter((caseItem) => caseItem.insights.severity === 'Media').length,
      Baixa: analyzed.filter((caseItem) => caseItem.insights.severity === 'Baixa').length,
    };
    const needsCounts = new Map<string, number>();
    const labelCounts = new Map<string, number>();

    analyzed.forEach((caseItem) => {
      caseItem.insights.needs.forEach((need) => {
        needsCounts.set(need, (needsCounts.get(need) ?? 0) + 1);
      });
      caseItem.insights.visibleFindings.forEach((finding) => {
        const label = finding.label;
        labelCounts.set(label, (labelCounts.get(label) ?? 0) + 1);
      });
    });

    const topNeeds = Array.from(needsCounts.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, 5);

    const topLabels = Array.from(labelCounts.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, 6);

    return {
      analyzed,
      totalFindings,
      severityCounts,
      topNeeds,
      topLabels,
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
    return [
      `Total de casos analisados: ${totals.analyzed.length}`,
      `Achados relevantes (sem estruturas): ${totals.totalFindings}`,
      `Severidade alta: ${totals.severityCounts.Alta}`,
      `Severidade media: ${totals.severityCounts.Media}`,
      `Severidade baixa: ${totals.severityCounts.Baixa}`,
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

  const handleJsonImport = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }
    setImportError(null);
    try {
      const text = await file.text();
      const payload = JSON.parse(text) as unknown;
      const rawCases = extractCaseArray(payload);
      if (rawCases.length === 0) {
        throw new Error('Nenhum caso encontrado no JSON.');
      }
      const normalizedCases = rawCases.map((raw, index) =>
        normalizeCaseFromJson(raw as Record<string, unknown>, index, modelType),
      );
      setCases((prev) => [...prev, ...normalizedCases]);
      setBatchError(null);
      setSummary('');
      setSummaryError(null);
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Falha ao importar JSON.';
      setImportError(message);
    } finally {
      event.target.value = '';
    }
  };

  const removeCase = (id: string) => {
    setCases((prev) => {
      const target = prev.find((caseItem) => caseItem.id === id);
      if (target?.previewUrl) {
        URL.revokeObjectURL(target.previewUrl);
      }
      return prev.filter((caseItem) => caseItem.id !== id);
    });
  };

  const clearBatch = () => {
    setCases((prev) => {
      prev.forEach((caseItem) => {
        if (caseItem.previewUrl) {
          URL.revokeObjectURL(caseItem.previewUrl);
        }
      });
      return [];
    });
    setBatchError(null);
    setImportError(null);
    setSummary('');
    setSummaryError(null);
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
                  Triagem Epidemiologica
                </p>
                <h1 className="mt-2 text-2xl font-semibold text-brand-blue dark:text-white">
                  Panorama clinico para priorizar comunidades remotas
                </h1>
                <p className="mt-2 text-sm text-slate-600 dark:text-slate-300 max-w-2xl">
                  Carregue grandes lotes de radiografias, identifique necessidades criticas e gere
                  uma fila de prioridade baseada em gravidade.
                </p>
              </div>
              <div className="flex flex-wrap items-center gap-2">
                <div className="flex rounded-full border border-white/30 dark:border-white/10 bg-white/70 dark:bg-slate-900/60 p-1 text-[11px] font-semibold uppercase tracking-widest text-brand-blue dark:text-white">
                  <button
                    type="button"
                    onClick={() => setModelType('mask_rcnn')}
                    className={[
                      'px-3 py-1 rounded-full transition-all',
                      modelType === 'mask_rcnn'
                        ? 'bg-white/80 text-brand-blue shadow-sm dark:bg-white/15 dark:text-white'
                        : 'text-brand-blue/60 hover:text-brand-blue dark:text-white/60 dark:hover:text-white',
                    ].join(' ')}
                  >
                    Mask R-CNN
                  </button>
                  <button
                    type="button"
                    onClick={() => setModelType('yolo')}
                    className={[
                      'px-3 py-1 rounded-full transition-all',
                      modelType === 'yolo'
                        ? 'bg-white/80 text-brand-blue shadow-sm dark:bg-white/15 dark:text-white'
                        : 'text-brand-blue/60 hover:text-brand-blue dark:text-white/60 dark:hover:text-white',
                    ].join(' ')}
                  >
                    YOLOv11
                  </button>
                </div>
                <button
                  type="button"
                  onClick={handleAnalyzeBatch}
                  disabled={isBatchAnalyzing || cases.length === 0}
                  className="rounded-full bg-brand-green px-4 py-2 text-xs font-semibold text-white shadow-md transition-all disabled:opacity-60 disabled:cursor-not-allowed"
                >
                  {isBatchAnalyzing
                    ? `Analisando ${batchProgress.completed}/${batchProgress.total}`
                    : 'Analisar lote'}
                </button>
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
                  Arraste um lote de radiografias ou clique para enviar.
                </p>
                <p className="mt-1 text-xs text-slate-500 dark:text-slate-400">
                  Suporta JPG/PNG. Ideal para triagem de 50+ pacientes.
                </p>
                <label className="mt-4 inline-flex cursor-pointer items-center gap-2 rounded-full bg-brand-blue px-4 py-2 text-xs font-semibold text-white shadow-md">
                  <UploadCloud className="h-4 w-4" />
                  Selecionar arquivos
                  <input
                    type="file"
                    accept="image/*"
                    multiple
                    onChange={handleFileSelection}
                    className="hidden"
                  />
                </label>
                <label className="mt-3 inline-flex cursor-pointer items-center gap-2 rounded-full border border-brand-blue/40 px-4 py-2 text-xs font-semibold text-brand-blue shadow-sm hover:bg-white/50 dark:border-white/20 dark:text-white dark:hover:bg-white/10">
                  <FileSearch className="h-4 w-4" />
                  Importar JSON
                  <input
                    type="file"
                    accept="application/json,.json"
                    onChange={handleJsonImport}
                    className="hidden"
                  />
                </label>
              </div>

              <div className="mt-4 flex flex-wrap items-center gap-2 text-[11px] uppercase tracking-wider text-slate-500 dark:text-slate-400">
                <button
                  type="button"
                  onClick={() => setNormalizeToothLabel((prev) => !prev)}
                  className="rounded-full border border-brand-blue/30 dark:border-white/20 px-3 py-1 font-semibold text-brand-blue dark:text-white hover:bg-white/40 dark:hover:bg-white/10 transition-all"
                >
                  {normalizeToothLabel ? 'Normalizando dente/esmalte' : 'Rotulos originais'}
                </button>
                <span>
                  {cases.length} radiografias carregadas
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
                      className="flex items-center gap-3 rounded-2xl border border-white/30 dark:border-white/10 bg-white/30 dark:bg-white/5 p-3"
                    >
                      <div className="h-12 w-12 overflow-hidden rounded-xl bg-white/60 dark:bg-slate-900/60 flex items-center justify-center">
                        {caseItem.previewUrl ? (
                          <img
                            src={caseItem.previewUrl}
                            alt={caseItem.name}
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
                              {caseItem.name}
                            </p>
                            <p className="text-[11px] uppercase tracking-wider text-slate-500 dark:text-slate-400">
                              {caseItem.source === 'json' ? 'JSON' : 'Upload'} ·{' '}
                              {caseItem.status === 'done'
                                ? `${caseItem.insights.visibleFindings.length} achados`
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
                        {caseItem.status === 'done' && caseItem.insights.topFindings.length > 0 ? (
                          <p className="mt-1 text-xs text-slate-600 dark:text-slate-300">
                            {caseItem.insights.topFindings.join(', ')}
                          </p>
                        ) : null}
                        {!caseItem.file ? (
                          <p className="mt-1 text-[11px] uppercase tracking-wider text-slate-400 dark:text-slate-500">
                            Somente JSON importado
                          </p>
                        ) : null}
                        {caseItem.status === 'error' && caseItem.error ? (
                          <p className="mt-1 text-xs text-red-500 dark:text-red-300">{caseItem.error}</p>
                        ) : null}
                      </div>
                      <div className="flex flex-col items-end gap-2">
                        <button
                          type="button"
                          onClick={() => analyzeCase(caseItem)}
                          disabled={isBatchAnalyzing || caseItem.status === 'analyzing' || !caseItem.file}
                          className="rounded-full border border-brand-blue/30 dark:border-white/20 px-3 py-1 text-[10px] font-semibold uppercase tracking-wider text-brand-blue dark:text-white hover:bg-white/40 dark:hover:bg-white/10 transition-all disabled:opacity-50"
                        >
                          {caseItem.status === 'done' ? 'Reanalisar' : 'Analisar'}
                        </button>
                        <button
                          type="button"
                          onClick={() => removeCase(caseItem.id)}
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
            </section>

            <section className="rounded-3xl border border-white/20 dark:border-white/10 bg-white/45 dark:bg-white/5 backdrop-blur-xl p-6 shadow-xl">
              <div className="flex flex-wrap items-center justify-between gap-3">
                <div>
                  <h2 className="text-sm font-semibold uppercase tracking-wider text-brand-blue dark:text-white">
                    Visao epidemiologica
                  </h2>
                  <p className="mt-1 text-xs text-slate-600 dark:text-slate-300">
                    Indicadores principais para tomada de decisao.
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
                    Severidade clinica
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
                      Distribuicao das principais suspeitas.
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
                  Prioridade clinica
                </h2>
                <p className="mt-1 text-xs text-slate-600 dark:text-slate-300">
                  Lista ordenada para visitas e encaminhamentos imediatos.
                </p>
              </div>
              <span className="rounded-full border border-brand-blue/30 dark:border-white/20 px-3 py-1 text-[11px] font-semibold uppercase tracking-wider text-brand-blue dark:text-white">
                {priorityCases.length} casos prioritarios
              </span>
            </div>

            <div className="mt-4 grid gap-3 lg:grid-cols-2">
              {priorityCases.length === 0 ? (
                <div className="rounded-2xl border border-white/30 dark:border-white/10 bg-white/30 dark:bg-white/5 p-4 text-sm text-slate-600 dark:text-slate-300">
                  Nenhum caso prioritario ainda.
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
                        {caseItem.name}
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
