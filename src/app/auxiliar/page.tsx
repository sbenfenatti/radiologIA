'use client';

import { useEffect, useMemo, useState } from 'react';
import { ArrowLeft, LogOut, Moon, Sun, UserRound } from 'lucide-react';
import Link from 'next/link';
import { usePathname, useRouter, useSearchParams } from 'next/navigation';
import { useSupabaseUser } from '@/hooks/use-supabase-user';
import { useSessionExpiry } from '@/hooks/use-session-expiry';
import { supabase } from '@/lib/supabase/client';
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover';
import FindingsViewer, { normalizeFindings, type RawFinding } from '@/components/triagem/FindingsViewer';

type ChatRole = 'user' | 'model' | 'system';

type ChatMessage = {
  role: ChatRole;
  text: string;
};

type Finding = {
  label: string;
  confidence?: number;
  bbox?: number[];
  segmentation?: number[][];
  model?: string;
  source_model?: string;
  sourceModel?: string;
  tooth_id?: number;
  toothId?: number;
};

const TOOTH_EXPECTED: Record<string, number> = {
  molar_sup: 4,
  molar_inf: 4,
  pre_molar_sup: 4,
  pre_molar_inf: 4,
  incisivo_central_sup: 2,
  incisivo_central_inf: 2,
  incisivo_lateral_sup: 2,
  incisivo_lateral_inf: 2,
  canino_sup: 2,
  canino_inf: 2,
  terceiro_molar_sup: 2,
  terceiro_molar_inf: 2,
};

const TOOTH_LABELS: Record<string, string> = {
  molar_sup: 'Molar superior',
  molar_inf: 'Molar inferior',
  pre_molar_sup: 'Pre-molar superior',
  pre_molar_inf: 'Pre-molar inferior',
  incisivo_central_sup: 'Incisivo central superior',
  incisivo_central_inf: 'Incisivo central inferior',
  incisivo_lateral_sup: 'Incisivo lateral superior',
  incisivo_lateral_inf: 'Incisivo lateral inferior',
  canino_sup: 'Canino superior',
  canino_inf: 'Canino inferior',
  terceiro_molar_sup: '3o molar superior',
  terceiro_molar_inf: '3o molar inferior',
};

const formatCountList = (items: { label: string; count: number }[]) =>
  items.map((item) => `${item.label} (${item.count}x)`).join(', ');

export default function AuxiliarClinicoPage() {
  const router = useRouter();
  const { user, isLoading } = useSupabaseUser();
  const [tokenChecked, setTokenChecked] = useState(false);
  const [hasTokenAccess, setHasTokenAccess] = useState(false);
  const [isDarkMode, setIsDarkMode] = useState(false);
  const [activeTab, setActiveTab] = useState<'chat' | 'rx'>('rx');
  const [isChatExpanded, setIsChatExpanded] = useState(false);
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imageMeta, setImageMeta] = useState<{ width: number; height: number } | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisError, setAnalysisError] = useState<string | null>(null);
  const [findings, setFindings] = useState<Finding[]>([]);
  const [chatHistory, setChatHistory] = useState<ChatMessage[]>([
    { role: 'system', text: 'Aguardando contexto da imagem.' },
  ]);
  const [chatInput, setChatInput] = useState('');
  const [isChatSending, setIsChatSending] = useState(false);
  const [copyStatus, setCopyStatus] = useState<string | null>(null);
  const [chatCopyStatus, setChatCopyStatus] = useState<string | null>(null);
  const [lastChatExchange, setLastChatExchange] = useState<{ request: unknown; response?: unknown } | null>(null);
  const isVisitor = hasTokenAccess && !user;
  const displayName = user?.user_metadata?.full_name || user?.email?.split('@')[0] || 'Usuário';
  const profileHref = isVisitor ? '/perfil/visitante' : '/perfil';
  const profileLabel = isVisitor ? 'Perfil (Visitante)' : 'Perfil';
  const isAuthenticated = Boolean(user || hasTokenAccess);
  const authIdentity = user?.id ?? (hasTokenAccess ? 'visitor' : null);
  const baseApiUrl = process.env.NEXT_PUBLIC_MODAL_API_URL ?? '';
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const debugEnabled = useMemo(() => {
    if (pathname?.includes('/debug')) {
      return true;
    }
    if (!searchParams) {
      return false;
    }
    const debugParam = searchParams.get('debug');
    return (
      debugParam === '1' ||
      debugParam === 'true' ||
      searchParams.has('debug') ||
      searchParams.has('debug1')
    );
  }, [pathname, searchParams]);

  const normalizeImageFile = (file: File) =>
    new Promise<File>((resolve, reject) => {
      const url = URL.createObjectURL(file);
      const img = new Image();
      img.onload = () => {
        const canvas = document.createElement('canvas');
        canvas.width = img.naturalWidth;
        canvas.height = img.naturalHeight;
        const ctx = canvas.getContext('2d');
        if (!ctx) {
          URL.revokeObjectURL(url);
          reject(new Error('Canvas indisponivel.'));
          return;
        }
        ctx.drawImage(img, 0, 0);
        canvas.toBlob(
          (blob) => {
            URL.revokeObjectURL(url);
            if (!blob) {
              reject(new Error('Falha ao normalizar imagem.'));
              return;
            }
            resolve(new File([blob], file.name, { type: blob.type || file.type }));
          },
          file.type || 'image/png',
          0.95,
        );
      };
      img.onerror = () => {
        URL.revokeObjectURL(url);
        reject(new Error('Falha ao carregar imagem.'));
      };
      img.src = url;
    });

  const tabClassName = (active: boolean) =>
    [
      'px-4 py-2 text-sm font-semibold uppercase tracking-wider rounded-full transition-all',
      active
        ? 'bg-white/70 text-brand-blue shadow-sm dark:bg-white/10 dark:text-white'
        : 'text-brand-blue/70 hover:text-brand-blue dark:text-white/60 dark:hover:text-white',
    ].join(' ');

  const renderBoldSegments = (line: string) => {
    const segments = line.split(/(\*\*[^*]+\*\*)/g);
    return segments.map((segment, index) => {
      if (segment.startsWith('**') && segment.endsWith('**') && segment.length > 4) {
        return <strong key={`b-${index}`}>{segment.slice(2, -2)}</strong>;
      }
      return <span key={`t-${index}`}>{segment}</span>;
    });
  };

  const renderChatText = (text: string) => {
    const lines = text.split('\n');
    return lines.map((line, index) => (
      <span key={`line-${index}`}>
        {renderBoldSegments(line)}
        {index < lines.length - 1 ? <br /> : null}
      </span>
    ));
  };

  const renderTypingIndicator = () => (
    <div className="text-left">
      <p className="text-[11px] tracking-wider text-slate-500 dark:text-slate-400">
        radiologIA
      </p>
      <div className="mt-1 inline-flex items-center gap-1 rounded-2xl border border-white/30 bg-white/60 px-3 py-2 text-slate-700 shadow-sm dark:border-white/10 dark:bg-slate-900/50 dark:text-slate-200">
        <span className="typing-dot inline-block h-2 w-2 rounded-full bg-brand-blue dark:bg-white/80" style={{ animationDelay: '0ms' }}></span>
        <span className="typing-dot inline-block h-2 w-2 rounded-full bg-brand-blue dark:bg-white/80" style={{ animationDelay: '150ms' }}></span>
        <span className="typing-dot inline-block h-2 w-2 rounded-full bg-brand-blue dark:bg-white/80" style={{ animationDelay: '300ms' }}></span>
      </div>
    </div>
  );

  const chatContext = useMemo(() => {
    if (findings.length === 0) {
      return '';
    }
    const groups = new Map<string, { label: string; count: number; confidence?: number }>();
    findings.forEach((finding) => {
      const key = finding.label.trim().toLowerCase();
      const entry = groups.get(key) ?? { label: finding.label, count: 0, confidence: undefined };
      entry.count += 1;
      if (finding.confidence != null) {
        entry.confidence =
          entry.confidence == null ? finding.confidence : Math.max(entry.confidence, finding.confidence);
      }
      groups.set(key, entry);
    });
    return Array.from(groups.values())
      .sort((a, b) => b.count - a.count)
      .map((finding) => {
        const count = finding.count > 1 ? ` (${finding.count} regioes)` : '';
        const confidence = finding.confidence != null
          ? ` (Confianca max: ${(finding.confidence * 100).toFixed(1)}%)`
          : '';
        return `- ${finding.label}${count}${confidence}`;
      })
      .join('\n');
  }, [findings]);

  const normalizedFindings = useMemo(
    () => normalizeFindings(findings as RawFinding[]),
    [findings],
  );
  const summary = useMemo(() => {
    if (normalizedFindings.length === 0) {
      return {
        foundTotal: 0,
        expectedTotal: 0,
        missing: [] as { label: string; count: number }[],
        pathologies: [] as { label: string; count: number }[],
        anatomy: [] as string[],
      };
    }
    const toothCounts = new Map<string, number>();
    const pathologyCounts = new Map<string, number>();
    const anatomyLabels = new Set<string>();
    normalizedFindings.forEach((finding) => {
      if (finding.category === 'tooth' && finding.sourceKind === 'yolo') {
        const key = finding.canonicalLabel;
        if (key) {
          toothCounts.set(key, (toothCounts.get(key) ?? 0) + 1);
        }
      }
      if (finding.category === 'pathology') {
        pathologyCounts.set(
          finding.displayLabel,
          (pathologyCounts.get(finding.displayLabel) ?? 0) + 1,
        );
      }
      if (finding.category === 'anatomy') {
        anatomyLabels.add(finding.displayLabel);
      }
    });

    const expectedTotal = Object.values(TOOTH_EXPECTED).reduce((sum, value) => sum + value, 0);
    const foundTotal = Array.from(toothCounts.values()).reduce((sum, value) => sum + value, 0);
    const missing = Object.entries(TOOTH_EXPECTED)
      .map(([key, expected]) => {
        const found = toothCounts.get(key) ?? 0;
        const missingCount = Math.max(0, expected - found);
        return missingCount > 0
          ? { label: TOOTH_LABELS[key] ?? key, count: missingCount }
          : null;
      })
      .filter((item): item is { label: string; count: number } => Boolean(item));
    const pathologies = Array.from(pathologyCounts.entries())
      .map(([label, count]) => ({ label, count }))
      .sort((a, b) => b.count - a.count);
    const anatomy = Array.from(anatomyLabels.values()).sort();

    return {
      foundTotal,
      expectedTotal,
      missing,
      pathologies,
      anatomy,
    };
  }, [normalizedFindings]);


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
    if (!imageFile) {
      setImageUrl(null);
      setImageMeta(null);
      return;
    }
    const nextUrl = URL.createObjectURL(imageFile);
    let isActive = true;
    const img = new Image();
    img.onload = () => {
      if (isActive) {
        setImageMeta({ width: img.naturalWidth, height: img.naturalHeight });
      }
    };
    img.src = nextUrl;
    setImageUrl(nextUrl);
    return () => {
      isActive = false;
      URL.revokeObjectURL(nextUrl);
    };
  }, [imageFile]);


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
      localStorage.removeItem('radiologia.auth.start');
      localStorage.removeItem('radiologia.auth.last');
      localStorage.removeItem('radiologia.auth.id');
      router.push('/');
    }
  };

  useSessionExpiry({ isActive: isAuthenticated, identity: authIdentity, onExpire: handleLogout });

  const handleFileChange = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }
    try {
      const normalized = await normalizeImageFile(file);
      setImageFile(normalized);
    } catch (error) {
      console.error(error);
      setImageFile(file);
    }
    setFindings([]);
    setAnalysisError(null);
    setChatHistory([{ role: 'system', text: 'Imagem carregada. Pronto para analisar.' }]);
  };

  const handleAnalyze = async () => {
    if (!imageFile) {
      setAnalysisError('Envie uma imagem antes de analisar.');
      return;
    }
    if (!baseApiUrl) {
      setAnalysisError('Configure NEXT_PUBLIC_MODAL_API_URL para usar o backend.');
      return;
    }
    setIsAnalyzing(true);
    setAnalysisError(null);
    try {
      const requests = [
        { modelType: 'yolo', sourceName: 'yolo' },
        { modelType: 'mask_rcnn', sourceName: 'detectron2' },
      ] as const;

      const results = await Promise.allSettled(
        requests.map(async ({ modelType, sourceName }) => {
          const formData = new FormData();
          formData.append('file', imageFile);
          formData.append('model_type', modelType);
          const response = await fetch(`${baseApiUrl}/analyze`, {
            method: 'POST',
            body: formData,
          });
          if (!response.ok) {
            const text = await response.text();
            throw new Error(text || `Falha ao analisar (${modelType}).`);
          }
          const data = await response.json();
          const rawFindings = Array.isArray(data.findings) ? data.findings : [];
          return rawFindings.map((finding: Finding) => ({
            ...finding,
            source_model: finding.source_model ?? sourceName,
            sourceModel: finding.sourceModel ?? sourceName,
            model: finding.model ?? modelType,
          }));
        }),
      );

      const merged: Finding[] = [];
      const errors: string[] = [];
      results.forEach((result, index) => {
        if (result.status === 'fulfilled') {
          merged.push(...result.value);
        } else {
          const label = requests[index].modelType;
          errors.push(`${label}: ${result.reason instanceof Error ? result.reason.message : 'erro'}`);
        }
      });

      if (merged.length === 0) {
        throw new Error(errors.join(' | ') || 'Falha ao analisar.');
      }

      setFindings(merged);
      if (errors.length) {
        setAnalysisError(`Alguns modelos falharam: ${errors.join(' | ')}`);
      }
      setChatHistory([
        { role: 'system', text: `Analise finalizada (${merged.length} achados).` },
      ]);
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Falha ao analisar.';
      setAnalysisError(message);
      setChatHistory([{ role: 'system', text: 'Erro na analise. Tente novamente.' }]);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleCopyAnalysisJson = async () => {
    if (!debugEnabled) {
      return;
    }
    if (findings.length === 0) {
      setCopyStatus('Sem analise para copiar.');
      window.setTimeout(() => setCopyStatus(null), 2000);
      return;
    }
    const payload = {
      name: imageFile?.name ?? 'upload',
      model_type: 'combined',
      image_width: imageMeta?.width ?? null,
      image_height: imageMeta?.height ?? null,
      findings,
    };
    const text = JSON.stringify(payload, null, 2);
    try {
      await navigator.clipboard.writeText(text);
      setCopyStatus('JSON copiado.');
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
      setCopyStatus(ok ? 'JSON copiado.' : 'Falha ao copiar JSON.');
    }
    window.setTimeout(() => setCopyStatus(null), 2000);
  };

  const handleSendChat = async () => {
    const text = chatInput.trim();
    if (!text) {
      return;
    }

    const nextHistory: ChatMessage[] = [...chatHistory, { role: 'user', text }];
    setChatHistory(nextHistory);
    setChatInput('');
    setIsChatSending(true);

    try {
      const payload = {
        context: chatContext || '(Sem contexto de imagem)',
        history: nextHistory
          .filter((message) => message.role !== 'system')
          .map((message) => ({ role: message.role, text: message.text })),
      };
      if (debugEnabled) {
        setLastChatExchange({ request: payload });
      }
      const response = await fetch('/api/gemini', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        let errorMessage = 'Falha ao enviar chat.';
        try {
          const data = await response.json();
          if (typeof data?.error === 'string' && data.error.trim()) {
            errorMessage = data.error;
          }
        } catch (error) {
          const textResponse = await response.text();
          if (textResponse) {
            errorMessage = textResponse;
          }
        }
        throw new Error(errorMessage);
      }

      const data = await response.json();
      if (debugEnabled) {
        setLastChatExchange({ request: payload, response: data });
      }
      setChatHistory((prev) => [...prev, { role: 'model', text: data.response }]);
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Falha no chat.';
      if (debugEnabled) {
        setLastChatExchange((prev) => (prev ? { ...prev, response: { error: message } } : null));
      }
      setChatHistory((prev) => [
        ...prev,
        { role: 'system', text: `Erro no chat: ${message}` },
      ]);
    } finally {
      setIsChatSending(false);
    }
  };

  const handleCopyChatJson = async () => {
    if (!debugEnabled) {
      return;
    }
    if (!lastChatExchange) {
      setChatCopyStatus('Nada para copiar.');
      window.setTimeout(() => setChatCopyStatus(null), 2000);
      return;
    }
    const payload = {
      generatedAt: new Date().toISOString(),
      ...lastChatExchange,
    };
    const text = JSON.stringify(payload, null, 2);
    try {
      await navigator.clipboard.writeText(text);
      setChatCopyStatus('JSON copiado.');
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
      setChatCopyStatus(ok ? 'JSON copiado.' : 'Falha ao copiar JSON.');
    }
    window.setTimeout(() => setChatCopyStatus(null), 2000);
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
      <header className="absolute left-6 right-6 top-6 z-30 pointer-events-auto flex flex-wrap items-center justify-between gap-4">
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

        <div className="glass-panel rounded-full p-1 flex items-center gap-1">
          <button
            type="button"
            onClick={() => setActiveTab('chat')}
            className={tabClassName(activeTab === 'chat')}
            aria-pressed={activeTab === 'chat'}
          >
            Consulta rapida
          </button>
          <button
            type="button"
            onClick={() => setActiveTab('rx')}
            className={tabClassName(activeTab === 'rx')}
            aria-pressed={activeTab === 'rx'}
          >
            Analise de RX
          </button>
        </div>

        <div className="flex flex-wrap items-center gap-3">
          <button
            onClick={toggleTheme}
            className="relative pointer-events-auto p-2.5 rounded-full glass-panel shadow-lg hover:scale-105 transition-transform text-brand-blue dark:text-brand-yellow"
            title="Alternar tema"
          >
            {isDarkMode ? <Sun className="h-5 w-5" /> : <Moon className="h-5 w-5" />}
          </button>

          <Popover>
            <PopoverTrigger asChild>
              <button
                className="relative pointer-events-auto flex items-center gap-2 rounded-full glass-panel border border-brand-blue/20 dark:border-white/15 px-3 py-2 text-xs uppercase tracking-wider font-semibold text-brand-blue/80 dark:text-white/80"
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
          className="w-[72vw] max-w-5xl opacity-90"
        />
      </div>

      <main className="relative z-10 min-h-screen pt-28 px-4 pb-6">
        <div className="w-full flex min-h-[calc(100vh-8.5rem)] flex-col">
          {activeTab === 'chat' ? (
            <section className="flex-1 border border-white/20 dark:border-white/10 rounded-3xl bg-white/45 dark:bg-white/5 backdrop-blur-xl p-6 shadow-xl">
              <div className="flex flex-wrap items-start justify-between gap-3">
                <div>
                  <h1 className="text-2xl font-semibold text-brand-blue dark:text-white">
                    Auxiliar Clinico
                  </h1>
                  <p className="mt-1 text-sm text-slate-600 dark:text-slate-300">
                    Faca perguntas rapidas sobre protocolos, condutas e casos.
                  </p>
                </div>
                <div className="flex items-center gap-2">
                  {debugEnabled ? (
                    <>
                      <button
                        type="button"
                        onClick={handleCopyChatJson}
                        className="rounded-full border border-brand-blue/40 px-4 py-2 text-xs font-semibold text-brand-blue shadow-md transition-all hover:bg-brand-blue/10 dark:border-white/20 dark:text-white/80 dark:hover:bg-white/10"
                      >
                        Copiar JSON
                      </button>
                      {chatCopyStatus ? (
                        <span className="text-[10px] uppercase tracking-wider text-brand-green">
                          {chatCopyStatus}
                        </span>
                      ) : null}
                    </>
                  ) : null}
                </div>
              </div>

              <div className="mt-6 flex h-full flex-col gap-4">
                <div className="flex-1 border border-white/30 dark:border-white/10 bg-white/30 dark:bg-white/5 rounded-2xl p-5 text-sm text-slate-600 dark:text-slate-300 overflow-y-auto">
                  <div className="space-y-3">
                    {chatHistory.map((message, index) => (
                      <div key={`${message.role}-${index}`}>
                        {message.role === 'system' ? (
                          <p className="text-center text-xs uppercase tracking-wider text-slate-500 dark:text-slate-400">
                            {message.text}
                          </p>
                        ) : (
                          <div className={message.role === 'user' ? 'text-right' : 'text-left'}>
                            <p className="text-[11px] tracking-wider text-slate-500 dark:text-slate-400">
                              {message.role === 'user' ? 'Voce' : 'radiologIA'}
                            </p>
                            <p className="mt-1 text-sm text-slate-700 dark:text-slate-200">
                              {renderChatText(message.text)}
                            </p>
                          </div>
                        )}
                      </div>
                    ))}
                    {isChatSending ? renderTypingIndicator() : null}
                  </div>
                </div>
                <div className="flex flex-col gap-3 sm:flex-row">
                  <textarea
                    value={chatInput}
                    onChange={(event) => setChatInput(event.target.value)}
                    onKeyDown={(event) => {
                      if (event.key === 'Enter' && !event.shiftKey) {
                        event.preventDefault();
                        handleSendChat();
                      }
                    }}
                    className="flex-1 resize-none rounded-2xl border border-white/30 dark:border-white/10 bg-white/60 dark:bg-slate-900/60 p-4 text-sm text-slate-700 dark:text-white focus:outline-none focus:ring-2 focus:ring-brand-green/40"
                    rows={3}
                    placeholder="Ex: Em um caso de reabsorcao externa, qual conduta sugerir?"
                  />
                  <button
                    onClick={handleSendChat}
                    disabled={isChatSending}
                    className="rounded-2xl bg-brand-green px-6 py-3 text-sm font-semibold text-white shadow-sm hover:shadow-md transition-all disabled:opacity-60 disabled:cursor-not-allowed"
                  >
                    Enviar
                  </button>
                </div>
              </div>
            </section>
          ) : (
            <section className="flex-1 border border-white/20 dark:border-white/10 rounded-3xl bg-white/45 dark:bg-white/5 backdrop-blur-xl shadow-xl overflow-hidden flex flex-col">
              <div className="flex-1 p-6">
                <div className="mb-6 grid gap-4 lg:grid-cols-[1.4fr_1fr]">
                  <div className="rounded-2xl border border-white/30 bg-white/70 p-4 text-sm text-slate-600 dark:border-white/10 dark:bg-slate-900/40 dark:text-slate-200">
                    <p className="text-xs font-semibold uppercase tracking-wider text-brand-blue dark:text-white">
                      Como interagir
                    </p>
                    <p className="mt-2 text-sm text-slate-600 dark:text-slate-300">
                      Clique diretamente em um dente para aproximar e ver detalhes do achado selecionado.
                    </p>
                  </div>
                  <div className="rounded-2xl border border-white/30 bg-white/70 p-4 text-sm text-slate-600 dark:border-white/10 dark:bg-slate-900/40 dark:text-slate-200">
                    <p className="text-xs font-semibold uppercase tracking-wider text-brand-blue dark:text-white">
                      Resumo do exame
                    </p>
                    {normalizedFindings.length === 0 ? (
                      <p className="mt-2 text-sm text-slate-600 dark:text-slate-300">
                        Envie e analise uma radiografia para gerar o resumo automatico.
                      </p>
                    ) : (
                      <>
                        <p className="mt-2 text-sm text-slate-600 dark:text-slate-300">
                          Radiografia panoramica com {summary.foundTotal} de {summary.expectedTotal} dentes estimados.
                        </p>
                        <p className="mt-2 text-xs text-slate-500 dark:text-slate-400">
                          {summary.missing.length
                            ? `Ausencias sugeridas: ${formatCountList(summary.missing)}.`
                            : 'Sem ausencias sugeridas pelo modelo.'}
                        </p>
                        <p className="mt-2 text-xs text-slate-500 dark:text-slate-400">
                          {summary.pathologies.length
                            ? `Patologias detectadas: ${formatCountList(summary.pathologies)}.`
                            : 'Nenhuma patologia detectada no exame atual.'}
                        </p>
                        <p className="mt-2 text-xs text-slate-500 dark:text-slate-400">
                          {summary.anatomy.length
                            ? `Estruturas segmentadas: ${summary.anatomy.join(', ')}.`
                            : 'Sem estruturas segmentadas neste caso.'}
                        </p>
                      </>
                    )}
                  </div>
                </div>
                <FindingsViewer
                  imageUrl={imageUrl}
                  findings={findings}
                  isLoading={isAnalyzing}
                  loadingLabel="Analisando modelos..."
                  title="Analise integrada do RX"
                  subtitle="Overlay unico com filtros e lista separada por patologia, tratamento e dentes."
                  enableToothFusionPreview
                  enableClickSelect
                  showList={false}
                  showTeethToggle={false}
                  structureLabel="Dentes"
                  toolbar={
                    <>
                      <label className="rounded-full bg-brand-blue px-4 py-2 text-xs font-semibold text-white shadow-md cursor-pointer">
                        Enviar imagem
                        <input type="file" accept="image/*" onChange={handleFileChange} className="hidden" />
                      </label>
                      <button
                        type="button"
                        onClick={handleAnalyze}
                        disabled={!imageFile || isAnalyzing}
                        className="rounded-full bg-brand-green px-4 py-2 text-xs font-semibold text-white shadow-md transition-all disabled:opacity-60 disabled:cursor-not-allowed"
                      >
                        {isAnalyzing ? 'Analisando...' : 'Analisar'}
                      </button>
                      {debugEnabled ? (
                        <>
                          <button
                            type="button"
                            onClick={handleCopyAnalysisJson}
                            className="rounded-full border border-brand-blue/40 px-4 py-2 text-xs font-semibold text-brand-blue shadow-md transition-all hover:bg-brand-blue/10 dark:border-white/20 dark:text-white/80 dark:hover:bg-white/10"
                          >
                            Copiar JSON
                          </button>
                          {copyStatus ? (
                            <span className="text-[10px] uppercase tracking-wider text-brand-green">
                              {copyStatus}
                            </span>
                          ) : null}
                        </>
                      ) : null}
                    </>
                  }
                />
                {analysisError ? (
                  <div className="mt-4 rounded-2xl border border-white/20 dark:border-white/10 bg-white/60 px-4 py-3 text-xs text-red-600 dark:bg-white/5 dark:text-red-300">
                    {analysisError}
                  </div>
                ) : null}
              </div>
              <div className="border-t border-white/20 dark:border-white/10 bg-white/60 dark:bg-slate-900/50 backdrop-blur-xl p-6">
                <div className="flex flex-wrap items-center justify-between gap-4">
                  <div>
                    <h2 className="text-sm font-semibold uppercase tracking-wider text-brand-blue dark:text-white">
                      Chat clinico
                    </h2>
                    <p className="mt-1 text-xs text-slate-600 dark:text-slate-300">
                      Conversa contextual com achados do RX.
                    </p>
                  </div>
                  <div className="flex items-center gap-2">
                    {debugEnabled ? (
                      <>
                        <button
                          type="button"
                          onClick={handleCopyChatJson}
                          className="rounded-full border border-brand-blue/40 px-4 py-2 text-xs font-semibold text-brand-blue shadow-md transition-all hover:bg-brand-blue/10 dark:border-white/20 dark:text-white/80 dark:hover:bg-white/10"
                        >
                          Copiar JSON
                        </button>
                        {chatCopyStatus ? (
                          <span className="text-[10px] uppercase tracking-wider text-brand-green">
                            {chatCopyStatus}
                          </span>
                        ) : null}
                      </>
                    ) : null}
                    <button
                      type="button"
                      onClick={() => setIsChatExpanded((prev) => !prev)}
                      className="rounded-full border border-brand-blue/30 dark:border-white/20 px-4 py-2 text-xs font-semibold uppercase tracking-wider text-brand-blue dark:text-white hover:bg-white/40 dark:hover:bg-white/10 transition-all"
                    >
                      {isChatExpanded ? 'Recolher chat' : 'Expandir chat'}
                    </button>
                  </div>
                </div>
                <div
                  className={[
                    'mt-4 flex flex-col gap-4 rounded-2xl border border-white/30 dark:border-white/10 bg-white/30 dark:bg-white/5 p-5',
                    isChatExpanded ? 'min-h-[32vh]' : 'min-h-[20vh]',
                  ].join(' ')}
                >
                  <div className="flex-1 space-y-3 overflow-y-auto text-sm text-slate-600 dark:text-slate-300">
                    {chatHistory.map((message, index) => (
                      <div key={`${message.role}-${index}`}>
                        {message.role === 'system' ? (
                          <p className="text-center text-xs uppercase tracking-wider text-slate-500 dark:text-slate-400">
                            {message.text}
                          </p>
                        ) : (
                          <div className={message.role === 'user' ? 'text-right' : 'text-left'}>
                            <p className="text-[11px] tracking-wider text-slate-500 dark:text-slate-400">
                              {message.role === 'user' ? 'Voce' : 'radiologIA'}
                            </p>
                            <p className="mt-1 text-sm text-slate-700 dark:text-slate-200">
                              {renderChatText(message.text)}
                            </p>
                          </div>
                        )}
                      </div>
                    ))}
                    {isChatSending ? renderTypingIndicator() : null}
                  </div>
                  <div className="flex flex-col gap-3 sm:flex-row">
                    <textarea
                      value={chatInput}
                      onChange={(event) => setChatInput(event.target.value)}
                      onKeyDown={(event) => {
                        if (event.key === 'Enter' && !event.shiftKey) {
                          event.preventDefault();
                          handleSendChat();
                        }
                      }}
                      className="flex-1 resize-none rounded-2xl border border-white/30 dark:border-white/10 bg-white/60 dark:bg-slate-900/60 p-4 text-sm text-slate-700 dark:text-white focus:outline-none focus:ring-2 focus:ring-brand-green/40"
                      rows={2}
                      placeholder="Pergunte sobre o caso ou sobre um achado especifico."
                    />
                    <button
                      onClick={handleSendChat}
                      disabled={isChatSending}
                      className="rounded-2xl bg-brand-green px-6 py-3 text-sm font-semibold text-white shadow-sm hover:shadow-md transition-all disabled:opacity-60 disabled:cursor-not-allowed"
                    >
                      {isChatSending ? 'Enviando...' : 'Enviar'}
                    </button>
                  </div>
                </div>
              </div>
            </section>
          )}
        </div>
      </main>
    </div>
  );
}
