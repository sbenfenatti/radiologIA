'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import { ArrowLeft, LogOut, Moon, Sun, UserRound } from 'lucide-react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { useSupabaseUser } from '@/hooks/use-supabase-user';
import { supabase } from '@/lib/supabase/client';
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover';

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
};

type GroupedFinding = {
  label: string;
  count: number;
  confidenceAvg?: number;
  confidenceMax?: number;
  model?: string;
  isAnatomy: boolean;
};

const ANATOMY_KEYWORDS = [
  'dentina',
  'esmalte',
  'polpa',
  'enamel',
  'dentin',
  'dentine',
  'pulp',
];

const isAnatomyLabel = (label: string) => {
  const normalized = label.toLowerCase();
  return ANATOMY_KEYWORDS.some((keyword) => normalized.includes(keyword));
};

export default function AuxiliarClinicoPage() {
  const router = useRouter();
  const { user, isLoading } = useSupabaseUser();
  const [tokenChecked, setTokenChecked] = useState(false);
  const [hasTokenAccess, setHasTokenAccess] = useState(false);
  const [isDarkMode, setIsDarkMode] = useState(false);
  const [activeTab, setActiveTab] = useState<'chat' | 'rx'>('rx');
  const [isChatExpanded, setIsChatExpanded] = useState(false);
  const [hideAnatomy, setHideAnatomy] = useState(true);
  const [groupByLabel, setGroupByLabel] = useState(true);
  const [modelType, setModelType] = useState<'yolo' | 'mask_rcnn'>('mask_rcnn');
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imageReady, setImageReady] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisError, setAnalysisError] = useState<string | null>(null);
  const [findings, setFindings] = useState<Finding[]>([]);
  const [chatHistory, setChatHistory] = useState<ChatMessage[]>([
    { role: 'system', text: 'Aguardando contexto da imagem.' },
  ]);
  const [chatInput, setChatInput] = useState('');
  const [isChatSending, setIsChatSending] = useState(false);
  const isVisitor = hasTokenAccess && !user;
  const displayName = user?.user_metadata?.full_name || user?.email?.split('@')[0] || 'Usuário';
  const profileHref = isVisitor ? '/perfil/visitante' : '/perfil';
  const profileLabel = isVisitor ? 'Perfil (Visitante)' : 'Perfil';
  const baseApiUrl = process.env.NEXT_PUBLIC_MODAL_API_URL ?? '';
  const imageRef = useRef<HTMLImageElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const tabClassName = (active: boolean) =>
    [
      'px-4 py-2 text-sm font-semibold uppercase tracking-wider rounded-full transition-all',
      active
        ? 'bg-white/70 text-brand-blue shadow-sm dark:bg-white/10 dark:text-white'
        : 'text-brand-blue/70 hover:text-brand-blue dark:text-white/60 dark:hover:text-white',
    ].join(' ');

  const filterButtonClass = (active: boolean) =>
    [
      'rounded-full border px-3 py-1 text-[11px] font-semibold uppercase tracking-wider transition-all',
      active
        ? 'bg-brand-green text-white border-brand-green/40 shadow-sm'
        : 'border-brand-blue/30 text-brand-blue hover:bg-white/40 dark:border-white/20 dark:text-white/70 dark:hover:text-white dark:hover:bg-white/10',
    ].join(' ');

  const visibleFindings = useMemo(() => {
    if (!hideAnatomy) {
      return findings;
    }
    return findings.filter((finding) => !isAnatomyLabel(finding.label));
  }, [findings, hideAnatomy]);

  const groupedFindings = useMemo(() => {
    if (!groupByLabel) {
      return visibleFindings
        .map<GroupedFinding>((finding) => ({
          label: finding.label,
          count: 1,
          confidenceAvg: finding.confidence,
          confidenceMax: finding.confidence,
          model: finding.model,
          isAnatomy: isAnatomyLabel(finding.label),
        }))
        .sort((a, b) => (b.confidenceMax ?? 0) - (a.confidenceMax ?? 0));
    }

    const groups = new Map<string, GroupedFinding & { confidenceTotal: number; confidenceCount: number }>();

    visibleFindings.forEach((finding) => {
      const key = finding.label.trim().toLowerCase();
      const entry = groups.get(key) ?? {
        label: finding.label,
        count: 0,
        confidenceAvg: undefined,
        confidenceMax: undefined,
        model: finding.model,
        isAnatomy: isAnatomyLabel(finding.label),
        confidenceTotal: 0,
        confidenceCount: 0,
      };

      entry.count += 1;
      if (finding.confidence != null) {
        entry.confidenceTotal += finding.confidence;
        entry.confidenceCount += 1;
        entry.confidenceMax = entry.confidenceMax == null
          ? finding.confidence
          : Math.max(entry.confidenceMax, finding.confidence);
      }
      if (!entry.model && finding.model) {
        entry.model = finding.model;
      }

      groups.set(key, entry);
    });

    return Array.from(groups.values())
      .map(({ confidenceTotal, confidenceCount, ...rest }) => ({
        ...rest,
        confidenceAvg: confidenceCount ? confidenceTotal / confidenceCount : undefined,
      }))
      .sort((a, b) => {
        const confidenceDelta = (b.confidenceMax ?? 0) - (a.confidenceMax ?? 0);
        if (confidenceDelta !== 0) {
          return confidenceDelta;
        }
        if (b.count !== a.count) {
          return b.count - a.count;
        }
        return a.label.localeCompare(b.label);
      });
  }, [groupByLabel, visibleFindings]);

  const filteredOutCount = findings.length - visibleFindings.length;

  const chatContext = useMemo(() => {
    if (groupedFindings.length === 0) {
      return '';
    }
    return groupedFindings
      .map((finding) => {
        const count = finding.count > 1 ? ` (${finding.count} regioes)` : '';
        const confidence = finding.confidenceAvg != null
          ? ` (Confianca media: ${(finding.confidenceAvg * 100).toFixed(1)}%)`
          : '';
        return `- ${finding.label}${count}${confidence}`;
      })
      .join('\n');
  }, [groupedFindings]);


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
      return;
    }
    const nextUrl = URL.createObjectURL(imageFile);
    setImageUrl(nextUrl);
    return () => {
      URL.revokeObjectURL(nextUrl);
    };
  }, [imageFile]);

  useEffect(() => {
    const drawFindings = () => {
      const canvas = canvasRef.current;
      const img = imageRef.current;
      if (!imageReady || !canvas || !img || !img.complete || img.naturalWidth === 0) {
        return;
      }

      const displayWidth = img.clientWidth;
      const displayHeight = img.clientHeight;
      canvas.width = displayWidth;
      canvas.height = displayHeight;

      const ctx = canvas.getContext('2d');
      if (!ctx) {
        return;
      }
      ctx.clearRect(0, 0, displayWidth, displayHeight);

      const scaleX = displayWidth / img.naturalWidth;
      const scaleY = displayHeight / img.naturalHeight;

      visibleFindings.forEach((finding) => {
        const isCaries = finding.label.toLowerCase().includes('carie');
        ctx.strokeStyle = isCaries ? '#f87171' : '#4ade80';
        ctx.fillStyle = isCaries ? 'rgba(248, 113, 113, 0.2)' : 'rgba(74, 222, 128, 0.2)';
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
      });
    };

    drawFindings();
    const handleResize = () => drawFindings();
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [visibleFindings, imageReady, imageUrl]);

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
      router.push('/');
    }
  };

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }
    setImageFile(file);
    setImageReady(false);
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
      const formData = new FormData();
      formData.append('file', imageFile);
      formData.append('model_type', modelType);

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
      setFindings(nextFindings);
      setChatHistory([
        { role: 'system', text: `Analise finalizada (${nextFindings.length} achados).` },
      ]);
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Falha ao analisar.';
      setAnalysisError(message);
      setChatHistory([{ role: 'system', text: 'Erro na analise. Tente novamente.' }]);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleSendChat = async () => {
    const text = chatInput.trim();
    if (!text) {
      return;
    }
    if (!baseApiUrl) {
      setChatHistory((prev) => [
        ...prev,
        { role: 'system', text: 'Configure NEXT_PUBLIC_MODAL_API_URL para usar o backend.' },
      ]);
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

      const response = await fetch(`${baseApiUrl}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        const textResponse = await response.text();
        throw new Error(textResponse || 'Falha ao enviar chat.');
      }

      const data = await response.json();
      setChatHistory((prev) => [...prev, { role: 'model', text: data.response }]);
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Falha no chat.';
      setChatHistory((prev) => [
        ...prev,
        { role: 'system', text: `Erro no chat: ${message}` },
      ]);
    } finally {
      setIsChatSending(false);
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
          className="w-[72vw] max-w-5xl opacity-90"
        />
      </div>

      <main className="relative z-10 min-h-screen pt-28 px-4 pb-6">
        <div className="w-full flex min-h-[calc(100vh-8.5rem)] flex-col">
          {activeTab === 'chat' ? (
            <section className="flex-1 border border-white/20 dark:border-white/10 rounded-3xl bg-white/45 dark:bg-white/5 backdrop-blur-xl p-6 shadow-xl">
              <div>
                <h1 className="text-2xl font-semibold text-brand-blue dark:text-white">
                  Auxiliar Clinico
                </h1>
                <p className="mt-1 text-sm text-slate-600 dark:text-slate-300">
                  Faca perguntas rapidas sobre protocolos, condutas e casos.
                </p>
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
                            <p className="text-[11px] uppercase tracking-wider text-slate-500 dark:text-slate-400">
                              {message.role === 'user' ? 'Voce' : 'RadiologIA'}
                            </p>
                            <p className="mt-1 text-sm text-slate-700 dark:text-slate-200 whitespace-pre-line">
                              {message.text}
                            </p>
                          </div>
                        )}
                      </div>
                    ))}
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
                <div className="relative h-full rounded-2xl border border-dashed border-white/40 dark:border-white/15 flex flex-col overflow-hidden">
                  <div className="absolute left-6 top-6 z-10 flex flex-wrap items-center gap-2">
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
                  </div>

                  <div className="absolute right-6 top-6 z-10 flex flex-wrap items-center gap-3">
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
                  </div>

                  <div className="flex-1 flex items-center justify-center px-6 py-10 text-center">
                    {imageUrl ? (
                      <div className="relative h-full w-full flex items-center justify-center">
                        <img
                          ref={imageRef}
                          src={imageUrl}
                          alt="Radiografia"
                          onLoad={() => {
                            const img = imageRef.current;
                            if (img && canvasRef.current) {
                              canvasRef.current.width = img.clientWidth;
                              canvasRef.current.height = img.clientHeight;
                            }
                            setImageReady(true);
                          }}
                          className="max-h-full max-w-full object-contain"
                        />
                        <canvas ref={canvasRef} className="absolute inset-0 h-full w-full pointer-events-none" />
                      </div>
                    ) : (
                      <div>
                        <p className="text-sm text-slate-600 dark:text-slate-300">
                          Arraste a radiografia panoramica ou clique para enviar.
                        </p>
                        <p className="mt-2 text-xs text-slate-500 dark:text-slate-400">
                          Suporta JPG ou PNG. Use a panoramica para melhores resultados.
                        </p>
                      </div>
                    )}
                  </div>
                  {analysisError ? (
                    <div className="border-t border-white/20 dark:border-white/10 px-6 py-3 text-xs text-red-600 dark:text-red-300">
                      {analysisError}
                    </div>
                  ) : null}
                </div>
              </div>
              <div className="border-t border-white/20 dark:border-white/10 bg-white/60 dark:bg-slate-900/50 backdrop-blur-xl">
                <div className="grid gap-0 lg:grid-cols-[1fr_1.6fr] divide-y lg:divide-y-0 lg:divide-x divide-white/20 dark:divide-white/10">
                  <div className="p-6">
                    <h2 className="text-sm font-semibold uppercase tracking-wider text-brand-blue dark:text-white">
                      Achados
                    </h2>
                    <p className="mt-1 text-xs text-slate-600 dark:text-slate-300">
                      Principais regioes e suspeitas detectadas.
                    </p>
                    <div className="mt-3 flex flex-wrap items-center justify-between gap-3 text-xs text-slate-600 dark:text-slate-300">
                      <div>
                        <span className="font-semibold text-brand-blue dark:text-white">
                          {groupByLabel ? `${groupedFindings.length} grupos` : `${groupedFindings.length} achados`}
                        </span>
                        {findings.length > 0 ? (
                          <span className="text-slate-500 dark:text-slate-400">
                            {` de ${findings.length} totais`}
                            {hideAnatomy && filteredOutCount > 0
                              ? ` · ${filteredOutCount} estruturas ocultas`
                              : ''}
                          </span>
                        ) : null}
                      </div>
                      <div className="flex flex-wrap items-center gap-2">
                        <button
                          type="button"
                          onClick={() => setGroupByLabel((prev) => !prev)}
                          aria-pressed={groupByLabel}
                          className={filterButtonClass(groupByLabel)}
                        >
                          Agrupar rotulos
                        </button>
                        <button
                          type="button"
                          onClick={() => setHideAnatomy((prev) => !prev)}
                          aria-pressed={hideAnatomy}
                          className={filterButtonClass(hideAnatomy)}
                        >
                          Ocultar estruturas
                        </button>
                      </div>
                    </div>
                    {hideAnatomy && filteredOutCount > 0 ? (
                      <p className="mt-2 text-[11px] uppercase tracking-wider text-slate-500 dark:text-slate-400">
                        Filtrando tecidos: dentina, esmalte, polpa.
                      </p>
                    ) : null}
                    <ul className="mt-4 space-y-3 text-sm text-slate-700 dark:text-slate-200">
                      {groupedFindings.length === 0 ? (
                        <li className="rounded-2xl border border-white/30 dark:border-white/10 bg-white/30 dark:bg-white/5 p-4">
                          <p className="font-semibold text-brand-blue dark:text-white">
                            {findings.length === 0
                              ? 'Sem achados ainda'
                              : hideAnatomy
                                ? 'Somente estruturas detectadas'
                                : 'Sem achados visiveis'}
                          </p>
                          <p className="mt-1 text-xs text-slate-600 dark:text-slate-300">
                            {findings.length === 0
                              ? 'Envie uma imagem para gerar resultados.'
                              : hideAnatomy
                                ? 'Desative "Ocultar estruturas" para listar dentina, esmalte e polpa.'
                                : 'Tente outro modelo ou uma nova imagem.'}
                          </p>
                        </li>
                      ) : (
                        groupedFindings.map((finding, index) => {
                          const confidenceValue = finding.confidenceMax ?? finding.confidenceAvg;
                          return (
                          <li
                            key={`${finding.label}-${index}`}
                            className="rounded-2xl border border-white/30 dark:border-white/10 bg-white/30 dark:bg-white/5 p-4"
                          >
                            <div className="flex items-center justify-between gap-3">
                              <div>
                                <p className="font-semibold text-brand-blue dark:text-white">
                                  {finding.label}
                                </p>
                                <div className="mt-1 flex flex-wrap items-center gap-2 text-[11px] uppercase tracking-wider text-slate-500 dark:text-slate-400">
                                  {finding.count > 1 ? (
                                    <span>{finding.count} regioes</span>
                                  ) : null}
                                  {finding.isAnatomy ? <span>estrutura</span> : null}
                                </div>
                              </div>
                              {confidenceValue != null ? (
                                <span className="text-xs text-slate-500 dark:text-slate-400">
                                  {(confidenceValue * 100).toFixed(1)}%
                                </span>
                              ) : null}
                            </div>
                            {!groupByLabel && finding.model ? (
                              <p className="mt-1 text-[11px] uppercase tracking-wider text-slate-500 dark:text-slate-400">
                                {finding.model}
                              </p>
                            ) : null}
                          </li>
                          );
                        })
                      )}
                    </ul>
                  </div>
                  <div className="p-6">
                    <div className="flex items-center justify-between gap-4">
                      <div>
                        <h2 className="text-sm font-semibold uppercase tracking-wider text-brand-blue dark:text-white">
                          Chat clinico
                        </h2>
                        <p className="mt-1 text-xs text-slate-600 dark:text-slate-300">
                          Conversa contextual com achados do RX.
                        </p>
                      </div>
                      <button
                        type="button"
                        onClick={() => setIsChatExpanded((prev) => !prev)}
                        className="rounded-full border border-brand-blue/30 dark:border-white/20 px-4 py-2 text-xs font-semibold uppercase tracking-wider text-brand-blue dark:text-white hover:bg-white/40 dark:hover:bg-white/10 transition-all"
                      >
                        {isChatExpanded ? 'Recolher chat' : 'Expandir chat'}
                      </button>
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
                                <p className="text-[11px] uppercase tracking-wider text-slate-500 dark:text-slate-400">
                                  {message.role === 'user' ? 'Voce' : 'RadiologIA'}
                                </p>
                                <p className="mt-1 text-sm text-slate-700 dark:text-slate-200 whitespace-pre-line">
                                  {message.text}
                                </p>
                              </div>
                            )}
                          </div>
                        ))}
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
                </div>
              </div>
            </section>
          )}
        </div>
      </main>
    </div>
  );
}
