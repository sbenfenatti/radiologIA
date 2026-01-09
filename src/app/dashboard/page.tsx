'use client';

import { useEffect, useRef, useState } from 'react';
import { motion } from 'framer-motion';
import { ScanLine, Brain, Moon, Sun, UserRound, LogOut } from 'lucide-react';
import { useRouter } from 'next/navigation';
import { useSupabaseUser } from '@/hooks/use-supabase-user';
import { supabase } from '@/lib/supabase/client';
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover';

const FeatureCard = ({
    title,
    description,
    infoId,
    Icon,
    accent,
    accentSoft,
    accentBorder,
    decorations,
}: {
    title: string;
    description: string;
    infoId: string;
    Icon: (props: { className?: string }) => JSX.Element;
    accent: string;
    accentSoft: string;
    accentBorder: string;
    decorations?: React.ReactNode;
}) => {
    const handleScroll = () => {
        const target = document.getElementById(infoId);
        if (target) {
            target.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
    };

    return (
        <div
            role="button"
            tabIndex={0}
            onClick={handleScroll}
            onKeyDown={(event) => {
                if (event.key === 'Enter' || event.key === ' ') {
                    event.preventDefault();
                    handleScroll();
                }
            }}
            className="group relative h-full w-full cursor-pointer overflow-hidden rounded-[34px] border border-white/60 bg-white/70 dark:border-white/10 dark:bg-slate-900/60 backdrop-blur-2xl shadow-[0_24px_60px_rgba(15,23,42,0.15)] transition-transform duration-300 ease-out hover:-translate-y-2 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand-blue/40"
        >
            <div className="absolute inset-0 pointer-events-none">
                {decorations}
            </div>
            <div className={`absolute -right-16 -top-16 h-44 w-44 rounded-full ${accentSoft} blur-3xl opacity-70`}></div>
            <div className={`absolute right-8 top-6 h-28 w-28 ${accent} opacity-10`}>
                <Icon className="h-full w-full" />
            </div>
            <div className="absolute -left-6 bottom-6 h-24 w-24 rounded-3xl border border-white/30 bg-white/40 dark:bg-white/5 backdrop-blur-md rotate-12"></div>
            <div className="relative z-10 flex h-full flex-col justify-between p-8">
                <div className="flex items-start justify-between gap-6">
                    <div>
                        <p className="text-xs uppercase tracking-[0.3em] text-brand-blue/60 dark:text-white/50">
                            Funcionalidade
                        </p>
                        <h2 className="mt-3 text-3xl font-semibold text-brand-blue dark:text-white">{title}</h2>
                        <p className="mt-3 text-sm text-slate-600 dark:text-slate-300">{description}</p>
                    </div>
                    <div className={`flex h-14 w-14 items-center justify-center rounded-2xl border ${accentBorder} bg-white/60 dark:bg-white/5`}>
                        <Icon className={`h-7 w-7 ${accent}`} />
                    </div>
                </div>
                <div className="mt-6 flex items-center justify-between text-xs">
                    <span className="text-brand-blue/60 dark:text-white/60">Clique no card para acessar</span>
                    <button
                        type="button"
                        onClick={(event) => {
                            event.stopPropagation();
                            handleScroll();
                        }}
                        className="rounded-full border border-brand-blue/20 px-4 py-1.5 text-[11px] font-semibold uppercase tracking-widest text-brand-blue/70 dark:border-white/15 dark:text-white/70 transition-colors hover:text-brand-blue dark:hover:text-white"
                    >
                        Saiba mais
                    </button>
                </div>
            </div>
        </div>
    );
};

export default function PresentationPage() {
    const router = useRouter();
    const { user, isLoading } = useSupabaseUser();
    const [tokenChecked, setTokenChecked] = useState(false);
    const [hasTokenAccess, setHasTokenAccess] = useState(false);
    const [isDarkMode, setIsDarkMode] = useState(false);
    const isVisitor = hasTokenAccess && !user;
    const displayName = user?.user_metadata?.full_name || user?.email?.split('@')[0] || 'Usuário';
    const profileHref = isVisitor ? '/perfil/visitante' : '/perfil';
    const profileLabel = isVisitor ? 'Perfil (Visitante)' : 'Perfil';
    const scrollRef = useRef<HTMLDivElement | null>(null);
    const sectionRefs = useRef<HTMLElement[]>([]);
    const scrollLockRef = useRef(false);
    const currentIndexRef = useRef(0);
    const touchStartYRef = useRef<number | null>(null);

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

    const clampIndex = (value: number) =>
        Math.min(Math.max(value, 0), sectionRefs.current.length - 1);

    const scrollToIndex = (index: number) => {
        const container = scrollRef.current;
        const section = sectionRefs.current[index];
        if (!container || !section) {
            return;
        }
        scrollLockRef.current = true;
        currentIndexRef.current = index;
        container.scrollTo({ top: section.offsetTop, behavior: 'smooth' });
        window.setTimeout(() => {
            scrollLockRef.current = false;
        }, 700);
    };

    const handleWheel = (event: React.WheelEvent<HTMLDivElement>) => {
        if (scrollLockRef.current || sectionRefs.current.length === 0) {
            event.preventDefault();
            return;
        }
        const direction = event.deltaY > 0 ? 1 : -1;
        const nextIndex = clampIndex(currentIndexRef.current + direction);
        if (nextIndex === currentIndexRef.current) {
            return;
        }
        event.preventDefault();
        scrollToIndex(nextIndex);
    };

    const handleScrollSync = () => {
        if (scrollLockRef.current) {
            return;
        }
        const container = scrollRef.current;
        if (!container) {
            return;
        }
        const scrollTop = container.scrollTop;
        let closestIndex = 0;
        let closestDistance = Number.POSITIVE_INFINITY;
        sectionRefs.current.forEach((section, index) => {
            const distance = Math.abs(section.offsetTop - scrollTop);
            if (distance < closestDistance) {
                closestDistance = distance;
                closestIndex = index;
            }
        });
        currentIndexRef.current = closestIndex;
    };

    const handleTouchStart = (event: React.TouchEvent<HTMLDivElement>) => {
        touchStartYRef.current = event.touches[0]?.clientY ?? null;
    };

    const handleTouchEnd = (event: React.TouchEvent<HTMLDivElement>) => {
        if (touchStartYRef.current === null) {
            return;
        }
        const endY = event.changedTouches[0]?.clientY ?? touchStartYRef.current;
        const deltaY = touchStartYRef.current - endY;
        touchStartYRef.current = null;
        if (Math.abs(deltaY) < 50) {
            return;
        }
        const direction = deltaY > 0 ? 1 : -1;
        scrollToIndex(clampIndex(currentIndexRef.current + direction));
    };

    if (isLoading || !tokenChecked) {
        return (
            <div className="flex h-screen w-full items-center justify-center">
                <div className="w-8 h-8 border-4 border-brand-green border-t-transparent rounded-full animate-spin"></div>
            </div>
        );
    }

    return (
        <div className="min-h-screen antialiased relative overflow-hidden transition-colors duration-500 text-gray-900 dark:text-gray-100 bg-background">
            <div className="absolute inset-0 w-full h-full overflow-hidden z-0 pointer-events-none flag-stage">
                <div className="absolute inset-0 bg-[url('https://grainy-gradients.vercel.app/noise.svg')] opacity-20 brightness-100 contrast-150 mix-blend-overlay"></div>
                <div className="flag-canvas">
                    <div className="flag-shape flag-green"></div>
                    <div className="flag-shape flag-yellow"></div>
                    <div className="flag-shape flag-blue"></div>
                    <div className="flag-glow"></div>
                </div>
                <div className="flag-glass"></div>
            </div>

            <header className="absolute right-6 top-6 z-20 flex flex-wrap items-center gap-3">
                <button
                    onClick={toggleTheme}
                    className="p-2.5 rounded-full glass-panel shadow-lg hover:scale-105 transition-transform text-brand-blue dark:text-brand-yellow"
                    title="Alternar tema"
                >
                    {isDarkMode ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
                </button>

                <Popover>
                    <PopoverTrigger asChild>
                        <button
                            className="flex items-center gap-2 rounded-full glass-panel border border-brand-blue/20 dark:border-white/15 px-3 py-2 text-xs uppercase tracking-wider font-semibold text-brand-blue/80 dark:text-white/80"
                            title="Perfil"
                        >
                            <span className="flex h-8 w-8 items-center justify-center rounded-full bg-brand-blue/10 text-brand-blue dark:bg-white/10 dark:text-white">
                                <UserRound className="w-4 h-4" />
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
                        <div className="h-px bg-brand-blue/10 dark:bg-white/10 my-2"></div>
                        <a
                            href={profileHref}
                            className="flex items-center gap-2 rounded-lg px-3 py-2 text-sm hover:bg-brand-blue/5 dark:hover:bg-white/5 transition-colors"
                        >
                            <UserRound className="w-4 h-4" />
                            {profileLabel}
                        </a>
                        <button
                            onClick={handleLogout}
                            className="mt-2 w-full flex items-center gap-2 rounded-lg px-3 py-2 text-sm text-white bg-brand-green shadow-sm hover:shadow-md transition-all"
                        >
                            <LogOut className="w-4 h-4" />
                            Sair
                        </button>
                    </PopoverContent>
                </Popover>
            </header>

            <div
                ref={scrollRef}
                className="relative z-10 h-screen overflow-y-auto scroll-snap-y"
                onWheel={handleWheel}
                onScroll={handleScrollSync}
                onTouchStart={handleTouchStart}
                onTouchEnd={handleTouchEnd}
            >
                <section
                    ref={(el) => {
                        if (el) {
                            sectionRefs.current[0] = el;
                        }
                    }}
                    className="scroll-snap-start min-h-screen flex items-start justify-center px-6 pt-20 pb-16"
                >
                    <div className="dashboard-stage">
                        <div className="absolute dashboard-logo">
                            <div className="relative flex items-center justify-center w-full h-full">
                                <img 
                                    src="https://i.ibb.co/9HFVnY4x/Gemini-Generated-Image-oc1jgfoc1jgfoc1j-Photoroom.png"
                                    alt="Logo RadiologIA"
                                    className="relative w-full h-full object-contain drop-shadow-xl logo-flag-sync"
                                />
                            </div>
                        </div>

                        <div className="absolute dashboard-wordmark">
                            <img
                                src={isDarkMode
                                    ? "https://i.ibb.co/yBWfdYwN/Captura-de-Tela-2026-01-08-s-13-17-59-removebg-preview.png"
                                    : "https://i.ibb.co/B5Lsvm4M/Captura-de-Tela-2026-01-08-s-12-59-47-removebg-preview.png"}
                                alt="radiologIA"
                                className="w-full h-full object-contain drop-shadow-sm"
                            />
                        </div>

                        <motion.div 
                            initial={{ opacity: 0, y: 20 }} 
                            animate={{ opacity: 1, y: 0 }} 
                            transition={{ duration: 0.8, delay: 0.4 }}
                            className="absolute dashboard-card-triagem"
                        >
                            <FeatureCard
                                title="Triagem Inteligente"
                                description="Envie exames para análise inicial e priorização automática."
                                infoId="triagem-info"
                                Icon={ScanLine}
                                accent="text-brand-green"
                                accentSoft="bg-brand-green/20"
                                accentBorder="border-brand-green/30"
                                decorations={
                                    <>
                                        <div className="absolute inset-0 opacity-40 dark:opacity-20 [background-image:linear-gradient(rgba(15,23,42,0.04)_1px,transparent_1px),linear-gradient(90deg,rgba(15,23,42,0.04)_1px,transparent_1px)] [background-size:28px_28px]"></div>
                                        <div className="absolute left-10 top-10 h-32 w-32 rounded-full border border-brand-green/25 bg-brand-green/10 blur-2xl"></div>
                                        <div className="absolute right-10 top-24 h-40 w-24 rounded-[28px] border border-brand-green/20 bg-white/40 dark:bg-white/5 backdrop-blur-md"></div>
                                        <div className="absolute right-14 top-16 h-44 w-1.5 bg-[linear-gradient(180deg,rgba(16,185,129,0.35),transparent)]"></div>
                                    </>
                                }
                            />
                        </motion.div>

                        <motion.div 
                            initial={{ opacity: 0, y: 20 }} 
                            animate={{ opacity: 1, y: 0 }} 
                            transition={{ duration: 0.8, delay: 0.5 }}
                            className="absolute dashboard-card-auxiliar"
                        >
                            <FeatureCard
                                title="Auxiliar Diagnóstico"
                                description="Aprofunde a análise com contexto clínico e histórico."
                                infoId="auxiliar-info"
                                Icon={Brain}
                                accent="text-brand-yellow"
                                accentSoft="bg-brand-yellow/20"
                                accentBorder="border-brand-yellow/30"
                                decorations={
                                    <>
                                        <div className="absolute inset-0 opacity-40 dark:opacity-20 [background-image:radial-gradient(rgba(234,179,8,0.28)_1px,transparent_1px)] [background-size:20px_20px]"></div>
                                        <div className="absolute left-12 top-12 h-28 w-28 rounded-full bg-brand-yellow/15 blur-2xl"></div>
                                        <div className="absolute right-12 bottom-10 h-20 w-40 rounded-[26px] border border-brand-yellow/25 bg-white/40 dark:bg-white/5 backdrop-blur-md -rotate-6"></div>
                                        <div className="absolute left-16 bottom-12 h-24 w-24 rounded-2xl border border-brand-yellow/20 bg-[radial-gradient(circle,rgba(234,179,8,0.2),transparent_70%)]"></div>
                                    </>
                                }
                            />
                        </motion.div>
                    </div>
                </section>

                <section
                    id="triagem-info"
                    ref={(el) => {
                        if (el) {
                            sectionRefs.current[1] = el;
                        }
                    }}
                    className="scroll-snap-start min-h-screen flex items-center justify-center px-6 py-16"
                >
                    <div className="w-full max-w-5xl rounded-[28px] border border-dashed border-brand-green/40 bg-white/60 dark:bg-white/5 backdrop-blur-md p-8 text-brand-blue dark:text-white">
                        <h3 className="text-2xl font-semibold">Triagem Inteligente</h3>
                        <p className="mt-2 text-sm text-slate-600 dark:text-slate-300">
                            Em breve: descrição completa e fluxo de uso da triagem.
                        </p>
                    </div>
                </section>

                <section
                    id="auxiliar-info"
                    ref={(el) => {
                        if (el) {
                            sectionRefs.current[2] = el;
                        }
                    }}
                    className="scroll-snap-start min-h-screen flex items-center justify-center px-6 py-16"
                >
                    <div className="w-full max-w-5xl rounded-[28px] border border-dashed border-brand-yellow/40 bg-white/60 dark:bg-white/5 backdrop-blur-md p-8 text-brand-blue dark:text-white">
                        <h3 className="text-2xl font-semibold">Auxiliar Diagnóstico</h3>
                        <p className="mt-2 text-sm text-slate-600 dark:text-slate-300">
                            Em breve: descrição completa e fluxo do auxiliar diagnóstico.
                        </p>
                    </div>
                </section>
            </div>
        </div>
    );
}
