'use client';

import { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { ScanSearch, BrainCircuit, Moon, Sun, UserRound, LogOut } from 'lucide-react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { useSupabaseUser } from '@/hooks/use-supabase-user';
import { useSessionExpiry } from '@/hooks/use-session-expiry';
import { supabase } from '@/lib/supabase/client';
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover';

const GlassCard = ({ children, className = '' }: { children: React.ReactNode, className?: string }) => (
    <div className={`bg-white/70 dark:bg-white/5 border border-white/50 dark:border-white/10 rounded-2xl p-6 backdrop-blur-md group ${className}`}>
        {children}
    </div>
);

const layout = {
    wordmark: {
        x: 677.3749969415367,
        y: 437.1041749250144,
        w: 289.51562707172707,
        h: 104.17708405386657,
    },
    cardTriagem: {
        x: 71.65624796319753,
        y: 496.3958296496421,
        w: 606.8645889023319,
        h: 346.18750376068056,
    },
    cardAuxiliar: {
        x: 973.2968855290674,
        y: 499.01562139438465,
        w: 606.8645889023319,
        h: 346.18750376068056,
    },
};

const ARTBOARD = { width: 1600, height: 900 };

const toStyle = (rect: { x: number; y: number; w: number; h: number }) => ({
    left: `${(rect.x / ARTBOARD.width) * 100}%`,
    top: `${(rect.y / ARTBOARD.height) * 100}%`,
    width: `${(rect.w / ARTBOARD.width) * 100}%`,
    height: `${(rect.h / ARTBOARD.height) * 100}%`,
});

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
    const isAuthenticated = Boolean(user || hasTokenAccess);
    const authIdentity = user?.id ?? (hasTokenAccess ? 'visitor' : null);

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
            localStorage.removeItem('radiologia.auth.start');
            localStorage.removeItem('radiologia.auth.last');
            localStorage.removeItem('radiologia.auth.id');
            router.push('/');
        }
    };

    useSessionExpiry({ isActive: isAuthenticated, identity: authIdentity, onExpire: handleLogout });

    if (isLoading || !tokenChecked) {
        return (
            <div className="flex h-screen w-full items-center justify-center">
                <div className="w-8 h-8 border-4 border-brand-green border-t-transparent rounded-full animate-spin"></div>
            </div>
        );
    }

    return (
        <div className="min-h-screen flex flex-col items-center justify-start px-6 pb-24 pt-12 antialiased relative overflow-hidden transition-colors duration-500 text-gray-900 dark:text-gray-100 bg-background">
            <div className="absolute inset-0 z-0 pointer-events-none flex items-center justify-center">
                <img
                    src="https://i.ibb.co/9HFVnY4x/Gemini-Generated-Image-oc1jgfoc1jgfoc1j-Photoroom.png"
                    alt="Logo RadiologIA"
                    className="w-[70vw] max-w-[980px] opacity-90 blur-2xl object-contain"
                />
            </div>

            <header className="absolute right-6 top-6 z-30 pointer-events-auto flex flex-wrap items-center gap-3">
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
                        <Link
                            href={profileHref}
                            className="flex items-center gap-2 rounded-lg px-3 py-2 text-sm hover:bg-brand-blue/5 dark:hover:bg-white/5 transition-colors"
                        >
                            <UserRound className="w-4 h-4" />
                            {profileLabel}
                        </Link>
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

            <div className="relative z-10 w-full max-w-[1600px] aspect-[16/9] mx-auto">
                <div className="absolute z-10" style={toStyle(layout.wordmark)}>
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
                    className="absolute z-10"
                    style={toStyle(layout.cardTriagem)}
                >
                    <Link href="/triagem" className="block group h-full">
                        <GlassCard className="h-full hover:border-brand-green/50 hover:-translate-y-2 transition-transform duration-300 ease-in-out">
                            <div className="flex flex-col items-center text-center p-6">
                                <div className="p-4 bg-brand-green/10 rounded-full border border-brand-green/20 mb-4">
                                    <ScanSearch className="w-11 h-11 text-brand-green"/>
                                </div>
                                <h2 className="text-2xl font-bold mb-2 text-brand-blue dark:text-white">Triagem</h2>
                                <p className="text-slate-600 dark:text-gray-300 text-sm">Envie exames para análise inicial e priorização automática.</p>
                            </div>
                        </GlassCard>
                    </Link>
                </motion.div>

                <motion.div
                    initial={{ opacity: 0, y: 20 }} 
                    animate={{ opacity: 1, y: 0 }} 
                    transition={{ duration: 0.8, delay: 0.5 }}
                    className="absolute z-10"
                    style={toStyle(layout.cardAuxiliar)}
                >
                    <Link href="/auxiliar" className="block group h-full">
                        <GlassCard className="h-full hover:border-brand-yellow/50 hover:-translate-y-2 transition-transform duration-300 ease-in-out">
                            <div className="flex flex-col items-center text-center p-6">
                                <div className="p-4 bg-brand-yellow/10 rounded-full border border-brand-yellow/20 mb-4">
                                    <BrainCircuit className="w-11 h-11 text-brand-yellow"/>
                                </div>
                                <h2 className="text-2xl font-bold mb-2 text-brand-blue dark:text-white">Auxiliar Diagnóstico</h2>
                                <p className="text-slate-600 dark:text-gray-300 text-sm">Aprofunde a análise com contexto clínico e histórico.</p>
                            </div>
                        </GlassCard>
                    </Link>
                </motion.div>
            </div>
        </div>
    );
}
