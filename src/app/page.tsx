'use client';

import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { useForm } from 'react-hook-form';
import { z } from 'zod';
import { zodResolver } from '@hookform/resolvers/zod';
import { useEffect, useState } from 'react';
import {
  Mail,
  Lock,
  Eye,
  EyeOff,
  ArrowRight,
  Activity,
  Moon,
  Sun,
  Key,
} from 'lucide-react';

import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from '@/components/ui/form';
import { cn } from '@/lib/utils';
import { supabase } from '@/lib/supabase/client';
import { useSupabaseUser } from '@/hooks/use-supabase-user';
import CreditFooter from '@/components/CreditFooter';

const loginSchema = z.object({
  email: z.string().email({ message: 'Por favor, insira um e-mail válido.' }),
  password: z.string().min(6, { message: 'A senha deve ter pelo menos 6 caracteres.' }),
});

type LoginFormValues = z.infer<typeof loginSchema>;

export default function LoginPage() {
  const router = useRouter();
  const { user, isLoading } = useSupabaseUser();

  const [showPassword, setShowPassword] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isTokenSubmitting, setIsTokenSubmitting] = useState(false);
  const [showOptions, setShowOptions] = useState(false);
  const [token, setToken] = useState('');
  const [isDarkMode, setIsDarkMode] = useState(false);

  const form = useForm<LoginFormValues>({
    resolver: zodResolver(loginSchema),
    defaultValues: {
      email: '',
      password: '',
    },
  });

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
    const newDark = !isDarkMode;
    setIsDarkMode(newDark);
    if (newDark) {
      document.documentElement.classList.add('dark');
      localStorage.setItem('theme', 'dark');
    } else {
      document.documentElement.classList.remove('dark');
      localStorage.setItem('theme', 'light');
    }
  };

  const onSubmit = async (data: LoginFormValues) => {
    setIsSubmitting(true);
    try {
      const { error } = await supabase.auth.signInWithPassword({
        email: data.email,
        password: data.password,
      });
      if (error) {
        console.error(error);
        alert('E-mail ou senha inválidos.');
        return;
      }
      router.push('/dashboard');
    } catch (error) {
      console.error(error);
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleTokenSubmit = async () => {
    if (!token.trim()) {
      alert('Informe o token.');
      return;
    }
    setIsTokenSubmitting(true);
    try {
      const response = await fetch('/api/token-login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ token }),
      });
      if (!response.ok) {
        alert('🚫 Token inválido.');
        return;
      }
      router.push('/dashboard');
    } catch (error) {
      console.error(error);
      alert('Não foi possível validar o token.');
    } finally {
      setIsTokenSubmitting(false);
    }
  };

  useEffect(() => {
    if (!isLoading && user) {
      router.push('/dashboard');
    }
  }, [isLoading, user, router]);

  if (isLoading || user) {
    return (
      <div className="flex h-screen w-full items-center justify-center bg-gray-50 dark:bg-[#001e45]">
        <div className="w-8 h-8 border-4 border-brand-green border-t-transparent rounded-full animate-spin"></div>
      </div>
    );
  }

  return (
    <div className="min-h-screen flex items-center justify-center relative overflow-hidden transition-colors duration-500 text-gray-900 dark:text-gray-100 p-4">
      <button
        onClick={toggleTheme}
        className="absolute top-6 right-6 z-50 p-3 rounded-full glass-panel shadow-lg hover:scale-110 transition-transform cursor-pointer text-brand-blue dark:text-brand-yellow border-none"
        title="Alternar Tema"
      >
        {isDarkMode ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
      </button>

      <div className="absolute inset-0 w-full h-full overflow-hidden -z-10 flag-stage">
        <div className="absolute inset-0 bg-[url('https://grainy-gradients.vercel.app/noise.svg')] opacity-20 brightness-100 contrast-150 mix-blend-overlay"></div>
        <div className="flag-canvas">
          <div className="flag-shape flag-green"></div>
          <div className="flag-shape flag-yellow"></div>
          <div className="flag-shape flag-blue"></div>
          <div className="flag-glow"></div>
        </div>
        <div className="flag-glass"></div>
      </div>

      <div className="w-full max-w-[30rem] sm:max-w-[34rem] lg:max-w-[38rem] glass-panel rounded-3xl shadow-2xl overflow-hidden relative z-10 animate-fade-in-up ring-1 ring-white/40 dark:ring-white/5">
        <div className="pt-10 pb-3 px-10 text-center flex flex-col items-center">
          <div className="w-60 h-60 sm:w-72 sm:h-72 mb-0 relative flex items-center justify-center group">
            <div className="absolute inset-0 bg-white/0 dark:bg-white/5 rounded-full blur-xl transition-all duration-500"></div>

            <img
              src="/brand/atom.png"
              alt="Logo RadiologIA"
              className="relative w-full h-full object-contain transition-transform hover:scale-105 duration-300 drop-shadow-md logo-flag-sync"
              onError={(e) => {
                (e.target as HTMLImageElement).style.display = 'none';
                document.getElementById('fallback-logo')?.classList.remove('hidden');
              }}
            />

            <div
              id="fallback-logo"
              className="hidden relative w-48 h-48 sm:w-56 sm:h-56 bg-gradient-to-tr from-brand-blue to-brand-green rounded-3xl flex items-center justify-center shadow-lg text-white border border-brand-yellow/30"
            >
              <Activity className="w-20 h-20" />
            </div>
          </div>

          <div className="h-20 -mt-12 relative w-full flex justify-center">
            <img
              src="/brand/wordmark-light.png"
              alt="radiologIA"
              className={cn('h-full object-contain drop-shadow-sm', isDarkMode ? 'hidden' : 'block')}
            />
            <img
              src="/brand/wordmark-dark.png"
              alt="radiologIA"
              className={cn('h-full object-contain drop-shadow-sm', isDarkMode ? 'block' : 'hidden')}
            />
          </div>
        </div>

        <div className="p-10 pt-4">
          <Form {...form}>
            <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-5">
              <FormField
                control={form.control}
                name="email"
                render={({ field }) => (
                  <FormItem className="space-y-1">
                    <FormLabel className="text-xs font-bold text-brand-blue dark:text-gray-300 uppercase tracking-wider ml-1 opacity-80">
                      E-mail
                    </FormLabel>
                    <FormControl>
                      <div className="relative group">
                        <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                          <Mail className="w-5 h-5 text-brand-blue/60 dark:text-gray-400 group-focus-within:text-brand-green transition-colors" />
                        </div>
                        <input
                          {...field}
                          type="email"
                          placeholder="email@radiologIA.com"
                          className="w-full pl-10 pr-4 py-3 text-base rounded-lg glass-input focus:ring-1 focus:ring-brand-green/50 focus:border-brand-green outline-none transition-all placeholder-gray-400 dark:placeholder-gray-500 text-brand-blue dark:text-white font-medium"
                          required
                        />
                      </div>
                    </FormControl>
                    <FormMessage className="text-[10px]" />
                  </FormItem>
                )}
              />

              <FormField
                control={form.control}
                name="password"
                render={({ field }) => (
                  <FormItem className="space-y-1">
                    <div className="flex justify-between items-center px-1">
                      <FormLabel className="text-xs font-bold text-brand-blue dark:text-gray-300 uppercase tracking-wider opacity-80">
                        Senha
                      </FormLabel>
                      <Link
                        href="#"
                        className="text-xs text-brand-green hover:text-brand-blue dark:text-brand-yellow dark:hover:text-white font-bold hover:underline transition-colors"
                      >
                        Esqueceu?
                      </Link>
                    </div>
                    <FormControl>
                      <div className="relative group">
                        <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                          <Lock className="w-5 h-5 text-brand-blue/60 dark:text-gray-400 group-focus-within:text-brand-green transition-colors" />
                        </div>
                        <input
                          {...field}
                          type={showPassword ? 'text' : 'password'}
                          placeholder="••••••••"
                          className="w-full pl-10 pr-12 py-3 text-base rounded-lg glass-input focus:ring-1 focus:ring-brand-green/50 focus:border-brand-green outline-none transition-all placeholder-gray-400 dark:placeholder-gray-500 text-brand-blue dark:text-white font-medium"
                          required
                        />
                        <button
                          type="button"
                          onClick={() => setShowPassword(!showPassword)}
                          className="absolute inset-y-0 right-0 pr-3 flex items-center text-brand-blue/60 hover:text-brand-blue dark:text-gray-400 dark:hover:text-white transition-colors cursor-pointer"
                        >
                          {showPassword ? <EyeOff className="w-5 h-5" /> : <Eye className="w-5 h-5" />}
                        </button>
                      </div>
                    </FormControl>
                    <FormMessage className="text-[10px]" />
                  </FormItem>
                )}
              />

              <button
                type="submit"
                disabled={isSubmitting}
                className="w-full bg-brand-green hover:bg-brand-green-dark text-white font-bold py-4 rounded-xl transition-all duration-300 shadow-lg shadow-brand-green/30 hover:shadow-brand-green/50 backdrop-blur-md border border-white/10 transform hover:-translate-y-0.5 active:scale-[0.98] flex items-center justify-center gap-2 mt-6 relative overflow-hidden group text-base disabled:opacity-80 disabled:cursor-not-allowed"
              >
                <div className="absolute inset-0 -translate-x-full group-hover:translate-x-full transition-transform duration-700 bg-gradient-to-r from-transparent via-white/20 to-transparent"></div>
                <span className="relative z-10">{isSubmitting ? 'Autenticando...' : 'Entrar'}</span>
                {!isSubmitting && <ArrowRight className="w-5 h-5 relative z-10" />}
                {isSubmitting && (
                  <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin-custom relative z-10"></div>
                )}
              </button>
            </form>
          </Form>

          <div className="relative mt-6 mb-4">
            <div className="absolute inset-0 flex items-center">
              <div className="w-full border-t border-brand-blue/20 dark:border-white/10"></div>
            </div>
            <div className="relative flex justify-center text-sm">
              <button
                onClick={() => setShowOptions(!showOptions)}
                className="px-4 py-1.5 bg-gray-50/50 dark:bg-slate-900/50 text-brand-blue/70 dark:text-brand-white/70 font-medium glass-panel rounded-full text-xs uppercase tracking-widest border border-brand-blue/10 dark:border-white/10 hover:bg-white dark:hover:bg-slate-800 hover:text-brand-blue dark:hover:text-white transition-all cursor-pointer shadow-sm hover:shadow-md"
              >
                {showOptions ? 'Ocultar' : 'Opções'}
              </button>
            </div>
          </div>

          {showOptions && (
            <div className="flex flex-col gap-3 overflow-hidden transition-all duration-300 animate-slide-down">
              <div className="flex gap-2 items-center">
                <div className="relative flex-1 group">
                  <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                    <Key className="w-4 h-4 text-brand-blue/50 dark:text-gray-400 group-focus-within:text-brand-yellow transition-colors" />
                  </div>
                  <input
                    type="text"
                    value={token}
                    onChange={(e) => setToken(e.target.value)}
                    placeholder="Token"
                    className="w-full pl-9 pr-3 py-2.5 text-sm rounded-lg glass-input focus:ring-1 focus:ring-brand-yellow/50 focus:border-brand-yellow outline-none transition-all text-brand-blue dark:text-white placeholder-gray-400 dark:placeholder-gray-500"
                  />
                </div>
                <button
                  type="button"
                  onClick={handleTokenSubmit}
                  disabled={isTokenSubmitting}
                  className="p-2.5 bg-brand-yellow hover:bg-yellow-400 text-brand-blue font-bold rounded-lg transition-all shadow-sm hover:shadow-md flex items-center justify-center aspect-square border border-brand-yellow disabled:opacity-70 disabled:cursor-not-allowed"
                  title="Entrar com Token"
                >
                  {isTokenSubmitting ? (
                    <div className="w-5 h-5 border-2 border-brand-blue/30 border-t-brand-blue rounded-full animate-spin-custom"></div>
                  ) : (
                    <ArrowRight className="w-5 h-5" />
                  )}
                </button>
              </div>
            </div>
          )}
          <CreditFooter className="mt-6" />
        </div>
      </div>
    </div>
  );
}
