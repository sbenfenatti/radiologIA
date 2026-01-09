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
  AlertCircle, 
  Activity, 
  Moon, 
  Sun,
  Key
} from 'lucide-react';

import { 
  Form, 
  FormControl, 
  FormField, 
  FormItem, 
  FormLabel, 
  FormMessage 
} from '@/components/ui/form';
import { initiateEmailSignIn, useAuth, useUser } from '@/firebase';
import { cn } from '@/lib/utils';

const loginSchema = z.object({
  email: z.string().email({ message: 'Por favor, insira um e-mail válido.' }),
  password: z.string().min(6, { message: 'A senha deve ter pelo menos 6 caracteres.' }),
});

type LoginFormValues = z.infer<typeof loginSchema>;

export default function LoginPage() {
  const auth = useAuth();
  const { user, isUserLoading } = useUser();
  const router = useRouter();
  
  const [showPassword, setShowPassword] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
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
      await initiateEmailSignIn(auth, data.email, data.password);
    } catch (error) {
      console.error(error);
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleTokenSubmit = () => {
    if (token.toUpperCase() === 'PRO2026') {
      router.push('/dashboard');
    } else {
      alert("🚫 Token Inválido.");
    }
  };

  useEffect(() => {
    if (user && !isUserLoading) {
      router.push('/dashboard');
    }
  }, [user, isUserLoading, router]);

  if (isUserLoading || user) {
    return (
      <div className="flex h-screen w-full items-center justify-center bg-gray-50 dark:bg-[#001e45]">
        <div className="w-8 h-8 border-4 border-brand-green border-t-transparent rounded-full animate-spin"></div>
      </div>
    );
  }

  return (
    <div className="min-h-screen flex items-center justify-center relative overflow-hidden transition-colors duration-500 bg-gray-50 dark:bg-[#001e45] text-gray-900 dark:text-gray-100 p-4">
      
      {/* Botão de Tema */}
      <button 
        onClick={toggleTheme}
        className="absolute top-6 right-6 z-50 p-3 rounded-full glass-panel shadow-lg hover:scale-110 transition-transform cursor-pointer text-brand-blue dark:text-brand-yellow border-none"
        title="Alternar Tema"
      >
        {isDarkMode ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
      </button>

      {/* Fundo Vivo */}
      <div className="absolute inset-0 w-full h-full overflow-hidden -z-10">
        <div className="absolute inset-0 bg-[url('https://grainy-gradients.vercel.app/noise.svg')] opacity-20 brightness-100 contrast-150 mix-blend-overlay"></div>
        <div className="absolute top-0 -left-4 w-96 h-96 bg-brand-green rounded-full mix-blend-multiply filter blur-3xl opacity-40 animate-blob dark:mix-blend-color-dodge dark:opacity-20"></div>
        <div className="absolute top-0 -right-4 w-96 h-96 bg-brand-yellow rounded-full mix-blend-multiply filter blur-3xl opacity-40 animate-blob animation-delay-2000 dark:mix-blend-color-dodge dark:opacity-15"></div>
        <div className="absolute -bottom-32 left-20 w-96 h-96 bg-brand-blue rounded-full mix-blend-multiply filter blur-3xl opacity-40 animate-blob animation-delay-4000 dark:mix-blend-color-dodge dark:opacity-30"></div>
      </div>

      {/* Card Principal */}
      <div className="w-full max-w-sm glass-panel rounded-3xl shadow-2xl overflow-hidden relative z-10 animate-fade-in-up ring-1 ring-white/40 dark:ring-white/5">
        
        {/* Área do Logo */}
        <div className="pt-8 pb-2 px-8 text-center flex flex-col items-center">
            
            <div className="w-40 h-40 mb-0 relative flex items-center justify-center group">
                <div className="absolute inset-0 bg-white/0 dark:bg-white/5 rounded-full blur-xl transition-all duration-500"></div>
                
                <img 
                    src="https://i.ibb.co/9HFVnY4x/Gemini-Generated-Image-oc1jgfoc1jgfoc1j-Photoroom.png" 
                    alt="Logo RadiologIA" 
                    className="relative w-full h-full object-contain transition-transform hover:scale-105 duration-300 drop-shadow-md"
                    onError={(e) => {
                      (e.target as HTMLImageElement).style.display = 'none';
                      document.getElementById('fallback-logo')?.classList.remove('hidden');
                    }}
                />
                
                <div id="fallback-logo" className="hidden relative w-32 h-32 bg-gradient-to-tr from-brand-blue to-brand-green rounded-3xl flex items-center justify-center shadow-lg text-white border border-brand-yellow/30">
                    <Activity className="w-12 h-12" />
                </div>
            </div>

            <div className="h-16 -mt-10 relative w-full flex justify-center">
                <img 
                    src="https://i.ibb.co/B5Lsvm4M/Captura-de-Tela-2026-01-08-s-12-59-47-removebg-preview.png" 
                    alt="radiologIA" 
                    className={cn("h-full object-contain drop-shadow-sm", isDarkMode ? "hidden" : "block")}
                />
                <img 
                    src="https://i.ibb.co/yBWfdYwN/Captura-de-Tela-2026-01-08-s-13-17-59-removebg-preview.png" 
                    alt="radiologIA" 
                    className={cn("h-full object-contain drop-shadow-sm", isDarkMode ? "block" : "hidden")}
                />
            </div>
        </div>

        <div className="p-8 pt-2">
            
            {/* Alert de erro seria aqui */}
            {/* <div id="error-alert" className="mb-4 p-2.5 bg-red-100 dark:bg-red-900/30 border border-red-200 dark:border-red-800 rounded-lg flex items-center gap-3 text-xs text-red-700 dark:text-red-200 animate-pulse">
                <AlertCircle className="w-4 h-4 flex-shrink-0" />
                <span className="font-medium">Credenciais inválidas.</span>
            </div> */}

            <Form {...form}>
              <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-4">
                <FormField
                  control={form.control}
                  name="email"
                  render={({ field }) => (
                    <FormItem className="space-y-1">
                      <FormLabel className="text-[10px] font-bold text-brand-blue dark:text-gray-300 uppercase tracking-wider ml-1 opacity-80">E-mail</FormLabel>
                      <FormControl>
                        <div className="relative group">
                          <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                            <Mail className="w-4 h-4 text-brand-blue/60 dark:text-gray-400 group-focus-within:text-brand-green transition-colors" />
                          </div>
                          <input 
                              {...field}
                              type="email" 
                              placeholder="dr.sergio@radiologia.com.br" 
                              className="w-full pl-9 pr-3 py-2 text-sm rounded-lg glass-input focus:ring-1 focus:ring-brand-green/50 focus:border-brand-green outline-none transition-all placeholder-gray-400 dark:placeholder-gray-500 text-brand-blue dark:text-white font-medium"
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
                        <FormLabel className="text-[10px] font-bold text-brand-blue dark:text-gray-300 uppercase tracking-wider opacity-80">Senha</FormLabel>
                        <Link href="#" className="text-[10px] text-brand-green hover:text-brand-blue dark:text-brand-yellow dark:hover:text-white font-bold hover:underline transition-colors">Esqueceu?</Link>
                      </div>
                      <FormControl>
                        <div className="relative group">
                          <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                            <Lock className="w-4 h-4 text-brand-blue/60 dark:text-gray-400 group-focus-within:text-brand-green transition-colors" />
                          </div>
                          <input 
                              {...field}
                              type={showPassword ? "text" : "password"} 
                              placeholder="••••••••" 
                              className="w-full pl-9 pr-10 py-2 text-sm rounded-lg glass-input focus:ring-1 focus:ring-brand-green/50 focus:border-brand-green outline-none transition-all placeholder-gray-400 dark:placeholder-gray-500 text-brand-blue dark:text-white font-medium"
                              required
                          />
                          <button 
                            type="button" 
                            onClick={() => setShowPassword(!showPassword)}
                            className="absolute inset-y-0 right-0 pr-3 flex items-center text-brand-blue/60 hover:text-brand-blue dark:text-gray-400 dark:hover:text-white transition-colors cursor-pointer"
                          >
                            {showPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
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
                    className="w-full bg-brand-green hover:bg-brand-green-dark text-white font-bold py-3 rounded-xl transition-all duration-300 shadow-lg shadow-brand-green/30 hover:shadow-brand-green/50 backdrop-blur-md border border-white/10 transform hover:-translate-y-0.5 active:scale-[0.98] flex items-center justify-center gap-2 mt-5 relative overflow-hidden group text-sm disabled:opacity-80 disabled:cursor-not-allowed"
                >
                    <div className="absolute inset-0 -translate-x-full group-hover:translate-x-full transition-transform duration-700 bg-gradient-to-r from-transparent via-white/20 to-transparent"></div>
                    <span className="relative z-10">{isSubmitting ? "Autenticando..." : "Entrar"}</span>
                    {!isSubmitting && <ArrowRight className="w-4 h-4 relative z-10" />}
                    {isSubmitting && <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin-custom relative z-10"></div>}
                </button>
              </form>
            </Form>

            {/* Botão de Opções (Compacto) */}
            <div className="relative mt-6 mb-4">
                <div className="absolute inset-0 flex items-center"><div className="w-full border-t border-brand-blue/20 dark:border-white/10"></div></div>
                <div className="relative flex justify-center text-sm">
                    <button 
                        onClick={() => setShowOptions(!showOptions)}
                        className="px-3 py-1 bg-gray-50/50 dark:bg-slate-900/50 text-brand-blue/70 dark:text-brand-white/70 font-medium glass-panel rounded-full text-[10px] uppercase tracking-widest border border-brand-blue/10 dark:border-white/10 hover:bg-white dark:hover:bg-slate-800 hover:text-brand-blue dark:hover:text-white transition-all cursor-pointer shadow-sm hover:shadow-md"
                    >
                        {showOptions ? "Ocultar" : "Opções"}
                    </button>
                </div>
            </div>

            {/* Container Oculto */}
            {showOptions && (
              <div className="flex flex-col gap-3 overflow-hidden transition-all duration-300 animate-slide-down">
                  <div className="flex gap-2 items-center">
                       <div className="relative flex-1 group">
                          <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                              <Key className="w-3.5 h-3.5 text-brand-blue/50 dark:text-gray-400 group-focus-within:text-brand-yellow transition-colors" />
                          </div>
                          <input 
                              type="text" 
                              value={token}
                              onChange={(e) => setToken(e.target.value)}
                              placeholder="Token" 
                              className="w-full pl-8 pr-3 py-2 text-xs rounded-lg glass-input focus:ring-1 focus:ring-brand-yellow/50 focus:border-brand-yellow outline-none transition-all text-brand-blue dark:text-white placeholder-gray-400 dark:placeholder-gray-500"
                          />
                      </div>
                      <button 
                          type="button"
                          onClick={handleTokenSubmit}
                          className="p-2 bg-brand-yellow hover:bg-yellow-400 text-brand-blue font-bold rounded-lg transition-all shadow-sm hover:shadow-md flex items-center justify-center aspect-square border border-brand-yellow"
                          title="Entrar com Token"
                      >
                          <ArrowRight className="w-4 h-4" />
                      </button>
                  </div>
              </div>
            )}

            {/* <p className="text-center mt-6 text-xs text-brand-blue/70 dark:text-gray-400">
                Novo aqui? <Link href="/signup" className="text-brand-green dark:text-brand-yellow font-bold hover:text-brand-green-light dark:hover:text-white hover:underline transition-colors">Criar conta</Link>
            </p> */}
        </div>
      </div>
    </div>
  );
}
