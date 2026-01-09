'use client';

import { motion } from 'framer-motion';
import { ScanSearch, BrainCircuit } from 'lucide-react';
import Link from 'next/link';

const GlassCard = ({ children, className = '' }: { children: React.ReactNode, className?: string }) => (
    <div className={`bg-white/5 border border-white/10 rounded-2xl p-6 backdrop-blur-md group ${className}`}>
        {children}
    </div>
);

export default function PresentationPage() {
    return (
        <div className="bg-[#001e45] text-white min-h-screen flex flex-col items-center justify-center p-4 antialiased relative overflow-hidden space-y-16">
            {/* Fundo Vivo */}
            <div className="fixed inset-0 w-full h-full overflow-hidden -z-10">
                <div className="absolute inset-0 bg-[url('https://grainy-gradients.vercel.app/noise.svg')] opacity-10"></div>
                <div className="absolute top-1/4 left-1/4 w-[500px] h-[500px] bg-brand-green/20 rounded-full mix-blend-screen filter blur-[100px] animate-blob"></div>
                <div className="absolute top-1/3 right-1/4 w-[400px] h-[400px] bg-brand-yellow/10 rounded-full mix-blend-screen filter blur-[80px] animate-blob animation-delay-2000"></div>
                <div className="absolute bottom-1/4 left-1/2 w-[600px] h-[600px] bg-brand-blue/30 rounded-full mix-blend-screen filter blur-[120px] animate-blob animation-delay-4000"></div>
            </div>

            {/* Container da Logo */}
            <div className="flex flex-row items-center justify-center">
                <img 
                    src="https://i.ibb.co/9HFVnY4x/Gemini-Generated-Image-oc1jgfoc1jgfoc1j-Photoroom.png"
                    alt="Logo RadiologIA"
                    className="w-48 h-48 object-contain drop-shadow-lg"
                />
                <img 
                    src="https://i.ibb.co/yBWfdYwN/Captura-de-Tela-2026-01-08-s-13-17-59-removebg-preview.png" 
                    alt="radiologIA"
                    className="h-10 object-contain drop-shadow-lg -ml-4"
                />
            </div>

            {/* Container dos Cards */}
            <motion.div 
                initial={{ opacity: 0, y: 20 }} 
                animate={{ opacity: 1, y: 0 }} 
                transition={{ duration: 1, delay: 0.5 }}
                className="w-full max-w-4xl"
            >
                <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                   
                    <Link href="#" className="block group">
                        <GlassCard className="h-full hover:border-brand-green/50 hover:-translate-y-2 transition-transform duration-300 ease-in-out">
                            <div className="flex flex-col items-center text-center p-4">
                                <div className="p-4 bg-brand-green/10 rounded-full border border-brand-green/20 mb-4">
                                    <ScanSearch className="w-10 h-10 text-brand-green"/>
                                </div>
                                <h2 className="text-2xl font-bold mb-2 text-white">Triagem Rápida</h2>
                                <p className="text-gray-300 text-sm">Envie um ou mais exames para uma análise inicial e priorização automática.</p>
                            </div>
                        </GlassCard>
                    </Link>

                    <Link href="#" className="block group">
                        <GlassCard className="h-full hover:border-brand-yellow/50 hover:-translate-y-2 transition-transform duration-300 ease-in-out">
                            <div className="flex flex-col items-center text-center p-4">
                                <div className="p-4 bg-brand-yellow/10 rounded-full border border-brand-yellow/20 mb-4">
                                    <BrainCircuit className="w-10 h-10 text-brand-yellow"/>
                                </div>
                                <h2 className="text-2xl font-bold mb-2 text-white">Assistente Clínico</h2>
                                <p className="text-gray-300 text-sm">Faça uma análise profunda de um caso, correlacionando com histórico do paciente.</p>
                            </div>
                        </GlassCard>
                    </Link>

                </div>
            </motion.div>
        </div>
    );
}
