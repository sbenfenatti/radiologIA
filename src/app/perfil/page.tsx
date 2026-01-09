'use client';

import Link from 'next/link';

export default function PerfilPage() {
  return (
    <div className="min-h-screen bg-background text-gray-900 dark:text-gray-100 flex items-center justify-center p-8">
      <div className="max-w-xl w-full glass-panel rounded-3xl p-8 text-center">
        <h1 className="text-2xl font-bold text-brand-blue dark:text-white">Perfil</h1>
        <p className="mt-3 text-sm text-brand-blue/70 dark:text-white/70">
          Área de perfil em construção. Em breve vamos permitir edição de dados e preferências.
        </p>
        <Link
          href="/dashboard"
          className="mt-6 inline-flex items-center justify-center rounded-full bg-brand-green px-6 py-2 text-sm font-semibold text-white shadow-md"
        >
          Voltar ao dashboard
        </Link>
      </div>
    </div>
  );
}
