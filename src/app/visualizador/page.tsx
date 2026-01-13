'use client';

import { useEffect, useMemo, useState } from 'react';
import FindingsViewer, {
  normalizeFindings,
  type RawFinding,
} from '@/components/triagem/FindingsViewer';

type SinglePayload = {
  name?: string;
  findings?: unknown[];
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

export default function VisualizadorSandboxPage() {
  const [payload, setPayload] = useState<SinglePayload | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    let isActive = true;
    const load = async () => {
      try {
        const response = await fetch('/batch_test/triagem_single_result.json', { cache: 'no-store' });
        if (!response.ok) {
          throw new Error('Nao foi possivel carregar o JSON local.');
        }
        const data = await response.json();
        const selected = Array.isArray(data) ? data[0] : data;
        if (isActive) {
          setPayload(selected as SinglePayload);
        }
      } catch (err) {
        if (isActive) {
          setError(err instanceof Error ? err.message : 'Falha ao carregar JSON.');
        }
      } finally {
        if (isActive) {
          setIsLoading(false);
        }
      }
    };
    load();
    return () => {
      isActive = false;
    };
  }, []);

  const imageUrl = payload?.name ? `/batch_test/${payload.name}` : undefined;
  const findings = Array.isArray(payload?.findings) ? payload?.findings : [];
  const normalizedFindings = useMemo(
    () => normalizeFindings(findings as RawFinding[]),
    [findings],
  );
  const summary = useMemo(() => {
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

  return (
    <div className="min-h-screen bg-background text-gray-900 dark:text-gray-100 px-6 py-10">
      <div className="mx-auto max-w-6xl">
        <div className="mb-6">
          <p className="text-xs uppercase tracking-[0.3em] text-brand-blue/60 dark:text-white/60">
            Sandbox
          </p>
          <h1 className="text-2xl font-semibold text-brand-blue dark:text-white">
            Visualizador de achados
          </h1>
          <p className="mt-1 text-sm text-slate-600 dark:text-slate-300">
            Ambiente isolado para testar overlay, filtros e integraçao entre modelos.
          </p>
        </div>

        {error ? (
          <div className="rounded-2xl border border-white/20 bg-white/60 p-4 text-sm text-red-600 dark:border-white/10 dark:bg-white/5 dark:text-red-300">
            {error}
          </div>
        ) : (
          <>
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
                    : 'Nenhuma patologia detectada no lote atual.'}
                </p>
                <p className="mt-2 text-xs text-slate-500 dark:text-slate-400">
                  {summary.anatomy.length
                    ? `Estruturas segmentadas: ${summary.anatomy.join(', ')}.`
                    : 'Sem estruturas segmentadas neste caso.'}
                </p>
              </div>
            </div>
            <FindingsViewer
              imageUrl={imageUrl}
              findings={findings}
              isLoading={isLoading}
              title="RX panoramico de teste"
              subtitle="JSON local com YOLO + Detectron (batch_test)."
              enableToothFusionPreview
              enableClickSelect
              showList={false}
            />
          </>
        )}
      </div>
    </div>
  );
}
