import type { ChartConfig } from "@/components/ui/chart"

export const prevalenceChartData = [
  { condition: "Cáries", prevalence: 45, fill: "var(--color-caries)" },
  { condition: "Gengivite", prevalence: 30, fill: "var(--color-gingivitis)" },
  { condition: "Periodontite", prevalence: 22, fill: "var(--color-periodontitis)" },
  { condition: "Dentes Inclusos", prevalence: 15, fill: "var(--color-impacted)" },
  { condition: "Abscesso", prevalence: 8, fill: "var(--color-abscess)" },
  { condition: "Cistos", prevalence: 5, fill: "var(--color-cysts)" },
]

export const prevalenceChartConfig = {
  prevalence: {
    label: "Prevalência (%)",
  },
  caries: {
    label: "Cáries",
    color: "hsl(var(--chart-1))",
  },
  gingivitis: {
    label: "Gengivite",
    color: "hsl(var(--chart-2))",
  },
  periodontitis: {
    label: "Periodontite",
    color: "hsl(var(--chart-3))",
  },
  impacted: {
    label: "Dentes Inclusos",
    color: "hsl(var(--chart-4))",
  },
  abscess: {
    label: "Abscesso",
    color: "hsl(var(--chart-5))",
  },
  cysts: {
    label: "Cistos",
    color: "hsl(var(--chart-1))",
  },
} satisfies ChartConfig

export const ageDistributionData = [
  { ageGroup: "18-24", caries: 28, gingivitis: 35 },
  { ageGroup: "25-34", caries: 42, gingivitis: 45 },
  { ageGroup: "35-44", caries: 55, gingivitis: 50 },
  { ageGroup: "45-54", caries: 68, gingivitis: 62 },
  { ageGroup: "55-64", caries: 75, gingivitis: 70 },
  { ageGroup: "65+", caries: 82, gingivitis: 78 },
]

export const ageDistributionConfig = {
  caries: {
    label: "Cáries",
    color: "hsl(var(--chart-1))",
  },
  gingivitis: {
    label: "Gengivite",
    color: "hsl(var(--chart-2))",
  },
} satisfies ChartConfig


export const findingsByPriorityData = [
    { priority: 'Alta', count: 18, fill: 'hsl(var(--destructive))' },
    { priority: 'Média', count: 45, fill: 'hsl(var(--chart-4))' },
    { priority: 'Baixa', count: 37, fill: 'hsl(var(--chart-1))' },
]

export const findingsByPriorityConfig = {
  count: {
    label: "Contagem",
  },
  high: {
    label: "Alta",
    color: "hsl(var(--destructive))",
  },
  medium: {
    label: "Média",
    color: "hsl(var(--chart-4))",
  },
  low: {
    label: "Baixa",
    color: "hsl(var(--chart-1))",
  },
} satisfies ChartConfig;

export const patientHistory = `
- Idade: 45
- Gênero: Masculino
- Última Visita: 6 meses atrás
- Queixa Principal: Sensibilidade ocasional no quadrante inferior direito.
- Histórico Médico: Nenhuma doença sistêmica conhecida. Não fumante.
- Histórico Odontológico: Múltiplas restaurações. Sem histórico de cirurgia periodontal.
`
