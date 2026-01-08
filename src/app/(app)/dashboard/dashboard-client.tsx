'use client';

import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import PrevalenceChart from '@/components/charts/prevalence-chart';
import AgeDistributionChart from '@/components/charts/age-distribution-chart';
import FindingsByPriorityChart from '@/components/charts/findings-by-priority-chart';
import { Button } from '@/components/ui/button';
import { Download } from 'lucide-react';
import { SidebarTrigger } from '@/components/ui/sidebar';

export default function DashboardClient() {
  return (
    <div className="flex-1 space-y-4 p-4 md:p-8 pt-6">
      <div className="flex items-center justify-between space-y-2">
        <div className="flex items-center gap-2">
          <SidebarTrigger className="md:hidden"/>
          <h2 className="text-3xl font-bold tracking-tight font-headline">Painel Epidemiológico</h2>
        </div>
        <div className="flex items-center space-x-2">
          <Button>
            <Download className="mr-2 h-4 w-4" />
            Baixar Relatório
          </Button>
        </div>
      </div>
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total de Análises</CardTitle>
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" className="h-4 w-4 text-muted-foreground"><path d="M21.21 15.89A10 10 0 1 1 8 2.83" /><path d="M22 12A10 10 0 0 0 12 2v10z" /></svg>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">12.540</div>
            <p className="text-xs text-muted-foreground">+20.1% do último mês</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Casos de Alta Prioridade</CardTitle>
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" className="h-4 w-4 text-muted-foreground"><path d="M12 2c2.4 0 4.7.4 6.9 1.2l-2.6 2.6A6.94 6.94 0 0 0 12 5c-3.9 0-7 3.1-7 7s3.1 7 7 7 7-3.1 7-7a6.94 6.94 0 0 0-.2-1.7l2.6-2.6A9.94 9.94 0 0 1 22 12c0 5.5-4.5 10-10 10S2 17.5 2 12 6.5 2 12 2Z" /><path d="m14 6-2-2-2 2" /><path d="M12 14v- телевизор" /></svg>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">231</div>
            <p className="text-xs text-muted-foreground">+15% do último mês</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Achado Mais Comum</CardTitle>
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" className="h-4 w-4 text-muted-foreground"><path d="M11 20A8 8 0 1 0 3 12v7a2 2 0 0 0 2 2h4.5a2 2 0 0 0 2-2zM3 12V5a2 2 0 0 1 2-2h12a2 2 0 0 1 2 2v12a2 2 0 0 1-2 2h-2.5" /><path d="M12.5 7.5a1 1 0 1 0 0-2 1 1 0 0 0 0 2Z" /><path d="M16.5 7.5a1 1 0 1 0 0-2 1 1 0 0 0 0 2Z" /><path d="M8.5 7.5a1 1 0 1 0 0-2 1 1 0 0 0 0 2Z" /></svg>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">Cáries</div>
            <p className="text-xs text-muted-foreground">45% de todos os achados</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Precisão da IA</CardTitle>
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" className="h-4 w-4 text-muted-foreground"><path d="m12 3-1.9 4.1-4.2.6 3 2.9-.7 4.2 3.8-2 3.8 2-.7-4.2 3-2.9-4.2-.6z" /></svg>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">98.2%</div>
            <p className="text-xs text-muted-foreground">Confirmado por especialistas</p>
          </CardContent>
        </Card>
      </div>
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-7">
        <Card className="col-span-4">
          <CardHeader>
            <CardTitle>Prevalência de Condições Comuns</CardTitle>
            <CardDescription>Um resumo das condições odontológicas mais comuns observadas.</CardDescription>
          </CardHeader>
          <CardContent className="pl-2">
            <PrevalenceChart />
          </CardContent>
        </Card>
        <Card className="col-span-4 md:col-span-3">
          <CardHeader>
            <CardTitle>Achados por Prioridade</CardTitle>
            <CardDescription>Distribuição de prioridades de triagem atribuídas pela IA.</CardDescription>
          </CardHeader>
          <CardContent>
            <FindingsByPriorityChart />
          </CardContent>
        </Card>
         <Card className="col-span-4 lg:col-span-7">
          <CardHeader className="flex flex-row items-center gap-4 space-y-0">
            <div className="grid gap-2">
                <CardTitle>Análise por Faixa Etária</CardTitle>
                <CardDescription>Prevalência de condições chave em diferentes faixas etárias.</CardDescription>
            </div>
            <Select defaultValue="caries-gingivitis">
                <SelectTrigger className="ml-auto w-[220px]">
                    <SelectValue placeholder="Selecionar métricas" />
                </SelectTrigger>
                <SelectContent>
                    <SelectItem value="caries-gingivitis">Cáries e Gengivite</SelectItem>
                    <SelectItem value="periodontitis-abscess">Periodontite e Abscesso</SelectItem>
                    <SelectItem value="impacted-cysts">Dentes Inclusos e Cistos</SelectItem>
                </SelectContent>
            </Select>
          </CardHeader>
          <CardContent className="pl-2">
            <AgeDistributionChart />
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
