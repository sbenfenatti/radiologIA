'use client';

import Image from 'next/image';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { ZoomIn, ZoomOut, RotateCcw } from 'lucide-react';
import type { AnalysisResult } from '@/lib/types';
import ChatAssistant from '@/components/chat-assistant';
import { SidebarTrigger } from '@/components/ui/sidebar';

const PriorityBadge = ({ priority }: { priority: 'low' | 'medium' | 'high' }) => {
  const className = {
    low: 'bg-green-500 hover:bg-green-600',
    medium: 'bg-yellow-500 hover:bg-yellow-600',
    high: 'bg-red-500 hover:bg-red-600',
  }

  const priorityLabels = {
    low: 'Baixa',
    medium: 'Média',
    high: 'Alta'
  };

  return <Badge variant={priority === 'high' ? 'destructive' : 'default'} className={`${className[priority]} text-white`}>{priorityLabels[priority]}</Badge>;
};

export default function AnalysisClient({ analysisData }: { analysisData: AnalysisResult }) {
  return (
    <div className="flex-1 flex flex-col md:flex-row h-screen max-h-screen overflow-hidden">
      <div className="flex-1 flex flex-col p-4 space-y-4">
        <div className="flex items-center gap-2">
            <SidebarTrigger className="md:hidden"/>
            <h2 className="text-2xl font-bold tracking-tight font-headline">Auxílio Diagnóstico</h2>
        </div>
        <Card className="flex-1 flex flex-col">
          <CardHeader className="flex flex-row items-center justify-between">
            <CardTitle>{analysisData.name}</CardTitle>
            <div className="flex items-center space-x-2">
              <Button variant="outline" size="icon">
                <ZoomIn className="h-4 w-4" />
              </Button>
              <Button variant="outline" size="icon">
                <ZoomOut className="h-4 w-4" />
              </Button>
              <Button variant="outline" size="icon">
                <RotateCcw className="h-4 w-4" />
              </Button>
            </div>
          </CardHeader>
          <CardContent className="flex-1 relative">
            <Image
              src={analysisData.dataUri || analysisData.url}
              alt={`Raio-X de ${analysisData.name}`}
              fill
              className="object-contain rounded-md"
              data-ai-hint="dental xray"
            />
          </CardContent>
        </Card>
      </div>

      <div className="md:w-1/3 w-full flex flex-col p-4 border-l bg-slate-50 dark:bg-slate-900/50 h-full">
         <ChatAssistant analysisResult={analysisData} />
      </div>
    </div>
  );
}
