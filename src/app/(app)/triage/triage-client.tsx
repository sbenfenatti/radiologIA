'use client';

import { useState } from 'react';
import Image from 'next/image';
import Link from 'next/link';
import { AlertCircle, FileUp, Loader2, Microscope } from 'lucide-react';
import { triageMultipleXrays } from '@/ai/flows/triage-multiple-xrays';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Badge } from '@/components/ui/badge';
import { PlaceHolderImages } from '@/lib/placeholder-images';
import type { AnalysisResult, XRay } from '@/lib/types';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { useToast } from '@/hooks/use-toast';
import { SidebarTrigger } from '@/components/ui/sidebar';

const readFileAsDataURL = (file: File): Promise<string> => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
};

const PriorityBadge = ({ priority }: { priority: 'low' | 'medium' | 'high' }) => {
  const variant = {
    low: 'default',
    medium: 'default',
    high: 'destructive',
  } as const;
  
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

  return <Badge variant={variant[priority]} className={`${className[priority]} text-white`}>{priorityLabels[priority]}</Badge>;
};

export default function TriageClient() {
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResults, setAnalysisResults] = useState<AnalysisResult[]>([]);
  const [error, setError] = useState<string | null>(null);
  const { toast } = useToast();

  const handleFileChange = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (!files || files.length === 0) return;

    setIsAnalyzing(true);
    setError(null);
    setAnalysisResults([]);

    try {
      const xrayInputs = await Promise.all(
        Array.from(files).map(async (file, index) => {
          const dataUri = await readFileAsDataURL(file);
          const placeholder = PlaceHolderImages[index % PlaceHolderImages.length];
          return {
            xrayDataUri: dataUri,
            xrayDetails: {
              id: placeholder.id,
              url: placeholder.imageUrl,
              name: file.name,
              description: placeholder.description,
              dataUri: dataUri,
            } as XRay,
          };
        })
      );

      const results = await triageMultipleXrays({
        xrays: xrayInputs.map(input => ({ xrayDataUri: input.xrayDataUri })),
      });
      
      const combinedResults: AnalysisResult[] = results.map((result, index) => ({
        ...xrayInputs[index].xrayDetails,
        ...result,
      }));

      setAnalysisResults(combinedResults);
    } catch (err) {
      console.error('Análise falhou:', err);
      setError('Ocorreu um erro durante a análise. Por favor, tente novamente.');
      toast({
        variant: "destructive",
        title: "Análise Falhou",
        description: "Não foi possível analisar as imagens de raio-X. Verifique os formatos dos arquivos e tente novamente.",
      });
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="flex-1 space-y-4 p-4 md:p-8 pt-6">
       <div className="flex items-center justify-between space-y-2">
         <div className="flex items-center gap-2">
            <SidebarTrigger className="md:hidden"/>
            <h2 className="text-3xl font-bold tracking-tight font-headline">Análise de Raio-X em Grupo</h2>
          </div>
      </div>
      <Card>
        <CardHeader>
          <CardTitle>Triagem de Raios-X</CardTitle>
          <CardDescription>Carregue múltiplas imagens de raio-X para triagem e priorização automatizadas.</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex flex-col items-center justify-center border-2 border-dashed border-muted-foreground/30 rounded-lg p-12 text-center space-y-4">
            <FileUp className="h-12 w-12 text-muted-foreground" />
            <p className="text-muted-foreground">Arraste e solte as imagens aqui, ou clique para procurar.</p>
            <Button asChild variant="outline">
              <label htmlFor="file-upload" className="cursor-pointer">
                Selecionar Arquivos
              </label>
            </Button>
            <Input id="file-upload" type="file" multiple onChange={handleFileChange} className="hidden" disabled={isAnalyzing} />
          </div>
        </CardContent>
      </Card>

      {isAnalyzing && (
        <div className="flex items-center justify-center p-8 space-x-2">
          <Loader2 className="h-8 w-8 animate-spin text-primary" />
          <p className="text-lg">Analisando imagens, por favor aguarde...</p>
        </div>
      )}

      {error && (
         <Alert variant="destructive">
            <AlertCircle className="h-4 w-4" />
            <AlertTitle>Erro</AlertTitle>
            <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {analysisResults.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Resultados da Análise</CardTitle>
            <CardDescription>Revise os achados e prioridades gerados pela IA.</CardDescription>
          </CardHeader>
          <CardContent>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead className="w-[100px]">Prévia</TableHead>
                  <TableHead>Nome do Arquivo</TableHead>
                  <TableHead>Principais Achados</TableHead>
                  <TableHead className="w-[100px]">Prioridade</TableHead>
                  <TableHead className="text-right w-[150px]">Ações</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {analysisResults.map((result, index) => (
                  <TableRow key={index}>
                    <TableCell>
                      <Image
                        alt={`Prévia do raio-X para ${result.name}`}
                        className="aspect-square rounded-md object-cover"
                        height="64"
                        src={result.dataUri || result.url}
                        width="64"
                      />
                    </TableCell>
                    <TableCell className="font-medium">{result.name}</TableCell>
                    <TableCell>{result.keyFindings}</TableCell>
                    <TableCell>
                      <PriorityBadge priority={result.priority} />
                    </TableCell>
                    <TableCell className="text-right">
                       <Link href={`/analysis/${result.id}?data=${encodeURIComponent(JSON.stringify(result))}`}>
                         <Button variant="outline" size="sm">
                            <Microscope className="mr-2 h-4 w-4" />
                            Analisar
                         </Button>
                       </Link>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
