'use client';

import { useState, useRef, useEffect } from 'react';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Send, Sparkles, FileText, Stethoscope, Loader2 } from 'lucide-react';
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar';
import { summarizeXrayFindings } from '@/ai/flows/summarize-xray-findings';
import { suggestPossibleDiagnoses } from '@/ai/flows/suggest-possible-diagnoses';
import type { AnalysisResult, Message } from '@/lib/types';
import { patientHistory } from '@/lib/data';
import { useToast } from '@/hooks/use-toast';
import { v4 as uuidv4 } from 'uuid'; // requires `npm install uuid @types/uuid`

const AiMessage = ({ children, isLoading }: { children: React.ReactNode, isLoading?: boolean }) => (
  <div className="flex items-start gap-3">
    <Avatar className="h-8 w-8 border">
      <AvatarFallback><Sparkles className="h-4 w-4 text-primary"/></AvatarFallback>
    </Avatar>
    <div className="bg-primary/10 text-primary-foreground rounded-lg p-3 max-w-xs md:max-w-md">
      {isLoading ? <Loader2 className="h-5 w-5 animate-spin" /> : children}
    </div>
  </div>
);

const UserMessage = ({ children }: { children: React.ReactNode }) => (
  <div className="flex items-start gap-3 justify-end">
    <div className="bg-primary text-primary-foreground rounded-lg p-3 max-w-xs md:max-w-md">
      {children}
    </div>
     <Avatar className="h-8 w-8 border">
      <AvatarImage src="https://picsum.photos/seed/doc/100/100" />
      <AvatarFallback>DS</AvatarFallback>
    </Avatar>
  </div>
);


export default function ChatAssistant({ analysisResult }: { analysisResult: AnalysisResult }) {
  const priorityLabels = {
    low: 'baixa',
    medium: 'média',
    high: 'alta'
  };

  const [messages, setMessages] = useState<Message[]>([
    {
      id: uuidv4(),
      role: 'assistant',
      content: `Olá! Sou seu assistente de IA. Analisei o raio-X "${analysisResult.name}". Como posso ajudar? O achado inicial é: "${analysisResult.keyFindings}" com prioridade ${priorityLabels[analysisResult.priority]}.`,
    }
  ]);
  const [input, setInput] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const scrollAreaRef = useRef<HTMLDivElement>(null);
  const { toast } = useToast();

  const handleSendMessage = (content: string, role: 'user' | 'assistant') => {
    const newMessage = { id: uuidv4(), role, content };
    setMessages(prev => [...prev, newMessage]);
    // Simulate AI response for custom questions
    if (role === 'user') {
        const loadingMessage = { id: uuidv4(), role: 'assistant' as const, content: '', isLoading: true };
        setMessages(prev => [...prev, loadingMessage]);
        setTimeout(() => {
            const aiResponse = { id: loadingMessage.id, role: 'assistant' as const, content: "Fui projetado para responder às ações predefinidas para resumir achados e sugerir diagnósticos. O suporte para perguntas personalizadas estará disponível em breve!" };
            setMessages(prev => prev.map(m => m.id === loadingMessage.id ? aiResponse : m));
        }, 1500);
    }
  };

  useEffect(() => {
    if (scrollAreaRef.current) {
      scrollAreaRef.current.scrollTo({
        top: scrollAreaRef.current.scrollHeight,
        behavior: 'smooth',
      });
    }
  }, [messages]);

  const handleSummarize = async () => {
    setIsProcessing(true);
    const userMessageId = uuidv4();
    const assistantMessageId = uuidv4();
    setMessages(prev => [
      ...prev,
      { id: userMessageId, role: 'user', content: 'Resuma os achados.' },
      { id: assistantMessageId, role: 'assistant', content: '', isLoading: true },
    ]);
    
    try {
      if (!analysisResult.dataUri) {
          throw new Error("Os dados da imagem de raio-X não estão disponíveis para análise.");
      }
      const result = await summarizeXrayFindings({
        xrayDataUri: analysisResult.dataUri,
        analysisResults: analysisResult.keyFindings,
      });
      setMessages(prev => prev.map(m => m.id === assistantMessageId ? { ...m, content: result.summary, isLoading: false } : m));
    } catch (e) {
      console.error(e);
      const errorMsg = 'Falha ao gerar o resumo.';
      toast({ variant: 'destructive', title: 'Erro', description: errorMsg });
      setMessages(prev => prev.map(m => m.id === assistantMessageId ? { ...m, content: `Desculpe, ${errorMsg}`, isLoading: false } : m));
    } finally {
      setIsProcessing(false);
    }
  };

  const handleDiagnose = async () => {
    setIsProcessing(true);
    const userMessageId = uuidv4();
    const assistantMessageId = uuidv4();
    setMessages(prev => [
      ...prev,
      { id: userMessageId, role: 'user', content: 'Sugira possíveis diagnósticos.' },
      { id: assistantMessageId, role: 'assistant', content: '', isLoading: true },
    ]);

    try {
      const result = await suggestPossibleDiagnoses({
        xrayAnalysis: analysisResult.keyFindings,
        patientHistory: patientHistory,
      });
      const diagnosesContent = (
        <div>
          <p className="font-semibold mb-2">Diagnósticos Possíveis:</p>
          <ul className="list-disc list-inside">
            {result.possibleDiagnoses.map((d, i) => <li key={i}>{d}</li>)}
          </ul>
          <p className="font-semibold mt-3 mb-1">Raciocínio:</p>
          <p>{result.reasoning}</p>
        </div>
      );
      setMessages(prev => prev.map(m => m.id === assistantMessageId ? { ...m, content: diagnosesContent, isLoading: false } : m));
    } catch (e) {
      console.error(e);
      const errorMsg = 'Falha ao sugerir diagnósticos.';
      toast({ variant: 'destructive', title: 'Erro', description: errorMsg });
      setMessages(prev => prev.map(m => m.id === assistantMessageId ? { ...m, content: `Desculpe, ${errorMsg}`, isLoading: false } : m));
    } finally {
      setIsProcessing(false);
    }
  };
  
  const handleFormSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim()) return;
    handleSendMessage(input, 'user');
    setInput('');
  };

  return (
    <div className="flex flex-col h-full">
      <h3 className="text-xl font-semibold mb-4 font-headline">Assistente de IA</h3>
      <ScrollArea className="flex-1 -mx-4">
        <div ref={scrollAreaRef} className="space-y-6 px-4">
          {messages.map((message) => (
            message.role === 'user' ? (
              <UserMessage key={message.id}>{message.content}</UserMessage>
            ) : (
              <AiMessage key={message.id} isLoading={message.isLoading}>{message.content}</AiMessage>
            )
          ))}
        </div>
      </ScrollArea>
      <div className="mt-4 space-y-4">
        <div className="flex gap-2">
          <Button variant="outline" className="flex-1" onClick={handleSummarize} disabled={isProcessing}>
            <FileText className="mr-2 h-4 w-4" /> Resumir
          </Button>
          <Button variant="outline" className="flex-1" onClick={handleDiagnose} disabled={isProcessing}>
            <Stethoscope className="mr-2 h-4 w-4" /> Diagnosticar
          </Button>
        </div>
        <form onSubmit={handleFormSubmit} className="flex gap-2">
          <Input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Faça uma pergunta de acompanhamento..."
            disabled={isProcessing}
          />
          <Button type="submit" size="icon" disabled={isProcessing || !input.trim()}>
            <Send className="h-4 w-4" />
          </Button>
        </form>
      </div>
    </div>
  );
}
