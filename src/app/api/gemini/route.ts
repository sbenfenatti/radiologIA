import { NextResponse } from 'next/server';

type ChatMessage = {
  role: 'user' | 'model' | 'system';
  text: string;
};

type ChatRequest = {
  context?: string;
  history?: ChatMessage[];
};

export const runtime = 'nodejs';

const DEFAULT_MODEL = 'gemini-3-flash-preview';
const MAX_HISTORY = 12;
const MAX_CONTEXT_CHARS = 6000;

const buildSystemPrompt = (context?: string) => {
  const trimmedContext = context?.trim().slice(0, MAX_CONTEXT_CHARS);
  const lines = [
    'Você é a RadiologIA, uma assistente odontológica avançada.',
    'Responda em português (pt-BR), de forma direta e técnica.',
    'Use termos corretos (mesial, distal, radiolúcido, radiopaco).',
    'Evite respostas genéricas; cite achados do contexto quando disponíveis.',
    'Se a pergunta for sobre achados principais, liste até 5 itens objetivos.',
  ];
  if (trimmedContext) {
    lines.push('Achados detectados na radiografia:');
    lines.push(trimmedContext);
  } else {
    lines.push('Nenhum contexto de imagem foi fornecido.');
  }
  return lines.join('\n');
};

export async function POST(request: Request) {
  const apiKey = process.env.GEMINI_API_KEY ?? process.env.GOOGLE_API_KEY;
  if (!apiKey) {
    return NextResponse.json(
      { error: 'GEMINI_API_KEY não configurada.' },
      { status: 500 },
    );
  }

  let payload: ChatRequest | null = null;
  try {
    payload = (await request.json()) as ChatRequest;
  } catch (error) {
    return NextResponse.json(
      { error: 'Corpo da requisicao invalido.' },
      { status: 400 },
    );
  }

  const history = Array.isArray(payload?.history) ? payload?.history : [];
  const trimmedHistory = history
    .filter((message) => message && typeof message.text === 'string')
    .slice(-MAX_HISTORY);

  const contents = trimmedHistory.map((message) => ({
    role: message.role === 'model' ? 'model' : 'user',
    parts: [{ text: message.text }],
  }));

  const model = process.env.GEMINI_MODEL ?? DEFAULT_MODEL;
  const url = `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${apiKey}`;

  try {
    const response = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        contents,
        systemInstruction: {
          role: 'user',
          parts: [{ text: buildSystemPrompt(payload?.context) }],
        },
        generationConfig: {
          temperature: 0.3,
          topP: 0.9,
          maxOutputTokens: 1024,
        },
      }),
    });

    if (!response.ok) {
      const text = await response.text();
      return NextResponse.json(
        { error: text || 'Falha ao chamar o Gemini.' },
        { status: 500 },
      );
    }

    const data = await response.json();
    const candidate = data?.candidates?.[0];
    const parts = candidate?.content?.parts ?? [];
    const reply =
      parts.map((part: { text?: string }) => part.text ?? '').join('').trim() ||
      'Resposta indisponivel.';

    return NextResponse.json({
      response: reply,
      finishReason: candidate?.finishReason ?? null,
    });
  } catch (error) {
    return NextResponse.json(
      { error: 'Falha ao comunicar com o Gemini.' },
      { status: 500 },
    );
  }
}
