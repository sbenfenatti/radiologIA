import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import BackgroundCanvas from './components/BackgroundCanvas';
import Chat from './components/Chat';
import { AvailableModels, ChatMessage, Finding, ModelType } from './types';

const DEMO_PASSWORD = 'rad2025';

const initialBotMessage: ChatMessage = {
  id: 'welcome-bot',
  sender: 'bot',
  text: 'Olá! Sou seu IAssistente. Pode me fazer uma pergunta sobre odontologia ou carregar uma radiografia para análise.',
};

const modelInfoTexts: Record<ModelType, string> = {
  yolo: 'Detecção de objetos',
  unet: 'Segmentação detalhada',
};

function isPointInPolygon(point: { x: number; y: number }, polygon: [number, number][]) {
  const { x, y } = point;
  let isInside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const [xi, yi] = polygon[i];
    const [xj, yj] = polygon[j];
    const intersect = (yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi;
    if (intersect) isInside = !isInside;
  }
  return isInside;
}

function App() {
  const [page, setPage] = useState<'login' | 'welcome' | 'analysis'>('login');
  const [passwordError, setPasswordError] = useState('');
  const [imageSrc, setImageSrc] = useState<string | null>(null);
  const [analysisResults, setAnalysisResults] = useState<Finding[]>([]);
  const [activeFinding, setActiveFinding] = useState<Finding | null>(null);
  const [hoveredFinding, setHoveredFinding] = useState<Finding | null>(null);
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([initialBotMessage]);
  const [currentFile, setCurrentFile] = useState<File | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [selectedModel, setSelectedModel] = useState<ModelType>('yolo');
  const [availableModels, setAvailableModels] = useState<AvailableModels>({ yolo: true, unet: false });

  const imageRef = useRef<HTMLImageElement | null>(null);
  const imageContainerRef = useRef<HTMLDivElement | null>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const spotlightCanvasRef = useRef<HTMLCanvasElement | null>(null);

  const addMessage = useCallback((message: ChatMessage) => {
    setChatMessages((prev) => [...prev, message]);
  }, []);

  const addBotMessage = useCallback(
    (text: string, thinking = false) =>
      addMessage({ id: crypto.randomUUID(), sender: 'bot', text, thinking }),
    [addMessage]
  );

  const addUserMessage = useCallback(
    (text: string) => addMessage({ id: crypto.randomUUID(), sender: 'user', text }),
    [addMessage]
  );

  const clearThinking = useCallback(() => {
    setChatMessages((prev) => prev.filter((msg) => !msg.thinking));
  }, []);

  const adjustCanvases = useCallback(() => {
    const img = imageRef.current;
    const container = imageContainerRef.current;
    if (!img || !container) return;
    const overlay = overlayCanvasRef.current;
    const spotlight = spotlightCanvasRef.current;
    const { naturalWidth, naturalHeight, clientWidth, clientHeight } = img;
    if (!naturalWidth || !naturalHeight) return;
    const top = (container.clientHeight - clientHeight) / 2;
    const left = (container.clientWidth - clientWidth) / 2;
    [overlay, spotlight].forEach((canvas) => {
      if (!canvas) return;
      canvas.style.width = `${clientWidth}px`;
      canvas.style.height = `${clientHeight}px`;
      canvas.style.top = `${top}px`;
      canvas.style.left = `${left}px`;
      canvas.width = clientWidth;
      canvas.height = clientHeight;
    });
  }, []);

  const clearCanvas = (canvas: HTMLCanvasElement | null) => {
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    ctx?.clearRect(0, 0, canvas.width, canvas.height);
  };

  const clearAllCanvases = useCallback(() => {
    clearCanvas(overlayCanvasRef.current);
    clearCanvas(spotlightCanvasRef.current);
    setActiveFinding(null);
    setHoveredFinding(null);
  }, []);

  const drawFinding = useCallback(
    (finding?: Finding | null) => {
      const canvas = overlayCanvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;
      clearCanvas(canvas);
      if (!finding || !finding.segmentation?.length) return;
      const { width, height } = canvas;
      const points = finding.segmentation.map((p) => [(p[0] / 100) * width, (p[1] / 100) * height]);
      ctx.beginPath();
      ctx.moveTo(points[0][0], points[0][1]);
      for (let i = 1; i < points.length; i += 1) {
        ctx.lineTo(points[i][0], points[i][1]);
      }
      ctx.closePath();
      ctx.strokeStyle = '#fb923c';
      ctx.lineWidth = 2;
      ctx.fillStyle = 'rgba(251, 146, 60, 0.4)';
      ctx.stroke();
      ctx.fill();
    },
    []
  );

  const drawSpotlight = useCallback((finding?: Finding | null) => {
    const canvas = spotlightCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    clearCanvas(canvas);
    if (!finding) return;
    const { width, height } = canvas;
    const points = finding.segmentation.map((p) => [(p[0] / 100) * width, (p[1] / 100) * height]);
    ctx.fillStyle = 'rgba(0, 0, 0, 0.6)';
    ctx.fillRect(0, 0, width, height);
    ctx.globalCompositeOperation = 'destination-out';
    ctx.beginPath();
    ctx.moveTo(points[0][0], points[0][1]);
    for (let i = 1; i < points.length; i += 1) {
      ctx.lineTo(points[i][0], points[i][1]);
    }
    ctx.closePath();
    ctx.fillStyle = 'white';
    ctx.fill();
    ctx.globalCompositeOperation = 'source-over';
  }, []);

  const handleLogin = (password: string) => {
    if (password === DEMO_PASSWORD) {
      setPage('welcome');
      setPasswordError('');
    } else {
      setPasswordError('Senha incorreta.');
    }
  };

  const fetchAvailableModels = useCallback(async () => {
    try {
      const response = await fetch('/models/available');
      if (response.ok) {
        const models = (await response.json()) as AvailableModels;
        setAvailableModels(models);
        if (!models[selectedModel] && models.default) {
          setSelectedModel(models.default);
        }
      }
    } catch (error) {
      // eslint-disable-next-line no-console
      console.error('Erro ao verificar modelos disponíveis:', error);
    }
  }, [selectedModel]);

  const handleFile = useCallback(
    (file: File) => {
      setCurrentFile(file);
      const reader = new FileReader();
      reader.onload = (event) => {
        setImageSrc(event.target?.result as string);
        setPage('analysis');
        clearAllCanvases();
        fetchAvailableModels();
        setTimeout(() => setIsAnalyzing(false), 0);
        setTimeout(() => setTimeout(runAnalysis, 0), 0);
      };
      reader.readAsDataURL(file);
    },
    [clearAllCanvases, fetchAvailableModels]
  );

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      handleFile(file);
      event.target.value = '';
    }
  };

  const handleUserMessage = async (text: string) => {
    addUserMessage(text);
    addBotMessage('', true);
    const prompt = analysisResults.length
      ? `Você é a RadiologIA, uma IA assistente para dentistas. Seja profissional, conciso e responda sempre em português do Brasil. Com base nos achados radiográficos abaixo, responda à pergunta do usuário.\n\nAchados detectados:\n${analysisResults
          .map((f) => `- ${f.label} (Confiança: ${(f.confidence * 100).toFixed(0)}%)`)
          .join('\n')}\n\nPergunta: "${text}"`
      : `Você é a RadiologIA, uma IA assistente para dentistas. Seja profissional, conciso e responda sempre em português do Brasil. Responda à seguinte pergunta geral de odontologia: "${text}"`;

    try {
      const response = await fetch('/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ history: [{ role: 'user', parts: [{ text: prompt }] }] }),
      });
      clearThinking();
      if (!response.ok) {
        const errorBody = await response.json();
        throw new Error(errorBody.error || `Erro do servidor: ${response.statusText}`);
      }
      const result = await response.json();
      const botResponse = result.candidates?.[0]?.content?.parts?.[0]?.text?.replace(/\*/g, '');
      if (botResponse) {
        addBotMessage(botResponse);
      } else {
        throw new Error('Formato de resposta da IA inválido recebido do backend.');
      }
    } catch (error) {
      clearThinking();
      // eslint-disable-next-line no-console
      console.error('Erro ao chamar o backend:', error);
      addBotMessage(`<b>Falha na Conexão:</b> Não foi possível processar sua solicitação. (${(error as Error).message})`);
    }
  };

  const runAnalysis = useCallback(async () => {
    if (!currentFile || isAnalyzing) return;
    if (!availableModels[selectedModel]) {
      addBotMessage(`❌ Modelo ${selectedModel.toUpperCase()} não está disponível. Selecione outro modelo.`);
      return;
    }
    setIsAnalyzing(true);
    clearAllCanvases();
    const formData = new FormData();
    formData.append('file', currentFile);
    formData.append('model_type', selectedModel);
    try {
      const response = await fetch('/analyze', { method: 'POST', body: formData });
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || `Erro HTTP! status: ${response.status}`);
      }
      const data = await response.json();
      setAnalysisResults(data.findings || []);
      const count = (data.findings || []).length;
      const message = count
        ? `Análise completa com ${data.model_used}. Identifiquei ${count} achados. Clique para visualizar ou me faça uma pergunta.`
        : `Análise completa com ${data.model_used}. Nenhum achado detectado.`;
      addBotMessage(
        `<div class="glass-effect rounded-lg p-3 w-full"><p class="mb-3">${message}</p><div class="space-y-2">${(data.findings || [])
          .map(
            (finding: Finding) =>
              `<div class="finding-item bg-white/20 p-2 rounded-md flex justify-between items-center" data-finding-id="${finding.id}"><span>${finding.label}</span><span class="text-xs text-slate-600">${(finding.confidence * 100).toFixed(0)}%</span></div>`
          )
          .join('')}</div></div>`
      );
    } catch (error) {
      // eslint-disable-next-line no-console
      console.error('Erro ao analisar imagem:', error);
      addBotMessage(`<b>Falha na Conexão:</b> ${(error as Error).message}`);
    } finally {
      setIsAnalyzing(false);
    }
  }, [addBotMessage, availableModels, clearAllCanvases, currentFile, isAnalyzing, selectedModel]);

  const handleFindingHover = useCallback(
    (finding: Finding | null) => {
      setHoveredFinding(finding);
      drawSpotlight(finding || undefined);
    },
    [drawSpotlight]
  );

  useEffect(() => {
    if (!imageSrc) return;
    const img = imageRef.current;
    if (!img) return;
    const onLoad = () => {
      adjustCanvases();
      if (activeFinding) drawFinding(activeFinding);
    };
    img.addEventListener('load', onLoad);
    return () => img.removeEventListener('load', onLoad);
  }, [imageSrc, adjustCanvases, activeFinding, drawFinding]);

  useEffect(() => {
    const handleResize = () => adjustCanvases();
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [adjustCanvases]);

  useEffect(() => {
    if (!imageContainerRef.current || analysisResults.length === 0) return;
    const handleMove = (event: MouseEvent) => {
      const canvas = spotlightCanvasRef.current;
      if (!canvas) return;
      const rect = canvas.getBoundingClientRect();
      const mouseX = event.clientX - rect.left;
      const mouseY = event.clientY - rect.top;
      let hovered: Finding | null = null;
      for (let i = analysisResults.length - 1; i >= 0; i -= 1) {
        const finding = analysisResults[i];
        const { width, height } = canvas;
        const polygon = finding.segmentation.map((p) => [(p[0] / 100) * width, (p[1] / 100) * height]) as [number, number][];
        if (isPointInPolygon({ x: mouseX, y: mouseY }, polygon)) {
          hovered = finding;
          break;
        }
      }
      if (hoveredFinding?.id !== hovered?.id) {
        handleFindingHover(hovered);
      }
    };
    const container = imageContainerRef.current;
    container.addEventListener('mousemove', handleMove);
    container.addEventListener('mouseleave', () => handleFindingHover(null));
    return () => {
      container.removeEventListener('mousemove', handleMove);
      container.removeEventListener('mouseleave', () => handleFindingHover(null));
    };
  }, [analysisResults, handleFindingHover, hoveredFinding]);

  const findingsList = useMemo(
    () =>
      analysisResults.map((finding) => (
        <div
          key={finding.id}
          className="finding-item bg-white/20 p-2 rounded-md flex justify-between items-center"
          onClick={() => {
            setActiveFinding(finding);
            drawFinding(finding);
          }}
          onMouseEnter={() => handleFindingHover(finding)}
          onMouseLeave={() => handleFindingHover(null)}
        >
          <span>{finding.label}</span>
          <span className="text-xs text-slate-600">{(finding.confidence * 100).toFixed(0)}%</span>
        </div>
      )),
    [analysisResults, drawFinding, handleFindingHover]
  );

  return (
    <div className="text-slate-800">
      <BackgroundCanvas />
      {page === 'login' && (
        <div id="login-page" className="flex flex-col h-screen justify-center items-center p-4 relative">
          <div className="logo" />
          <div className="w-full max-w-sm flex flex-col items-center glass-effect rounded-2xl p-8">
            <h2 className="text-2xl font-bold mb-6">Acesso ao Protótipo</h2>
            <div className="w-full mb-4">
              <label htmlFor="password-input" className="block text-sm font-medium text-slate-600 mb-2">
                Senha de Acesso
              </label>
              <input
                type="password"
                id="password-input"
                className="w-full bg-slate-800/20 border border-slate-600/30 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-sky-500 text-slate-800 placeholder:text-slate-500"
                onKeyDown={(e) => {
                  if (e.key === 'Enter') handleLogin((e.target as HTMLInputElement).value);
                }}
              />
              <p id="error-message" className="text-red-500 text-sm mt-2 h-4">
                {passwordError}
              </p>
            </div>
            <button
              id="login-button"
              className="w-full glass-button py-3 px-6 rounded-xl text-lg"
              onClick={() => handleLogin((document.getElementById('password-input') as HTMLInputElement).value)}
            >
              Entrar
            </button>
          </div>
          <footer className="footer">Desenvolvido por: Sérgio H. Benfenatti Botelho - DDS, LLM trainer - 🇧🇷</footer>
        </div>
      )}

      {page === 'welcome' && (
        <div id="welcome-page" className="flex flex-col h-screen justify-center items-center p-4 relative">
          <div className="logo" />
          <div id="welcome-chat-container" className="w-full max-w-4xl h-1/2 flex flex-col glass-effect rounded-2xl">
            <Chat messages={chatMessages} onSend={handleUserMessage} />
          </div>
          <label htmlFor="image-upload-welcome" className="glass-button cursor-pointer py-3 px-6 rounded-xl mt-8 text-lg">
            Analisar Radiografia
          </label>
          <input type="file" id="image-upload-welcome" className="hidden" accept="image/*" onChange={handleFileUpload} />
          <footer className="footer">Desenvolvido por: Sérgio H. Benfenatti Botelho - DDS, LLM trainer - 🇧🇷</footer>
        </div>
      )}

      {page === 'analysis' && (
        <div id="analysis-page" className="h-screen flex-col relative flex">
          <header className="glass-effect m-2.5 rounded-2xl flex justify-between items-center z-10 p-4">
            <div className="flex items-center space-x-3">
              <div className="logo small" />
            </div>
            <div className="flex items-center space-x-4">
              <div className="flex flex-col items-center">
                <div className="model-selector" id="model-selector">
                  {(['yolo', 'unet'] as ModelType[]).map((model) => (
                    <div
                      key={model}
                      className={`model-option ${selectedModel === model ? 'active' : ''} ${!availableModels[model] ? 'inactive' : ''}`}
                      onClick={() => availableModels[model] && setSelectedModel(model)}
                    >
                      {model.toUpperCase()}
                    </div>
                  ))}
                </div>
                <div className="model-info" id="model-info">
                  {modelInfoTexts[selectedModel] || 'Modelo selecionado'}
                </div>
              </div>
              <label htmlFor="image-upload-analysis" className="glass-button cursor-pointer py-2 px-5 rounded-xl">
                Carregar Nova
              </label>
              <input type="file" id="image-upload-analysis" className="hidden" accept="image/*" onChange={handleFileUpload} />
              <button
                id="analyze-btn"
                className="glass-button py-2 px-5 rounded-xl disabled:opacity-50 disabled:cursor-not-allowed"
                disabled={!currentFile || isAnalyzing}
                onClick={runAnalysis}
              >
                Analisar Imagem
              </button>
            </div>
          </header>

          <main className="flex-grow flex flex-col md:flex-row overflow-hidden p-2.5 pt-0 gap-2.5">
            <div
              id="viewer-panel"
              className="w-full md:w-2/3 h-1/2 md:h-full glass-effect rounded-2xl p-4 flex flex-col items-center justify-center relative overflow-hidden"
            >
              <div
                id="image-container"
                ref={imageContainerRef}
                className={`relative w-full h-full ${imageSrc ? 'flex' : 'hidden'} justify-center items-center cursor-crosshair`}
              >
                <img ref={imageRef} id="radiograph-img" src={imageSrc ?? ''} alt="Radiografia a ser analisada" className="rounded-md shadow-lg max-w-full max-h-full object-contain" />
                <canvas ref={overlayCanvasRef} id="overlay-canvas" className="absolute pointer-events-none" />
                <canvas ref={spotlightCanvasRef} id="spotlight-canvas" className="absolute pointer-events-none" />
              </div>
              {!imageSrc && (
                <div id="viewer-placeholder" className="text-center text-slate-500 flex flex-col justify-center items-center">
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    className="mx-auto h-16 w-16 opacity-50"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                    strokeWidth="1.5"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      d="M2.25 15.75l5.159-5.159a2.25 2.25 0 013.182 0l5.159 5.159m-1.5-1.5l1.409-1.409a2.25 2.25 0 013.182 0l2.909 2.909m-18 3.75h16.5a1.5 1.5 0 001.5-1.5V6a1.5 1.5 0 00-1.5-1.5H3.75A1.5 1.5 0 002.25 6v12a1.5 1.5 0 001.5 1.5zm10.5-11.25h.008v.008h-.008V8.25zm.158 0L10.5 9.75M10.5 12a1.5 1.5 0 11-3 0 1.5 1.5 0 013 0z"
                    />
                  </svg>
                  <p className="mt-4 text-lg">Carregue uma radiografia para começar.</p>
                </div>
              )}
              {isAnalyzing && (
                <div id="loader" className="absolute inset-0 glass-effect flex-col items-center justify-center z-20 flex">
                  <div className="loader ease-linear rounded-full border-8 border-t-8 border-gray-400 h-24 w-24 mb-4" />
                  <p className="text-xl font-semibold">Analisando com IA...</p>
                  <p className="text-sm text-slate-600 mt-2" id="loader-model-info">
                    Modelo: {selectedModel.toUpperCase()}
                  </p>
                </div>
              )}
            </div>

            <div
              id="analysis-chat-container"
              className="w-full md:w-1/3 h-1/2 md:h-full flex flex-col glass-effect rounded-2xl overflow-hidden"
            >
              <Chat messages={chatMessages} onSend={handleUserMessage} disabled={isAnalyzing} />
              {analysisResults.length > 0 && (
                <div className="p-4 border-t border-black/10 space-y-2">
                  <p className="text-sm text-slate-700">
                    {analysisResults.length > 0
                      ? `Análise completa com ${selectedModel.toUpperCase()}. Identifiquei ${analysisResults.length} achados.`
                      : 'Nenhum achado detectado.'}
                  </p>
                  <div className="space-y-2">{findingsList}</div>
                </div>
              )}
            </div>
          </main>

          <footer className="footer">Desenvolvido por: Sérgio H. Benfenatti Botelho - DDS, LLM trainer - 🇧🇷</footer>
        </div>
      )}
    </div>
  );
}

export default App;
