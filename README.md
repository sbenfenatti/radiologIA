# RadiologIA Frontend (React + Vite)

Esta branch foca apenas no frontend React, convertendo o HTML original em uma SPA moderna. O backend anterior foi removido para simplificar os testes desta branch.

## Pré-requisitos
- Node.js 18+
- npm (ou outro gerenciador compatível)

## Instalação
Como o ambiente de exemplo pode ter restrições de rede, as dependências estão listadas em `package.json`. Em um ambiente com acesso ao npm, rode:

```bash
npm install
```

## Desenvolvimento

```bash
npm run dev
```

A aplicação usa Vite (porta padrão 5173). O estilo reaproveita o Tailwind via CDN e classes customizadas.

## Build

```bash
npm run build
```

## Estrutura
- `src/App.tsx`: página única com login, tela de boas-vindas e análise integradas.
- `src/components/BackgroundCanvas.tsx`: animação do fundo em canvas.
- `src/components/Chat.tsx`: chat reutilizável para as etapas de boas-vindas e análise.
- `src/styles.css`: estilos globais (glassmorphism e classes utilitárias).

## Integração com backend
As chamadas continuam apontando para os endpoints existentes do backend em FastAPI:
- `POST /analyze` para análise de imagem (envia `file` e `model_type`).
- `POST /chat` para respostas contextuais.
- `GET /models/available` para verificar modelos disponíveis.

Configure a URL base via proxy do Vite ou ajuste os caminhos conforme necessário.
