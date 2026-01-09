# RadiologIA

MVP do RadiologIA: interface de login e landing inicial para um assistente de diagnóstico odontológico com IA.

## Status
- UI do login finalizada.
- Autenticação em migração para Supabase (OTP planejado).

## Stack
- Next.js 16 (App Router)
- TypeScript
- Tailwind CSS + shadcn/ui
- Supabase (Auth planejado)
- Genkit (IA, futuro)

## Como rodar localmente
1) Instale as dependências:
   ```bash
   npm install
   ```
2) Crie `.env.local`:
   ```bash
   NEXT_PUBLIC_SUPABASE_URL=...
   NEXT_PUBLIC_SUPABASE_ANON_KEY=...
   ```
3) Rode o app:
   ```bash
   npm run dev
   ```

## Estrutura principal
- `src/app/page.tsx`: login
- `src/app/dashboard/page.tsx`: página inicial pós-login
- `src/components/ui/*`: componentes de UI
- `src/ai/*`: fluxos Genkit (planejado)

## Scripts
- `npm run dev`: servidor local
- `npm run build`: build de produção
- `npm run lint`: lint
- `npm run typecheck`: typecheck
- `npm run genkit:dev`: dev server do Genkit

## Deploy (Vercel)
Configure as variáveis de ambiente do Supabase no painel da Vercel:
- `NEXT_PUBLIC_SUPABASE_URL`
- `NEXT_PUBLIC_SUPABASE_ANON_KEY`
