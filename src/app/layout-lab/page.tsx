'use client';

import { useEffect, useMemo, useState } from 'react';
import CreditFooter from '@/components/CreditFooter';

type Rect = {
  x: number;
  y: number;
  w: number;
  h: number;
};

type Handle = 'move' | 'tl' | 'tr' | 'bl' | 'br';

type DragState = {
  handle: Handle;
  startX: number;
  startY: number;
  startRect: Rect;
};

const HANDLE_SIZE = 10;
const MIN_SIZE = 40;

function clamp(value: number, min: number, max: number) {
  return Math.max(min, Math.min(max, value));
}

function ResizableBox({
  rect,
  label,
  onChange,
  children,
}: {
  rect: Rect;
  label: string;
  onChange: (next: Rect) => void;
  children: React.ReactNode;
}) {
  const [dragState, setDragState] = useState<DragState | null>(null);

  useEffect(() => {
    if (!dragState) {
      return;
    }

    const onMove = (event: PointerEvent) => {
      const dx = event.clientX - dragState.startX;
      const dy = event.clientY - dragState.startY;
      const start = dragState.startRect;
      let next = { ...start };

      switch (dragState.handle) {
        case 'move':
          next.x = start.x + dx;
          next.y = start.y + dy;
          break;
        case 'br':
          next.w = Math.max(MIN_SIZE, start.w + dx);
          next.h = Math.max(MIN_SIZE, start.h + dy);
          break;
        case 'bl':
          next.w = Math.max(MIN_SIZE, start.w - dx);
          next.h = Math.max(MIN_SIZE, start.h + dy);
          next.x = start.x + dx;
          break;
        case 'tr':
          next.w = Math.max(MIN_SIZE, start.w + dx);
          next.h = Math.max(MIN_SIZE, start.h - dy);
          next.y = start.y + dy;
          break;
        case 'tl':
          next.w = Math.max(MIN_SIZE, start.w - dx);
          next.h = Math.max(MIN_SIZE, start.h - dy);
          next.x = start.x + dx;
          next.y = start.y + dy;
          break;
      }

      onChange(next);
    };

    const onUp = () => setDragState(null);

    window.addEventListener('pointermove', onMove);
    window.addEventListener('pointerup', onUp);

    return () => {
      window.removeEventListener('pointermove', onMove);
      window.removeEventListener('pointerup', onUp);
    };
  }, [dragState, onChange]);

  const onPointerDown = (handle: Handle) => (event: React.PointerEvent) => {
    event.preventDefault();
    setDragState({
      handle,
      startX: event.clientX,
      startY: event.clientY,
      startRect: rect,
    });
  };

  return (
    <div
      className="absolute border border-dashed border-brand-blue/40 dark:border-white/30 select-none"
      style={{
        left: rect.x,
        top: rect.y,
        width: rect.w,
        height: rect.h,
        touchAction: 'none',
      }}
    >
      <div
        className="absolute inset-0 cursor-move z-10"
        onPointerDown={onPointerDown('move')}
      />
      <div className="absolute inset-0 pointer-events-none">
        {children}
      </div>
      <div className="absolute -top-6 left-0 text-xs font-semibold text-brand-blue/80 dark:text-white/80">
        {label} · x:{Math.round(rect.x)} y:{Math.round(rect.y)} w:{Math.round(rect.w)} h:{Math.round(rect.h)}
      </div>
      {(['tl', 'tr', 'bl', 'br'] as Handle[]).map((handle) => {
        const positionStyles: Record<Handle, React.CSSProperties> = {
          move: {},
          tl: { left: -HANDLE_SIZE / 2, top: -HANDLE_SIZE / 2, cursor: 'nwse-resize' },
          tr: { right: -HANDLE_SIZE / 2, top: -HANDLE_SIZE / 2, cursor: 'nesw-resize' },
          bl: { left: -HANDLE_SIZE / 2, bottom: -HANDLE_SIZE / 2, cursor: 'nesw-resize' },
          br: { right: -HANDLE_SIZE / 2, bottom: -HANDLE_SIZE / 2, cursor: 'nwse-resize' },
        };
        if (handle === 'move') {
          return null;
        }
        return (
          <div
            key={handle}
            className="absolute rounded-sm bg-brand-blue/80 dark:bg-white/80 shadow-sm z-20"
            style={{
              width: HANDLE_SIZE,
              height: HANDLE_SIZE,
              ...positionStyles[handle],
            }}
            onPointerDown={onPointerDown(handle)}
          />
        );
      })}
    </div>
  );
}

export default function LayoutLabPage() {
  const [logoRect, setLogoRect] = useState<Rect>({ x: 220, y: -80, w: 520, h: 520 });
  const [wordmarkRect, setWordmarkRect] = useState<Rect>({ x: 780, y: 160, w: 220, h: 80 });
  const [cardLeftRect, setCardLeftRect] = useState<Rect>({ x: 140, y: 620, w: 420, h: 220 });
  const [cardRightRect, setCardRightRect] = useState<Rect>({ x: 620, y: 620, w: 420, h: 220 });

  const layout = useMemo(
    () => ({
      logo: logoRect,
      wordmark: wordmarkRect,
      cardTriagem: cardLeftRect,
      cardAuxiliar: cardRightRect,
    }),
    [logoRect, wordmarkRect, cardLeftRect, cardRightRect]
  );

  const copyLayout = async () => {
    const text = JSON.stringify(layout, null, 2);
    await navigator.clipboard.writeText(text);
  };

  return (
    <div className="min-h-screen bg-background text-gray-900 dark:text-gray-100 relative overflow-hidden">
      <div className="absolute inset-0 w-full h-full overflow-hidden -z-10 flag-stage">
        <div className="absolute inset-0 bg-[url('https://grainy-gradients.vercel.app/noise.svg')] opacity-20 brightness-100 contrast-150 mix-blend-overlay"></div>
        <div className="flag-canvas">
          <div className="flag-shape flag-green"></div>
          <div className="flag-shape flag-yellow"></div>
          <div className="flag-shape flag-blue"></div>
          <div className="flag-glow"></div>
        </div>
        <div className="flag-glass"></div>
      </div>

      <div className="fixed left-6 top-6 z-20 max-w-sm rounded-2xl bg-white/80 dark:bg-slate-900/80 backdrop-blur-lg border border-white/30 p-4 shadow-lg">
        <h1 className="text-sm font-semibold text-brand-blue dark:text-white">Layout Lab</h1>
        <p className="text-xs text-brand-blue/70 dark:text-white/70 mt-1">
          Arraste para mover. Use os cantos para redimensionar. Copie o JSON e me envie.
        </p>
        <button
          onClick={copyLayout}
          className="mt-3 rounded-full bg-brand-green text-white text-xs font-semibold px-4 py-2"
        >
          Copiar layout
        </button>
        <pre className="mt-3 text-[10px] leading-4 text-brand-blue/80 dark:text-white/80 max-h-52 overflow-auto whitespace-pre-wrap">
{JSON.stringify(layout, null, 2)}
        </pre>
      </div>

      <ResizableBox rect={logoRect} label="Logo (atomo)" onChange={setLogoRect}>
        <img
          src="/brand/atom.png"
          alt="Logo RadiologIA"
          className="absolute inset-0 w-full h-full object-contain drop-shadow-xl"
          draggable={false}
        />
      </ResizableBox>

      <ResizableBox rect={wordmarkRect} label="Wordmark" onChange={setWordmarkRect}>
        <img
          src="/brand/wordmark-light.png"
          alt="radiologIA"
          className="absolute inset-0 w-full h-full object-contain drop-shadow-sm"
          draggable={false}
        />
      </ResizableBox>

      <ResizableBox rect={cardLeftRect} label="Card Triagem" onChange={setCardLeftRect}>
        <div className="absolute inset-0 rounded-3xl border border-white/30 bg-white/60 dark:bg-slate-900/60 backdrop-blur-md"></div>
      </ResizableBox>

      <ResizableBox rect={cardRightRect} label="Card Auxiliar" onChange={setCardRightRect}>
        <div className="absolute inset-0 rounded-3xl border border-white/30 bg-white/60 dark:bg-slate-900/60 backdrop-blur-md"></div>
      </ResizableBox>
      <CreditFooter className="absolute bottom-4 left-0 right-0" />
    </div>
  );
}
