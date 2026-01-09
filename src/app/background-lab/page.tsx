'use client';

import { useEffect, useMemo, useState } from 'react';

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
const STAGE_WIDTH = 1600;
const STAGE_HEIGHT = 900;

const defaultLayout = {
  green: { x: 250.26500277534126, y: 99.87500297650695, w: 1088.9235358019732, h: 694.7239522156306 },
  yellow: { x: 266.43, y: 109.29, w: 1067.14, h: 681.43 },
  blue: { x: 575, y: 225, w: 450, h: 450 },
};

const defaultEffects = {
  greenOpacity: 0.3,
  yellowOpacity: 0.3,
  blueOpacity: 0.3,
  greenBlur: 36,
  yellowBlur: 28,
  blueBlur: 16,
  glassOpacity: 0.12,
  glassBlur: 18,
};

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

export default function BackgroundLabPage() {
  const [greenRect, setGreenRect] = useState<Rect>(defaultLayout.green);
  const [yellowRect, setYellowRect] = useState<Rect>(defaultLayout.yellow);
  const [blueRect, setBlueRect] = useState<Rect>(defaultLayout.blue);
  const [greenOpacity, setGreenOpacity] = useState(defaultEffects.greenOpacity);
  const [yellowOpacity, setYellowOpacity] = useState(defaultEffects.yellowOpacity);
  const [blueOpacity, setBlueOpacity] = useState(defaultEffects.blueOpacity);
  const [greenBlur, setGreenBlur] = useState(defaultEffects.greenBlur);
  const [yellowBlur, setYellowBlur] = useState(defaultEffects.yellowBlur);
  const [blueBlur, setBlueBlur] = useState(defaultEffects.blueBlur);
  const [glassOpacity, setGlassOpacity] = useState(defaultEffects.glassOpacity);
  const [glassBlur, setGlassBlur] = useState(defaultEffects.glassBlur);

  const layout = useMemo(
    () => ({
      green: greenRect,
      yellow: yellowRect,
      blue: blueRect,
    }),
    [greenRect, yellowRect, blueRect]
  );

  const effects = useMemo(
    () => ({
      greenOpacity,
      yellowOpacity,
      blueOpacity,
      greenBlur,
      yellowBlur,
      blueBlur,
      glassOpacity,
      glassBlur,
    }),
    [
      greenOpacity,
      yellowOpacity,
      blueOpacity,
      greenBlur,
      yellowBlur,
      blueBlur,
      glassOpacity,
      glassBlur,
    ]
  );

  const copyLayout = async () => {
    const text = JSON.stringify({ layout, effects }, null, 2);
    await navigator.clipboard.writeText(text);
  };

  const resetLayout = () => {
    setGreenRect(defaultLayout.green);
    setYellowRect(defaultLayout.yellow);
    setBlueRect(defaultLayout.blue);
    setGreenOpacity(defaultEffects.greenOpacity);
    setYellowOpacity(defaultEffects.yellowOpacity);
    setBlueOpacity(defaultEffects.blueOpacity);
    setGreenBlur(defaultEffects.greenBlur);
    setYellowBlur(defaultEffects.yellowBlur);
    setBlueBlur(defaultEffects.blueBlur);
    setGlassOpacity(defaultEffects.glassOpacity);
    setGlassBlur(defaultEffects.glassBlur);
  };

  return (
    <div className="min-h-screen bg-background text-gray-900 dark:text-gray-100 relative overflow-hidden">
      <div className="fixed left-6 top-6 z-20 max-w-sm rounded-2xl bg-white/80 dark:bg-slate-900/80 backdrop-blur-lg border border-white/30 p-4 shadow-lg">
        <h1 className="text-sm font-semibold text-brand-blue dark:text-white">Background Lab</h1>
        <p className="text-xs text-brand-blue/70 dark:text-white/70 mt-1">
          Arraste e redimensione as formas. A ordem final é: verde (base), amarelo (meio), azul (topo).
        </p>
        <div className="mt-4 space-y-3 text-[11px] text-brand-blue/70 dark:text-white/70">
          <div>
            <label className="flex items-center justify-between">
              <span>Opacidade verde</span>
              <span>{greenOpacity.toFixed(2)}</span>
            </label>
            <input
              type="range"
              min="0.05"
              max="0.6"
              step="0.01"
              value={greenOpacity}
              onChange={(event) => setGreenOpacity(Number(event.target.value))}
              className="w-full"
            />
          </div>
          <div>
            <label className="flex items-center justify-between">
              <span>Opacidade amarelo</span>
              <span>{yellowOpacity.toFixed(2)}</span>
            </label>
            <input
              type="range"
              min="0.05"
              max="0.6"
              step="0.01"
              value={yellowOpacity}
              onChange={(event) => setYellowOpacity(Number(event.target.value))}
              className="w-full"
            />
          </div>
          <div>
            <label className="flex items-center justify-between">
              <span>Opacidade azul</span>
              <span>{blueOpacity.toFixed(2)}</span>
            </label>
            <input
              type="range"
              min="0.05"
              max="0.6"
              step="0.01"
              value={blueOpacity}
              onChange={(event) => setBlueOpacity(Number(event.target.value))}
              className="w-full"
            />
          </div>
          <div>
            <label className="flex items-center justify-between">
              <span>Blur verde</span>
              <span>{Math.round(greenBlur)}px</span>
            </label>
            <input
              type="range"
              min="0"
              max="60"
              step="1"
              value={greenBlur}
              onChange={(event) => setGreenBlur(Number(event.target.value))}
              className="w-full"
            />
          </div>
          <div>
            <label className="flex items-center justify-between">
              <span>Blur amarelo</span>
              <span>{Math.round(yellowBlur)}px</span>
            </label>
            <input
              type="range"
              min="0"
              max="60"
              step="1"
              value={yellowBlur}
              onChange={(event) => setYellowBlur(Number(event.target.value))}
              className="w-full"
            />
          </div>
          <div>
            <label className="flex items-center justify-between">
              <span>Blur azul</span>
              <span>{Math.round(blueBlur)}px</span>
            </label>
            <input
              type="range"
              min="0"
              max="60"
              step="1"
              value={blueBlur}
              onChange={(event) => setBlueBlur(Number(event.target.value))}
              className="w-full"
            />
          </div>
          <div>
            <label className="flex items-center justify-between">
              <span>Glass fundo</span>
              <span>{glassOpacity.toFixed(2)}</span>
            </label>
            <input
              type="range"
              min="0"
              max="0.35"
              step="0.01"
              value={glassOpacity}
              onChange={(event) => setGlassOpacity(Number(event.target.value))}
              className="w-full"
            />
          </div>
          <div>
            <label className="flex items-center justify-between">
              <span>Blur glass</span>
              <span>{Math.round(glassBlur)}px</span>
            </label>
            <input
              type="range"
              min="0"
              max="40"
              step="1"
              value={glassBlur}
              onChange={(event) => setGlassBlur(Number(event.target.value))}
              className="w-full"
            />
          </div>
        </div>
        <div className="mt-3 flex flex-wrap gap-2">
          <button
            onClick={copyLayout}
            className="rounded-full bg-brand-green text-white text-xs font-semibold px-4 py-2"
          >
            Copiar JSON
          </button>
          <button
            onClick={resetLayout}
            className="rounded-full border border-brand-blue/30 text-brand-blue text-xs font-semibold px-4 py-2"
          >
            Resetar
          </button>
        </div>
        <pre className="mt-3 text-[10px] leading-4 text-brand-blue/80 dark:text-white/80 max-h-52 overflow-auto whitespace-pre-wrap">
{JSON.stringify({ layout, effects }, null, 2)}
        </pre>
      </div>

      <div
        className="relative mx-auto mt-24 border border-dashed border-brand-blue/20 dark:border-white/10 bg-white/60 dark:bg-slate-900/40 shadow-inner"
        style={{ width: STAGE_WIDTH, height: STAGE_HEIGHT }}
      >
        <div
          className="absolute inset-0 pointer-events-none"
          style={{
            background: `rgba(255, 255, 255, ${glassOpacity})`,
            backdropFilter: `blur(${glassBlur}px)`,
          }}
        ></div>
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,rgba(2,39,118,0.08),transparent_60%)]"></div>

      <ResizableBox rect={greenRect} label="Verde (base)" onChange={setGreenRect}>
          <div
            className="absolute inset-0 rounded-[32px] bg-[#009c3b]"
            style={{ opacity: greenOpacity, filter: `blur(${greenBlur}px)` }}
          ></div>
        </ResizableBox>

      <ResizableBox rect={yellowRect} label="Amarelo (losango)" onChange={setYellowRect}>
          <div
            className="absolute inset-0 bg-[#ffdf00]"
            style={{
              opacity: yellowOpacity,
              filter: `blur(${yellowBlur}px)`,
              clipPath: 'polygon(50% 0%, 100% 50%, 50% 100%, 0% 50%)',
            }}
          ></div>
        </ResizableBox>

      <ResizableBox rect={blueRect} label="Azul (topo)" onChange={setBlueRect}>
          <div
            className="absolute inset-0 rounded-full bg-[#002776]"
            style={{ opacity: blueOpacity, filter: `blur(${blueBlur}px)` }}
          ></div>
        </ResizableBox>
      </div>
    </div>
  );
}
