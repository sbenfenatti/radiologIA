import { useEffect, useRef } from 'react';

interface Shape {
  type: 'circle' | 'square' | 'triangle';
  size: number;
  x: number;
  y: number;
  vx: number;
  vy: number;
  color: string;
}

const shapeTypes: Shape['type'][] = ['circle', 'square', 'triangle'];
const colors = ['rgba(179, 209, 255, 0.5)', 'rgba(194, 230, 208, 0.5)', 'rgba(255, 249, 196, 0.5)'];
const shapeCount = 12;

const BackgroundCanvas = () => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const shapesRef = useRef<Shape[]>([]);
  const animationRef = useRef<number>();

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const resize = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };
    resize();

    const createShape = (): Shape => ({
      type: shapeTypes[Math.floor(Math.random() * shapeTypes.length)],
      size: Math.random() * 80 + 60,
      x: Math.random() * canvas.width,
      y: Math.random() * canvas.height,
      vx: (Math.random() - 0.5) * 0.5,
      vy: (Math.random() - 0.5) * 0.5,
      color: colors[Math.floor(Math.random() * colors.length)],
    });

    const initShapes = () => {
      shapesRef.current = Array.from({ length: shapeCount }, createShape);
    };

    const draw = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      shapesRef.current.forEach((shape) => {
        if (shape.x < 0 || shape.x > canvas.width) shape.vx *= -1;
        if (shape.y < 0 || shape.y > canvas.height) shape.vy *= -1;
        shape.x += shape.vx;
        shape.y += shape.vy;

        ctx.fillStyle = shape.color;
        ctx.beginPath();
        if (shape.type === 'circle') {
          ctx.arc(shape.x, shape.y, shape.size / 2, 0, Math.PI * 2);
        } else if (shape.type === 'square') {
          ctx.rect(shape.x - shape.size / 2, shape.y - shape.size / 2, shape.size, shape.size);
        } else {
          ctx.moveTo(shape.x, shape.y - shape.size / 2);
          ctx.lineTo(shape.x - shape.size / 2, shape.y + shape.size / 2);
          ctx.lineTo(shape.x + shape.size / 2, shape.y + shape.size / 2);
          ctx.closePath();
        }
        ctx.fill();
      });
      animationRef.current = requestAnimationFrame(draw);
    };

    initShapes();
    draw();
    window.addEventListener('resize', resize);

    return () => {
      window.removeEventListener('resize', resize);
      if (animationRef.current) cancelAnimationFrame(animationRef.current);
    };
  }, []);

  return <canvas ref={canvasRef} id="background-canvas" />;
};

export default BackgroundCanvas;
