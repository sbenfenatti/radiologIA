'use client';

export function Logo({ className }: { className?: string }) {
  return (
    <img src="/logo.svg" alt="Logo" className={className} />
  );
}
