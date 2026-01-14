type CreditFooterProps = {
  className?: string;
};

const baseClassName = 'text-center text-[10px] text-brand-blue/60 dark:text-white/60';

export default function CreditFooter({ className }: CreditFooterProps) {
  const classNameValue = className ? `${baseClassName} ${className}` : baseClassName;
  return <div className={classNameValue}>Desenvolvido por Sérgio H. Benfenatti Botelho 🇧🇷</div>;
}
