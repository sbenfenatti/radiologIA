import { useMemo } from 'react';
import { ChatMessage } from '../types';

interface ChatProps {
  messages: ChatMessage[];
  onSend: (text: string) => void;
  disabled?: boolean;
}

const Chat = ({ messages, onSend, disabled }: ChatProps) => {
  const list = useMemo(() => messages, [messages]);

  const handleSend = (event: React.FormEvent) => {
    event.preventDefault();
    const formData = new FormData(event.target as HTMLFormElement);
    const text = (formData.get('message') as string)?.trim();
    if (text) {
      onSend(text);
      (event.target as HTMLFormElement).reset();
    }
  };

  return (
    <div className="flex flex-col h-full">
      <div className="p-4 border-b border-black/10 flex items-center justify-between">
        <h2 className="text-lg font-semibold">IAssistente Clínico</h2>
      </div>
      <div className="chat-container flex-grow p-4 overflow-y-auto space-y-4">
        {list.map((message) => (
          <div key={message.id} className={`flex ${message.sender === 'bot' ? 'justify-start' : 'justify-end'}`}>
            <div
              className={
                message.sender === 'bot'
                  ? 'glass-effect rounded-lg p-3 max-w-sm'
                  : 'bg-blue-500 text-white rounded-lg p-3 max-w-sm'
              }
            >
              {message.thinking ? (
                <div className="thinking-bubble">
                  <span className="bounce1" />
                  <span className="bounce2" />
                  <span className="bounce3" />
                </div>
              ) : (
                <p className="text-slate-800 whitespace-pre-wrap" dangerouslySetInnerHTML={{ __html: message.text }} />
              )}
            </div>
          </div>
        ))}
      </div>
      <form className="p-4 border-t border-black/10" onSubmit={handleSend}>
        <div className="flex space-x-3 items-center">
          <input
            name="message"
            type="text"
            className="chat-input flex-grow bg-white/30 border border-slate-900/20 px-4 py-2 rounded-lg focus:outline-none focus:ring-2 focus:ring-sky-500 placeholder:text-slate-500"
            disabled={disabled}
            placeholder="Digite sua mensagem"
          />
          <button className="chat-send glass-button rounded-lg p-2.5" type="submit" disabled={disabled}>
            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="h-5 w-5">
              <path d="M12 5l7 7-7 7" />
              <path d="M5 12h14" />
            </svg>
          </button>
        </div>
      </form>
    </div>
  );
};

export default Chat;
