import React, { useRef, useEffect } from 'react';
import { 
  Send, 
  Scale, 
  User, 
  ShieldCheck, 
  Globe2, 
  Clock, 
  Loader2,
  FileCheck2,
  AlertCircle
} from 'lucide-react';
import type { ChatMessage, CitationItem } from '../types/api';
import { RefusalCard } from './RefusalCard';

interface ChatAreaProps {
  messages: ChatMessage[];
  loading: boolean;
  onSendMessage: (query: string) => void;
  onSelectCitation: (citation: CitationItem) => void;
  selectedCitationId: string | null;
  onSelectExampleQuery: (query: string) => void;
}

export const ChatArea: React.FC<ChatAreaProps> = ({
  messages,
  loading,
  onSendMessage,
  onSelectCitation,
  selectedCitationId,
  onSelectExampleQuery,
}) => {
  const [inputQuery, setInputQuery] = React.useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, loading]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (inputQuery.trim() && !loading) {
      onSendMessage(inputQuery.trim());
      setInputQuery('');
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  // Helper to render text with interactive citation badges
  const renderFormattedText = (text: string, citations: CitationItem[]) => {
    const parts = text.split(/(\[\d+\])/g);
    
    return parts.map((part, idx) => {
      const match = part.match(/\[(\d+)\]/);
      if (match) {
        const citNum = match[1];
        const matchedCit = citations.find(c => c.citation_id === citNum) || citations[parseInt(citNum, 10) - 1];
        const isActive = selectedCitationId === citNum || (matchedCit && matchedCit.citation_id === selectedCitationId);

        return (
          <button
            key={idx}
            type="button"
            onClick={() => {
              if (matchedCit) {
                onSelectCitation(matchedCit);
              }
            }}
            className={`citation-badge ${isActive ? 'active' : ''}`}
            title={matchedCit ? `View Evidence: ${matchedCit.document} (${matchedCit.section || 'General'})` : `Citation [${citNum}]`}
          >
            [{citNum}]
          </button>
        );
      }
      return <span key={idx}>{part}</span>;
    });
  };

  return (
    <div className="flex-1 flex flex-col h-[calc(100vh-4rem)] bg-slate-50/50 overflow-hidden">
      {/* Messages Scroll Area */}
      <div className="flex-1 overflow-y-auto p-4 sm:p-6 space-y-6">
        {messages.length === 0 ? (
          /* Empty Landing State */
          <div className="max-w-2xl mx-auto py-8 text-center space-y-6">
            <div className="inline-flex items-center justify-center p-3 bg-gradient-to-br from-amber-500/20 to-amber-600/10 rounded-2xl border border-amber-500/30 shadow-inner">
              <Scale className="h-10 w-10 text-amber-600" />
            </div>

            <div className="space-y-2">
              <h2 className="text-2xl font-bold tracking-tight text-slate-900 font-['Outfit']">
                IP-SAKTI Sahayak
              </h2>
              <p className="text-xs sm:text-sm text-slate-600 max-w-lg mx-auto leading-relaxed">
                Authoritative AI Assistant for <strong>Indian Patent Law</strong>, <strong>AYUSH & Ayurveda Regulations</strong>, <strong>Traditional Knowledge (TKDL)</strong>, <strong>Biological Diversity Act</strong>, and <strong>PCT / WIPO Treaties</strong>.
              </p>
            </div>

            {/* Quick Starter Cards */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-left pt-2">
              <button
                onClick={() => onSelectExampleQuery("What does Section 3(p) of the Indian Patents Act, 1970 state regarding traditional knowledge?")}
                className="p-3.5 rounded-xl bg-white border border-slate-200 hover:border-amber-400 hover:shadow-sm transition cursor-pointer text-xs group"
              >
                <div className="flex items-center justify-between mb-1.5">
                  <span className="font-bold text-amber-800 text-[10px] uppercase">Exact Provision</span>
                  <span className="text-[10px] text-slate-400">Patents Act</span>
                </div>
                <p className="font-medium text-slate-800 group-hover:text-slate-950">
                  What does Section 3(p) of the Indian Patents Act, 1970 state regarding traditional knowledge?
                </p>
              </button>

              <button
                onClick={() => onSelectExampleQuery("Can an Ayurvedic invention using traditional knowledge and biological resources be patented in India?")}
                className="p-3.5 rounded-xl bg-white border border-slate-200 hover:border-amber-400 hover:shadow-sm transition cursor-pointer text-xs group"
              >
                <div className="flex items-center justify-between mb-1.5">
                  <span className="font-bold text-emerald-800 text-[10px] uppercase">Cross-Domain</span>
                  <span className="text-[10px] text-slate-400">Ayurveda + TK + NBA</span>
                </div>
                <p className="font-medium text-slate-800 group-hover:text-slate-950">
                  Can an Ayurvedic invention using traditional knowledge and biological resources be patented in India?
                </p>
              </button>

              <button
                onClick={() => onSelectExampleQuery("राष्ट्रीय जैव विविधता प्राधिकरण (NBA) से पेटेंट के लिए अनुमति कब आवश्यक होती है?")}
                className="p-3.5 rounded-xl bg-white border border-slate-200 hover:border-amber-400 hover:shadow-sm transition cursor-pointer text-xs group"
              >
                <div className="flex items-center justify-between mb-1.5">
                  <span className="font-bold text-orange-800 text-[10px] uppercase">हिन्दी Multilingual</span>
                  <span className="text-[10px] text-slate-400">Biological Diversity</span>
                </div>
                <p className="font-medium text-slate-800 group-hover:text-slate-950">
                  राष्ट्रीय जैव विविधता प्राधिकरण (NBA) से पेटेंट के लिए अनुमति कब आवश्यक होती है?
                </p>
              </button>

              <button
                onClick={() => onSelectExampleQuery("What are the timeline and requirements for entering the National Phase in India under the Patent Cooperation Treaty (PCT)?")}
                className="p-3.5 rounded-xl bg-white border border-slate-200 hover:border-amber-400 hover:shadow-sm transition cursor-pointer text-xs group"
              >
                <div className="flex items-center justify-between mb-1.5">
                  <span className="font-bold text-blue-800 text-[10px] uppercase">International IP</span>
                  <span className="text-[10px] text-slate-400">PCT 31-Month Rule</span>
                </div>
                <p className="font-medium text-slate-800 group-hover:text-slate-950">
                  What are the timeline and requirements for entering the National Phase in India under the PCT?
                </p>
              </button>
            </div>
          </div>
        ) : (
          /* Message List */
          messages.map((msg) => (
            <div key={msg.id} className="space-y-4 max-w-3xl mx-auto">
              {/* User Message */}
              {msg.sender === 'user' && (
                <div className="flex items-start justify-end gap-2.5">
                  <div className="bg-slate-900 text-white p-3.5 rounded-2xl rounded-tr-xs max-w-[85%] sm:max-w-[75%] shadow-sm text-xs sm:text-sm font-medium leading-relaxed">
                    {msg.query}
                  </div>
                  <div className="h-7 w-7 rounded-full bg-slate-800 text-slate-300 flex items-center justify-center shrink-0 mt-0.5 text-xs">
                    <User className="h-4 w-4" />
                  </div>
                </div>
              )}

              {/* Assistant Message */}
              {msg.sender === 'assistant' && msg.response && (
                <div className="flex items-start gap-2.5">
                  <div className="h-8 w-8 rounded-lg bg-gradient-to-br from-amber-500 to-amber-700 text-slate-950 font-bold flex items-center justify-center shrink-0 mt-0.5 shadow-sm">
                    <Scale className="h-4 w-4 stroke-[2.5]" />
                  </div>

                  <div className="flex-1 space-y-3 min-w-0">
                    {/* Assistant Response Card */}
                    <div className="bg-white border border-slate-200 rounded-2xl p-4 sm:p-5 shadow-xs space-y-4">
                      {/* Meta Header Badges */}
                      <div className="flex flex-wrap items-center gap-1.5 pb-3 border-b border-slate-100 text-[11px]">
                        {/* Jurisdiction Badge */}
                        {msg.response.jurisdiction && (
                          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-slate-100 text-slate-700 font-semibold border border-slate-200">
                            <Globe2 className="h-3 w-3 text-slate-500" />
                            {msg.response.jurisdiction}
                          </span>
                        )}

                        {/* Domain Badges */}
                        {msg.response.domains?.map((dom, dIdx) => (
                          <span
                            key={dIdx}
                            className="px-2 py-0.5 rounded-full bg-amber-50 text-amber-800 font-semibold border border-amber-200 text-[10px] uppercase"
                          >
                            {dom}
                          </span>
                        ))}

                        {/* Evidence Verification Status Badge */}
                        {msg.response.is_refusal ? (
                          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-amber-100 text-amber-900 font-semibold border border-amber-300">
                            <AlertCircle className="h-3 w-3 text-amber-700" />
                            Scope Safeguard
                          </span>
                        ) : (
                          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-emerald-50 text-emerald-800 font-semibold border border-emerald-200">
                            <ShieldCheck className="h-3 w-3 text-emerald-600" />
                            Evidence Verified ({Math.round(msg.response.validation.claim_support_rate * 100)}% Claim Support)
                          </span>
                        )}
                      </div>

                      {/* Content / Answer */}
                      {msg.response.is_refusal ? (
                        <RefusalCard
                          answer={msg.response.answer}
                          queryType={msg.response.query_type}
                        />
                      ) : (
                        <div className="text-xs sm:text-sm text-slate-800 leading-relaxed space-y-3 prose-slate max-w-none">
                          {msg.response.answer.split('\n\n').map((paragraph, pIdx) => {
                            if (paragraph.startsWith('### Sources')) {
                              return null;
                            }
                            if (paragraph.startsWith('###')) {
                              const heading = paragraph.replace(/^###\s*/, '');
                              return (
                                <h4 key={pIdx} className="text-xs font-bold text-slate-700 pt-2 border-t border-slate-100 tracking-tight">
                                  {heading}
                                </h4>
                              );
                            }
                            return (
                              <p key={pIdx} className="whitespace-pre-line">
                                {renderFormattedText(paragraph, msg.response?.citations || [])}
                              </p>
                            );
                          })}
                        </div>
                      )}

                      {/* Structured Citations Cards Strip */}
                      {!msg.response.is_refusal && msg.response.citations && msg.response.citations.length > 0 && (
                        <div className="pt-3 border-t border-slate-100 space-y-2">
                          <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider flex items-center gap-1">
                            <FileCheck2 className="h-3 w-3 text-amber-600" />
                            Cited Authoritative Sources ({msg.response.citations.length})
                          </span>

                          <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                            {msg.response.citations.map((cit) => (
                              <button
                                key={cit.citation_id}
                                onClick={() => onSelectCitation(cit)}
                                className={`text-left p-2 rounded-lg border transition text-xs cursor-pointer ${
                                  selectedCitationId === cit.citation_id
                                    ? 'bg-amber-50/80 border-amber-400 shadow-xs'
                                    : 'bg-slate-50/70 border-slate-200 hover:bg-slate-100 hover:border-slate-300'
                                }`}
                              >
                                <div className="flex items-center justify-between mb-0.5">
                                  <span className="font-bold text-slate-900 text-[11px] truncate max-w-[180px]">
                                    [{cit.citation_id}] {cit.document}
                                  </span>
                                  <span className="text-[9px] font-semibold px-1.5 py-0.2 rounded bg-white text-slate-600 border border-slate-200 shrink-0">
                                    {cit.source_tier?.split(':')[0] || 'Tier 1'}
                                  </span>
                                </div>
                                <div className="text-[10px] text-slate-500 flex items-center gap-2">
                                  <span>{cit.section || 'Section'}</span>
                                  {cit.page !== undefined && <span>• p. {cit.page}</span>}
                                  <span className="text-amber-700 font-medium ml-auto">Inspect →</span>
                                </div>
                              </button>
                            ))}
                          </div>
                        </div>
                      )}

                      {/* Observability Footer */}
                      <div className="pt-2 border-t border-slate-100 flex flex-wrap items-center justify-between text-[10px] text-slate-400">
                        <span className="flex items-center gap-1">
                          <Clock className="h-3 w-3" />
                          Latency: {Math.round(msg.response.metadata.latency_ms)} ms
                        </span>
                        <span>
                          Orchestration: LangGraph ({msg.response.metadata.generation_attempts} gen attempt)
                        </span>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* Error State */}
              {msg.sender === 'assistant' && msg.error && (
                <div className="flex items-start gap-2.5">
                  <div className="h-8 w-8 rounded-lg bg-rose-100 text-rose-800 flex items-center justify-center shrink-0 mt-0.5">
                    <AlertCircle className="h-4 w-4" />
                  </div>
                  <div className="bg-rose-50 border border-rose-200 text-rose-900 p-3.5 rounded-2xl text-xs sm:text-sm font-medium">
                    {msg.error}
                  </div>
                </div>
              )}
            </div>
          ))
        )}

        {/* Loading Indicator */}
        {loading && (
          <div className="flex items-start gap-2.5 max-w-3xl mx-auto">
            <div className="h-8 w-8 rounded-lg bg-gradient-to-br from-amber-500 to-amber-700 text-slate-950 font-bold flex items-center justify-center shrink-0 animate-pulse">
              <Scale className="h-4 w-4 stroke-[2.5]" />
            </div>
            <div className="bg-white border border-slate-200 rounded-2xl p-4 shadow-xs space-y-2 text-xs text-slate-600 flex items-center gap-3">
              <Loader2 className="h-4 w-4 text-amber-600 animate-spin" />
              <div className="space-y-0.5">
                <span className="font-semibold text-slate-900 block">
                  LangGraph Orchestrating Verification...
                </span>
                <span className="text-[11px] text-slate-400">
                  Retrieving BGE-M3 + BM25 evidence • Reranking • Claim-level citation validation
                </span>
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Query Input Box */}
      <div className="p-4 bg-white border-t border-slate-200 shadow-lg">
        <form onSubmit={handleSubmit} className="max-w-3xl mx-auto relative">
          <div className="relative flex items-center rounded-xl border border-slate-300 bg-slate-50/50 shadow-inner focus-within:ring-2 focus-within:ring-amber-500 focus-within:border-amber-500 focus-within:bg-white transition">
            <textarea
              ref={textareaRef}
              value={inputQuery}
              onChange={(e) => setInputQuery(e.target.value)}
              onKeyDown={handleKeyDown}
              disabled={loading}
              placeholder="Ask an IP, Ayurveda, Traditional Knowledge, or Biodiversity question (English / हिन्दी / Hinglish)..."
              rows={1}
              className="w-full py-3.5 pl-4 pr-12 text-xs sm:text-sm text-slate-900 placeholder:text-slate-400 bg-transparent resize-none focus:outline-none max-h-32"
            />
            <button
              type="submit"
              disabled={!inputQuery.trim() || loading}
              className={`absolute right-2.5 p-2 rounded-lg transition ${
                inputQuery.trim() && !loading
                  ? 'bg-slate-900 text-amber-400 hover:bg-slate-800 shadow-sm cursor-pointer'
                  : 'bg-slate-200 text-slate-400 cursor-not-allowed'
              }`}
              aria-label="Send Query"
            >
              <Send className="h-4 w-4" />
            </button>
          </div>
          <div className="flex items-center justify-between text-[10px] text-slate-400 px-1 pt-1.5">
            <span>Press <strong>Enter</strong> to send, <strong>Shift + Enter</strong> for newline</span>
            <span className="hidden sm:inline">Strict evidence grounding active</span>
          </div>
        </form>
      </div>
    </div>
  );
};
