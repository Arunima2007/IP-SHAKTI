import React from 'react';
import { X, BookOpen, Scale, CheckCircle2 } from 'lucide-react';
import type { DocumentInfo } from '../types/api';

interface DocumentsModalProps {
  isOpen: boolean;
  onClose: () => void;
  documents: DocumentInfo[];
}

export const DocumentsModal: React.FC<DocumentsModalProps> = ({
  isOpen,
  onClose,
  documents
}) => {
  const [selectedCategory, setSelectedCategory] = React.useState<string>('all');

  if (!isOpen) return null;

  const categories = ['all', ...Array.from(new Set(documents.map(d => d.category)))];

  const filtered = selectedCategory === 'all' 
    ? documents 
    : documents.filter(d => d.category === selectedCategory);

  const getTierColor = (tier: string) => {
    if (tier.includes('Tier 1')) return 'bg-amber-100 text-amber-900 border-amber-300';
    if (tier.includes('Tier 2')) return 'bg-blue-100 text-blue-900 border-blue-300';
    return 'bg-slate-100 text-slate-900 border-slate-300';
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/60 backdrop-blur-xs p-4">
      <div className="bg-white rounded-2xl max-w-4xl w-full max-h-[85vh] flex flex-col shadow-2xl border border-slate-200 overflow-hidden animate-in fade-in zoom-in-95 duration-150">
        {/* Header */}
        <div className="p-5 border-b border-slate-200 bg-slate-900 text-white flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="h-9 w-9 rounded-lg bg-amber-500 flex items-center justify-center text-slate-950">
              <BookOpen className="h-5 w-5" />
            </div>
            <div>
              <h2 className="text-base font-bold font-['Outfit']">
                Authoritative Knowledge Base Inventory
              </h2>
              <p className="text-xs text-slate-400">
                22 Validated Primary Statutes, Official Guidelines, Treaties & Compendiums (5,212 Chunks)
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition cursor-pointer"
            aria-label="Close modal"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        {/* Category Filter Pills */}
        <div className="p-3 bg-slate-50 border-b border-slate-200 flex items-center gap-1.5 overflow-x-auto text-xs">
          {categories.map(cat => (
            <button
              key={cat}
              onClick={() => setSelectedCategory(cat)}
              className={`px-3 py-1 rounded-full font-medium whitespace-nowrap transition cursor-pointer ${
                selectedCategory === cat
                  ? 'bg-slate-900 text-white shadow-xs'
                  : 'bg-white text-slate-600 border border-slate-200 hover:bg-slate-100'
              }`}
            >
              {cat === 'all' ? `All Documents (${documents.length})` : cat}
            </button>
          ))}
        </div>

        {/* Document List */}
        <div className="flex-1 overflow-y-auto p-5 grid grid-cols-1 md:grid-cols-2 gap-3">
          {filtered.map(doc => (
            <div
              key={doc.id}
              className="p-3.5 rounded-xl border border-slate-200 bg-white hover:border-amber-400 hover:shadow-xs transition space-y-2"
            >
              <div className="flex items-start justify-between gap-2">
                <h3 className="text-xs font-bold text-slate-900 leading-snug line-clamp-2">
                  {doc.title}
                </h3>
                <span className={`text-[10px] font-semibold px-2 py-0.5 rounded border shrink-0 ${getTierColor(doc.authority_tier)}`}>
                  {doc.authority_tier.split(':')[0]}
                </span>
              </div>

              <div className="flex items-center gap-3 text-[11px] text-slate-500 font-medium">
                <span className="flex items-center gap-1">
                  <Scale className="h-3 w-3 text-slate-400" />
                  {doc.jurisdiction}
                </span>
                <span>•</span>
                <span>{doc.chunk_count} Chunks</span>
                {doc.year && (
                  <>
                    <span>•</span>
                    <span>Year: {doc.year}</span>
                  </>
                )}
              </div>

              <div className="pt-1.5 border-t border-slate-100 flex items-center justify-between text-[10px] text-slate-400 font-mono">
                <span className="truncate max-w-[200px]">{doc.id}</span>
                <span className="text-emerald-600 font-sans font-semibold flex items-center gap-1">
                  <CheckCircle2 className="h-3 w-3" /> Indexed
                </span>
              </div>
            </div>
          ))}
        </div>

        {/* Footer */}
        <div className="p-3 bg-slate-50 border-t border-slate-200 flex items-center justify-between text-xs text-slate-500">
          <span>Strict citation grounding active against all 22 official sources.</span>
          <button
            onClick={onClose}
            className="px-4 py-1.5 bg-slate-900 text-white rounded-lg font-semibold hover:bg-slate-800 transition cursor-pointer"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
};
