import React from 'react';
import { X, BookOpen, Scale, FileText, Copy, Check } from 'lucide-react';
import type { CitationItem } from '../types/api';

interface SourceViewerProps {
  citation: CitationItem | null;
  onClose: () => void;
}

export const SourceViewer: React.FC<SourceViewerProps> = ({ citation, onClose }) => {
  const [copied, setCopied] = React.useState(false);

  if (!citation) {
    return (
      <aside className="w-80 lg:w-96 bg-slate-50 border-l border-slate-200 hidden md:flex flex-col h-[calc(100vh-4rem)] p-6 items-center justify-center text-center text-slate-400">
        <div className="h-12 w-12 rounded-full bg-slate-100 flex items-center justify-center mb-3">
          <BookOpen className="h-6 w-6 text-slate-300" />
        </div>
        <h3 className="text-sm font-semibold text-slate-600 mb-1">Source & Evidence Inspector</h3>
        <p className="text-xs text-slate-400 max-w-[220px]">
          Click on any numbered citation tag like <span className="citation-badge">[1]</span> inside an answer to inspect its verified legal excerpt and authority tier.
        </p>
      </aside>
    );
  }

  const handleCopy = () => {
    if (citation.excerpt) {
      navigator.clipboard.writeText(citation.excerpt);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  const getTierBadge = (tier?: string) => {
    if (tier?.includes('Tier 1')) {
      return {
        label: 'Tier 1: Primary Legislation / Statute',
        bg: 'bg-amber-500/10 text-amber-800 border-amber-300'
      };
    } else if (tier?.includes('Tier 2')) {
      return {
        label: 'Tier 2: Official Guideline / Regulation',
        bg: 'bg-blue-500/10 text-blue-800 border-blue-300'
      };
    } else {
      return {
        label: 'Tier 3: Institutional Source / Treaty Study',
        bg: 'bg-slate-500/10 text-slate-800 border-slate-300'
      };
    }
  };

  const tierBadge = getTierBadge(citation.source_tier);

  return (
    <aside className="w-80 lg:w-96 bg-white border-l border-slate-200 flex flex-col h-[calc(100vh-4rem)] shadow-lg z-30 shrink-0">
      {/* Header */}
      <div className="p-4 border-b border-slate-200 bg-slate-50 flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <div className="h-7 w-7 rounded-md bg-amber-500 text-slate-950 font-bold flex items-center justify-center text-xs shadow-xs">
            [{citation.citation_id}]
          </div>
          <div>
            <h3 className="text-xs font-bold text-slate-900 uppercase tracking-wide">
              Verified Legal Evidence
            </h3>
            <span className="text-[10px] text-slate-500 font-mono">
              Provenance: {citation.evidence_id}
            </span>
          </div>
        </div>
        <button
          onClick={onClose}
          className="p-1 text-slate-400 hover:text-slate-600 rounded-md hover:bg-slate-200/60 transition cursor-pointer"
          aria-label="Close Source Inspector"
        >
          <X className="h-4 w-4" />
        </button>
      </div>

      {/* Body Content */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {/* Source Authority Tier Card */}
        <div className="space-y-1.5">
          <span className="text-[10px] font-bold uppercase tracking-wider text-slate-400">
            Source Authority Hierarchy
          </span>
          <div className={`p-2.5 rounded-lg border text-xs font-semibold flex items-center gap-2 ${tierBadge.bg}`}>
            <Scale className="h-4 w-4 shrink-0" />
            <span>{tierBadge.label}</span>
          </div>
        </div>

        {/* Metadata Details Grid */}
        <div className="bg-slate-50 p-3 rounded-lg border border-slate-200 space-y-2 text-xs">
          <div>
            <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider block">
              Authoritative Document
            </span>
            <span className="font-semibold text-slate-900 break-words flex items-start gap-1.5 mt-0.5">
              <FileText className="h-3.5 w-3.5 text-slate-500 shrink-0 mt-0.5" />
              {citation.document}
            </span>
          </div>

          <div className="grid grid-cols-2 gap-2 pt-1 border-t border-slate-200">
            <div>
              <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider block">
                Provision / Section
              </span>
              <span className="font-medium text-slate-800">
                {citation.section || 'General Provision'}
              </span>
            </div>
            <div>
              <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider block">
                Page Number
              </span>
              <span className="font-medium text-slate-800">
                {citation.page !== undefined ? `Page ${citation.page}` : 'Official Gazette / Act'}
              </span>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-2 pt-1 border-t border-slate-200">
            <div>
              <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider block">
                Jurisdiction
              </span>
              <span className="font-medium text-slate-800">
                {citation.jurisdiction || 'India'}
              </span>
            </div>
            {citation.domain && (
              <div>
                <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider block">
                  Domain
                </span>
                <span className="font-medium text-slate-800 uppercase text-[11px]">
                  {citation.domain}
                </span>
              </div>
            )}
          </div>
        </div>

        {/* Verbatim Excerpt */}
        <div className="space-y-1.5">
          <div className="flex items-center justify-between">
            <span className="text-[10px] font-bold uppercase tracking-wider text-slate-400">
              Verbatim Statutory Excerpt
            </span>
            <button
              onClick={handleCopy}
              className="text-[10px] font-medium text-slate-600 hover:text-slate-900 flex items-center gap-1 cursor-pointer"
            >
              {copied ? (
                <>
                  <Check className="h-3 w-3 text-emerald-600" />
                  <span className="text-emerald-600 font-semibold">Copied</span>
                </>
              ) : (
                <>
                  <Copy className="h-3 w-3" />
                  <span>Copy Text</span>
                </>
              )}
            </button>
          </div>
          <div className="p-3 bg-amber-50/50 border border-amber-200 rounded-lg text-xs leading-relaxed text-slate-800 font-serif italic max-h-72 overflow-y-auto shadow-inner">
            "{citation.excerpt || 'Excerpt text preserved in authoritative knowledge base.'}"
          </div>
        </div>
      </div>
    </aside>
  );
};
