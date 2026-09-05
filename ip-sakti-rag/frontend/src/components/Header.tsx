import React from 'react';
import { Scale, Globe, ShieldCheck, AlertCircle, BookOpen } from 'lucide-react';
import type { HealthResponse } from '../types/api';

interface HeaderProps {
  health: HealthResponse | null;
  language: string;
  onLanguageChange: (lang: string) => void;
  onToggleKbModal: () => void;
}

export const Header: React.FC<HeaderProps> = ({
  health,
  language,
  onLanguageChange,
  onToggleKbModal,
}) => {
  return (
    <header className="bg-slate-900 text-white border-b border-slate-800 sticky top-0 z-40 shadow-md">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo & Title */}
          <div className="flex items-center space-x-3">
            <div className="h-10 w-10 rounded-lg bg-gradient-to-br from-amber-500 to-amber-700 flex items-center justify-center shadow-inner">
              <Scale className="h-6 w-6 text-slate-950 stroke-[2.5]" />
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <h1 className="text-lg font-bold tracking-tight text-white font-['Outfit']">
                  IP-SAKTI Sahayak
                </h1>
                <span className="hidden sm:inline-block px-2 py-0.5 text-xs font-semibold rounded bg-amber-500/20 text-amber-300 border border-amber-500/30">
                  SIH 2026
                </span>
              </div>
              <p className="text-xs text-slate-400 font-medium">
                आयुष एवं बौद्धिक संपदा सहायक • AI Legal & Ayurveda IP Assistant
              </p>
            </div>
          </div>

          {/* Right Actions: KB button, Health Status, Language Selector */}
          <div className="flex items-center space-x-3 sm:space-x-4">
            {/* Knowledge Base Catalog Button */}
            <button
              onClick={onToggleKbModal}
              className="flex items-center space-x-1.5 px-3 py-1.5 text-xs font-medium rounded-md bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 transition cursor-pointer"
              title="View 22 Authoritative Documents"
            >
              <BookOpen className="h-4 w-4 text-amber-400" />
              <span className="hidden md:inline">22 Legal Docs</span>
            </button>

            {/* Health Status Indicator */}
            <div className="hidden lg:flex items-center space-x-2 px-3 py-1.5 rounded-md bg-slate-800/80 border border-slate-700 text-xs">
              {health?.backend_connected ? (
                <>
                  <ShieldCheck className="h-4 w-4 text-emerald-400" />
                  <span className="text-slate-300 font-medium">
                    Verified KB: <strong className="text-emerald-400">{health.total_chunks_indexed.toLocaleString()} Chunks</strong>
                  </span>
                </>
              ) : (
                <>
                  <AlertCircle className="h-4 w-4 text-amber-400 animate-pulse" />
                  <span className="text-slate-400">Connecting...</span>
                </>
              )}
            </div>

            {/* Language Selector */}
            <div className="flex items-center space-x-1 bg-slate-800 border border-slate-700 rounded-md px-2 py-1">
              <Globe className="h-3.5 w-3.5 text-slate-400" />
              <select
                value={language}
                onChange={(e) => onLanguageChange(e.target.value)}
                className="bg-transparent text-xs text-slate-200 font-medium focus:outline-none cursor-pointer pr-1"
                aria-label="Select Language"
              >
                <option value="auto" className="bg-slate-900 text-white">Auto Detect</option>
                <option value="en" className="bg-slate-900 text-white">English</option>
                <option value="hi" className="bg-slate-900 text-white">हिन्दी (Hindi)</option>
                <option value="hinglish" className="bg-slate-900 text-white">Hinglish</option>
              </select>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
};
