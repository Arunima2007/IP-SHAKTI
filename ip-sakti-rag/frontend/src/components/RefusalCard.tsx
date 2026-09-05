import React from 'react';
import { AlertTriangle, ShieldAlert, BookOpen } from 'lucide-react';

interface RefusalCardProps {
  answer: string;
  queryType?: string;
}

export const RefusalCard: React.FC<RefusalCardProps> = ({ answer, queryType }) => {
  const isOutOfScope = queryType === 'OUT_OF_SCOPE' || answer.includes('Scope Notice');

  return (
    <div className="rounded-xl border border-amber-300 bg-gradient-to-br from-amber-50 to-orange-50/40 p-4 shadow-sm space-y-3 my-2">
      <div className="flex items-start gap-3">
        <div className="h-8 w-8 rounded-lg bg-amber-500/20 border border-amber-400 flex items-center justify-center shrink-0 mt-0.5">
          {isOutOfScope ? (
            <ShieldAlert className="h-4 w-4 text-amber-800" />
          ) : (
            <AlertTriangle className="h-4 w-4 text-amber-800" />
          )}
        </div>
        <div className="space-y-1">
          <div className="flex items-center gap-2">
            <h4 className="text-xs font-bold text-amber-950 uppercase tracking-wider">
              {isOutOfScope ? 'Out-of-Scope Domain Boundary Notice' : 'Evidence Sufficiency Safeguard'}
            </h4>
            <span className="text-[10px] font-semibold px-2 py-0.2 rounded-full bg-amber-200/80 text-amber-900 border border-amber-300">
              Safe Refusal
            </span>
          </div>
          <p className="text-xs font-medium text-amber-900/90 leading-relaxed">
            {answer.split('### Scope Notice')[0].trim()}
          </p>
        </div>
      </div>

      {isOutOfScope && answer.includes('Scope Notice') && (
        <div className="pt-2.5 border-t border-amber-200/70 text-xs text-amber-900/80 space-y-1">
          <div className="flex items-center gap-1.5 font-semibold text-amber-950 text-[11px]">
            <BookOpen className="h-3.5 w-3.5 text-amber-700" />
            <span>Authoritative System Scope</span>
          </div>
          <p className="text-[11px] leading-normal pl-5">
            IP-SAKTI Sahayak strictly specializes in Indian Intellectual Property Law (Patents, Trademarks, Copyrights, Designs), AYUSH & Ayurveda regulations, Biological Diversity governance, and international patent treaties (PCT, WIPO, EPO).
          </p>
        </div>
      )}
    </div>
  );
};
