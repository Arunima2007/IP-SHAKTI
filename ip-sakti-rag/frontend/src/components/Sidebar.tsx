import React from 'react';
import { 
  Scale, 
  Leaf, 
  BookMarked, 
  Sparkles, 
  Trash2, 
  HelpCircle,
  FileText
} from 'lucide-react';

interface SidebarProps {
  onSelectQuery: (query: string) => void;
  onClearChat: () => void;
  hasMessages: boolean;
}

interface DemoQuery {
  id: string;
  category: string;
  query: string;
  badge: string;
  color: string;
}

const DEMO_QUERIES: DemoQuery[] = [
  {
    id: '1',
    category: 'Exact Statutory Lookup',
    query: 'What does Section 3(p) of the Indian Patents Act, 1970 state regarding traditional knowledge?',
    badge: 'Section 3(p)',
    color: 'bg-amber-100 text-amber-800 border-amber-300'
  },
  {
    id: '2',
    category: 'Ayurveda & Patents',
    query: 'Can a classical Ayurvedic formulation described in the Ayurvedic Pharmacopoeia of India be patented as an invention?',
    badge: 'Ayurveda IP',
    color: 'bg-emerald-100 text-emerald-800 border-emerald-300'
  },
  {
    id: '3',
    category: 'Biodiversity & NBA',
    query: 'When is prior approval from the National Biodiversity Authority (NBA) required before applying for a patent in India?',
    badge: 'Section 6 NBA',
    color: 'bg-teal-100 text-teal-800 border-teal-300'
  },
  {
    id: '4',
    category: 'Traditional Knowledge',
    query: 'What role does the Traditional Knowledge Digital Library (TKDL) play in preventing biopiracy and invalid patent grants?',
    badge: 'TKDL Prior Art',
    color: 'bg-indigo-100 text-indigo-800 border-indigo-300'
  },
  {
    id: '5',
    category: 'International IP / PCT',
    query: 'What are the timeline and requirements for entering the National Phase in India under the Patent Cooperation Treaty (PCT)?',
    badge: 'PCT 31-Month',
    color: 'bg-blue-100 text-blue-800 border-blue-300'
  },
  {
    id: '6',
    category: 'Hindi Multilingual',
    query: 'राष्ट्रीय जैव विविधता प्राधिकरण (NBA) से पेटेंट के लिए अनुमति कब आवश्यक होती है?',
    badge: 'हिन्दी (Hindi)',
    color: 'bg-orange-100 text-orange-800 border-orange-300'
  },
  {
    id: '7',
    category: 'Hinglish Code-Mixed',
    query: 'Kya Ayurvedic plants aur biological resources use karke banaye gaye invention ko India me patent mil sakta hai?',
    badge: 'Hinglish',
    color: 'bg-purple-100 text-purple-800 border-purple-300'
  },
  {
    id: '8',
    category: 'Cross-Domain Intersection',
    query: 'Can an Ayurvedic invention using traditional knowledge and biological resources be patented in India?',
    badge: 'Cross-Domain',
    color: 'bg-rose-100 text-rose-800 border-rose-300'
  }
];

export const Sidebar: React.FC<SidebarProps> = ({
  onSelectQuery,
  onClearChat,
  hasMessages
}) => {
  return (
    <aside className="w-80 bg-white border-r border-slate-200 flex flex-col h-[calc(100vh-4rem)] overflow-hidden shrink-0 shadow-sm">
      {/* Knowledge Base Scope Overview */}
      <div className="p-4 border-b border-slate-200 bg-slate-50/70">
        <div className="flex items-center justify-between mb-2">
          <span className="text-xs font-bold uppercase tracking-wider text-slate-500 flex items-center gap-1.5">
            <BookMarked className="h-3.5 w-3.5 text-amber-600" />
            Knowledge Base Pillars
          </span>
          <span className="text-[10px] font-semibold px-2 py-0.5 rounded-full bg-emerald-100 text-emerald-800 border border-emerald-300">
            22 Statutes & Guides
          </span>
        </div>
        <div className="grid grid-cols-2 gap-1.5 text-[11px] font-medium text-slate-700">
          <div className="flex items-center gap-1.5 p-1.5 rounded bg-white border border-slate-200/80">
            <Scale className="h-3 w-3 text-amber-600 shrink-0" />
            <span className="truncate">Patents Act 1970</span>
          </div>
          <div className="flex items-center gap-1.5 p-1.5 rounded bg-white border border-slate-200/80">
            <Leaf className="h-3 w-3 text-emerald-600 shrink-0" />
            <span className="truncate">Biodiversity 2002</span>
          </div>
          <div className="flex items-center gap-1.5 p-1.5 rounded bg-white border border-slate-200/80">
            <Sparkles className="h-3 w-3 text-indigo-600 shrink-0" />
            <span className="truncate">AYUSH & Aahara</span>
          </div>
          <div className="flex items-center gap-1.5 p-1.5 rounded bg-white border border-slate-200/80">
            <FileText className="h-3 w-3 text-blue-600 shrink-0" />
            <span className="truncate">PCT & WIPO Treaties</span>
          </div>
        </div>
      </div>

      {/* Curated SIH Demo Prompts */}
      <div className="flex-1 overflow-y-auto p-4 space-y-3">
        <div className="flex items-center justify-between">
          <span className="text-xs font-bold uppercase tracking-wider text-slate-500 flex items-center gap-1.5">
            <HelpCircle className="h-3.5 w-3.5 text-blue-600" />
            Curated SIH Test Queries
          </span>
          <span className="text-[10px] text-slate-400">Click to Run</span>
        </div>

        <div className="space-y-2">
          {DEMO_QUERIES.map((item) => (
            <button
              key={item.id}
              onClick={() => onSelectQuery(item.query)}
              className="w-full text-left p-2.5 rounded-lg border border-slate-200 bg-white hover:bg-slate-50 hover:border-amber-400 transition group shadow-2xs cursor-pointer"
            >
              <div className="flex items-center justify-between mb-1">
                <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">
                  {item.category}
                </span>
                <span className={`text-[10px] font-semibold px-1.5 py-0.5 rounded border ${item.color}`}>
                  {item.badge}
                </span>
              </div>
              <p className="text-xs font-medium text-slate-800 line-clamp-2 group-hover:text-slate-950">
                {item.query}
              </p>
            </button>
          ))}
        </div>
      </div>

      {/* Footer Controls */}
      <div className="p-3 border-t border-slate-200 bg-slate-50">
        <button
          onClick={onClearChat}
          disabled={!hasMessages}
          className={`w-full flex items-center justify-center space-x-1.5 px-3 py-2 text-xs font-semibold rounded-md border transition ${
            hasMessages 
              ? 'bg-white text-rose-700 border-rose-200 hover:bg-rose-50 hover:border-rose-300 shadow-2xs cursor-pointer' 
              : 'bg-slate-100 text-slate-400 border-slate-200 cursor-not-allowed'
          }`}
        >
          <Trash2 className="h-3.5 w-3.5" />
          <span>Clear Conversation</span>
        </button>
      </div>
    </aside>
  );
};
