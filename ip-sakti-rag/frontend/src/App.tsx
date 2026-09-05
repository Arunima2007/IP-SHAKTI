import { useState, useEffect } from 'react';
import { Header } from './components/Header';
import { Sidebar } from './components/Sidebar';
import { ChatArea } from './components/ChatArea';
import { SourceViewer } from './components/SourceViewer';
import { DocumentsModal } from './components/DocumentsModal';
import type { ChatMessage, CitationItem, DocumentInfo, HealthResponse } from './types/api';
import { sendChatQuery, checkHealth, fetchDocuments } from './services/api';

export function App() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [language, setLanguage] = useState<string>('auto');
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [documents, setDocuments] = useState<DocumentInfo[]>([]);
  const [isKbModalOpen, setIsKbModalOpen] = useState<boolean>(false);
  const [selectedCitation, setSelectedCitation] = useState<CitationItem | null>(null);

  // Initial Health & Documents Fetch
  useEffect(() => {
    async function initSystem() {
      try {
        const healthData = await checkHealth();
        setHealth(healthData);
      } catch (e) {
        console.warn('Backend health check error:', e);
      }

      try {
        const docData = await fetchDocuments();
        setDocuments(docData.documents || []);
      } catch (e) {
        console.warn('Document catalog fetch error:', e);
      }
    }
    initSystem();
  }, []);

  const handleSendMessage = async (queryText: string) => {
    if (!queryText.trim()) return;

    const userMessage: ChatMessage = {
      id: `user-${Date.now()}`,
      sender: 'user',
      timestamp: new Date().toISOString(),
      query: queryText,
    };

    setMessages((prev) => [...prev, userMessage]);
    setLoading(true);

    try {
      const response = await sendChatQuery({
        query: queryText,
        language: language === 'auto' ? undefined : language,
      });

      const assistantMessage: ChatMessage = {
        id: `assistant-${Date.now()}`,
        sender: 'assistant',
        timestamp: new Date().toISOString(),
        response: response,
      };

      setMessages((prev) => [...prev, assistantMessage]);

      // If response has citations, auto-select the first citation for immediate inspection
      if (response.citations && response.citations.length > 0) {
        setSelectedCitation(response.citations[0]);
      } else {
        setSelectedCitation(null);
      }
    } catch (err: any) {
      const errorMessage: ChatMessage = {
        id: `error-${Date.now()}`,
        sender: 'assistant',
        timestamp: new Date().toISOString(),
        error: err.message || 'An unexpected error occurred while communicating with the API.',
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  const handleClearChat = () => {
    setMessages([]);
    setSelectedCitation(null);
  };

  return (
    <div className="min-h-screen bg-slate-100 flex flex-col font-sans">
      {/* Navigation Header */}
      <Header
        health={health}
        language={language}
        onLanguageChange={setLanguage}
        onToggleKbModal={() => setIsKbModalOpen(true)}
      />

      {/* Main 3-Column Layout */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left Sidebar: Knowledge Base Pillars & Demo Queries */}
        <div className="hidden md:block">
          <Sidebar
            onSelectQuery={handleSendMessage}
            onClearChat={handleClearChat}
            hasMessages={messages.length > 0}
          />
        </div>

        {/* Center: Interactive Chat Area */}
        <ChatArea
          messages={messages}
          loading={loading}
          onSendMessage={handleSendMessage}
          onSelectCitation={setSelectedCitation}
          selectedCitationId={selectedCitation?.citation_id || null}
          onSelectExampleQuery={handleSendMessage}
        />

        {/* Right Sidebar: Source & Evidence Inspector */}
        <SourceViewer
          citation={selectedCitation}
          onClose={() => setSelectedCitation(null)}
        />
      </div>

      {/* Authoritative Documents Modal */}
      <DocumentsModal
        isOpen={isKbModalOpen}
        onClose={() => setIsKbModalOpen(false)}
        documents={documents}
      />
    </div>
  );
}

export default App;
