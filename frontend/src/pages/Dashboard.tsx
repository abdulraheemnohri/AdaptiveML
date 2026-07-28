import React, { useState, useEffect } from 'react';
import { AIWorkspacePage } from './AIWorkspacePage';
import { MemoryPage } from './MemoryPage';
import { DatasetManagerPage } from './DatasetManagerPage';
import { QualityPoisoningPage } from './QualityPoisoningPage';
import { GapResearchPage } from './GapResearchPage';
import { TrainingLabPage } from './TrainingLabPage';
import { RegistryPromotionPage } from './RegistryPromotionPage';
import { TestingLabPage } from './TestingLabPage';
import { ObservabilityPage } from './ObservabilityPage';

interface SystemState {
  currentModel: string;
  knowledge: string;
  newCapabilities: string;
  oldCapabilitiesRetained: string;
  forgettingRisk: string;
  dataTrust: string;
  modelSafety: string;
  modelQuality: string;
  currentLearning: string;
  nextAction: string;
  status: string;
}

export const Dashboard: React.FC = () => {
  // Tab layout spanning all 48 sections
  const [activeTab, setActiveTab] = useState<string>('control');

  // Unified global API status state
  const [state, setState] = useState<SystemState>({
    currentModel: "Qwen2.5-Omni-3B Adaptive v3.4.2",
    knowledge: "+24.8%",
    newCapabilities: "+12",
    oldCapabilitiesRetained: "99.1%",
    forgettingRisk: "0.3%",
    dataTrust: "96.7%",
    modelSafety: "98.9%",
    modelQuality: "94.2%",
    currentLearning: "Researching → Data Validation",
    nextAction: "Continual Learning",
    status: "SAFE TO CONTINUE"
  });

  const [controlLogs, setControlLogs] = useState<string[]>([
    "Brain Core initialized.",
    "Experience replay memory load status: [1.2M entries].",
    "Anti-forgetting firewall: Standard monitoring level.",
    "System status: SAFE TO CONTINUE"
  ]);

  const [loadingAction, setLoadingAction] = useState<string | null>(null);

  // Sync state from FastAPI backend
  const fetchAllData = async () => {
    try {
      const response = await fetch('/status');
      if (response.ok) {
        const data = await response.json();
        setState({
          currentModel: data.current_model || "Qwen2.5-Omni-3B Adaptive v3.4.2",
          knowledge: data.knowledge || "+24.8%",
          newCapabilities: data.new_capabilities || "+12",
          oldCapabilitiesRetained: data.old_capabilities_retained || "99.1%",
          forgettingRisk: data.forgetting_risk || "0.3%",
          dataTrust: data.data_trust || "96.7%",
          modelSafety: data.model_safety || "98.9%",
          modelQuality: data.model_quality || "94.2%",
          currentLearning: data.current_learning || "Researching → Data Validation",
          nextAction: data.next_action || "Continual Learning",
          status: data.status || "SAFE TO CONTINUE"
        });
      }
    } catch (e) {
      // Offline fallback
    }
  };

  useEffect(() => {
    fetchAllData();
    const interval = setInterval(fetchAllData, 4000);
    return () => clearInterval(interval);
  }, []);

  // Controls triggers
  const triggerControl = async (
    actionName: string,
    endpoint: string,
    localStateUpdate: Partial<SystemState>,
    logMessage: string
  ) => {
    setLoadingAction(actionName);
    const timeStr = new Date().toLocaleTimeString();
    setControlLogs(prev => [`[${timeStr}] ⚙️ Triggering: ${actionName}...`, ...prev]);
    setState(prev => ({ ...prev, ...localStateUpdate }));

    try {
      const response = await fetch(`/control/${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });
      if (response.ok) {
        const data = await response.json();
        if (data.state) {
          setState({
            currentModel: data.state.current_model,
            knowledge: data.state.knowledge,
            newCapabilities: data.state.new_capabilities,
            oldCapabilitiesRetained: data.state.old_capabilities_retained,
            forgettingRisk: data.state.forgetting_risk,
            dataTrust: data.state.data_trust,
            modelSafety: data.state.model_safety,
            modelQuality: data.state.model_quality,
            currentLearning: data.state.current_learning,
            nextAction: data.state.next_action,
            status: data.state.status
          });
        }
        setControlLogs(prev => [`[${timeStr}] ✅ ${logMessage} (Synced)`, ...prev]);
      } else {
        setControlLogs(prev => [`[${timeStr}] ℹ️ ${logMessage} (Demo Mode)`, ...prev]);
      }
    } catch (error) {
      setControlLogs(prev => [`[${timeStr}] ℹ️ ${logMessage} (Demo Mode)`, ...prev]);
    } finally {
      setTimeout(() => setLoadingAction(null), 500);
    }
  };

  // Sidebar Menu Config
  const sidebarItems = [
    { id: 'control', label: 'Command Center', icon: '🕹️' },
    { id: 'workspace', label: 'AI Workspace', icon: '💬' },
    { id: 'memory', label: 'Memory & RAG', icon: '🧠' },
    { id: 'ingestion', label: 'Ingestion & Pipeline', icon: '📥' },
    { id: 'quality', label: 'Quality & Poisoning', icon: '🛡️' },
    { id: 'gaps', label: 'Gap & Research', icon: '🔍' },
    { id: 'training', label: 'Continual Training', icon: '⚖️' },
    { id: 'registry', label: 'Registry & Promotion', icon: '📦' },
    { id: 'testing', label: 'Testing Lab', icon: '🧪' },
    { id: 'observability', label: 'Observability & System', icon: '📊' }
  ];

  return (
    <div className="flex h-screen bg-slate-950 font-sans overflow-hidden text-slate-100 animate-fadeIn">

      {/* SIDEBAR NAVIGATION */}
      <aside className="w-64 bg-slate-900 border-r border-slate-800 flex flex-col h-full flex-shrink-0">
        <div className="p-5 border-b border-slate-800 flex items-center space-x-3">
          <span className="text-2xl font-black bg-indigo-600 text-white h-10 w-10 flex items-center justify-center rounded-xl shadow-lg">Ω</span>
          <div>
            <h1 className="text-sm font-extrabold text-white tracking-tight leading-none">Adaptive Omni ML</h1>
            <p className="text-[10px] text-slate-500 font-bold uppercase mt-1">Brain Platform</p>
          </div>
        </div>

        <nav className="flex-grow p-3 space-y-1 overflow-y-auto">
          {sidebarItems.map(item => (
            <button
              key={item.id}
              onClick={() => setActiveTab(item.id)}
              className={`w-full flex items-center space-x-3 px-4 py-2.5 rounded-xl text-xs font-bold transition-all ${
                activeTab === item.id
                  ? 'bg-indigo-600 text-white shadow-md'
                  : 'text-slate-400 hover:bg-slate-800/60 hover:text-white'
              }`}
            >
              <span className="text-base">{item.icon}</span>
              <span>{item.label}</span>
            </button>
          ))}
        </nav>
      </aside>

      {/* MAIN VIEWPORT */}
      <main className="flex-grow flex flex-col min-w-0 bg-slate-950 overflow-y-auto p-8">

        {/* TAB 1: COMMAND CENTER */}
        {activeTab === 'control' && (
          <div className="space-y-8 animate-fadeIn">
            <header className="bg-gradient-to-r from-slate-900 to-indigo-950 text-white p-6 rounded-2xl border border-indigo-900/40 shadow-xl">
              <div className="flex flex-col md:flex-row justify-between md:items-center">
                <div>
                  <h1 className="text-2xl font-black tracking-tight">Adaptive Brain Control</h1>
                  <p className="text-xs text-indigo-300 mt-1">Central command and master evolutionary settings of Qwen2.5-Omni-3B.</p>
                </div>
                <span className="mt-4 md:mt-0 bg-emerald-950/80 text-emerald-400 border border-emerald-500/40 px-4 py-1.5 rounded-full text-xs font-bold tracking-wide">
                  STATUS: {state.status}
                </span>
              </div>
            </header>

            <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
              <div className="lg:col-span-7 bg-slate-900 border border-slate-800 rounded-2xl p-6 space-y-6">
                <h2 className="text-sm font-black text-slate-400 uppercase tracking-wider">System Observability Screen</h2>

                <div className="bg-slate-950 rounded-xl p-4 border border-slate-800 flex items-center justify-between">
                  <div>
                    <span className="text-[10px] text-slate-500 font-bold uppercase tracking-wider">CURRENT MODEL</span>
                    <p className="text-base font-extrabold text-cyan-300 mt-1">{state.currentModel}</p>
                  </div>
                </div>

                <div className="grid grid-cols-2 sm:grid-cols-3 gap-4">
                  {[
                    { label: "KNOWLEDGE GROWTH", val: state.knowledge, color: "text-indigo-400" },
                    { label: "NEW CAPABILITIES", val: state.newCapabilities, color: "text-teal-400" },
                    { label: "OLD CAPABILITIES RETAINED", val: state.oldCapabilitiesRetained, color: "text-emerald-400" },
                    { label: "FORGETTING RISK", val: state.forgettingRisk, color: "text-red-400" },
                    { label: "DATA TRUST", val: state.dataTrust, color: "text-cyan-400" },
                    { label: "MODEL SAFETY", val: state.modelSafety, color: "text-purple-400" }
                  ].map((item, idx) => (
                    <div key={idx} className="bg-slate-950 rounded-xl p-4 border border-slate-800">
                      <span className="text-[10px] text-slate-500 font-bold uppercase block">{item.label}</span>
                      <span className={`text-xl font-black block mt-2 ${item.color}`}>{item.val}</span>
                    </div>
                  ))}
                </div>
              </div>

              <div className="lg:col-span-5 bg-slate-900 border border-slate-800 rounded-2xl p-6 space-y-6">
                <h2 className="text-sm font-black text-slate-400 uppercase tracking-wider">Direct Controller Panel</h2>
                <div className="grid grid-cols-3 gap-3">
                  <button onClick={() => triggerControl("Start Learning", "start-learning", { currentLearning: "Continual Learning", nextAction: "Evaluate & Compare", status: "LEARNING..." }, "Learning cycle started.")} className="p-3 bg-emerald-600 hover:bg-emerald-700 text-white rounded-xl text-xs font-bold transition">▶ Start</button>
                  <button onClick={() => triggerControl("Pause", "pause-learning", { currentLearning: "PAUSED", status: "PAUSED" }, "Learning paused.")} className="p-3 bg-slate-700 hover:bg-slate-600 text-white rounded-xl text-xs font-bold transition">⏸ Pause</button>
                  <button onClick={() => triggerControl("Stop", "stop-learning", { currentLearning: "STOPPED", status: "STOPPED" }, "Learning stopped.")} className="p-3 bg-red-600 hover:bg-red-700 text-white rounded-xl text-xs font-bold transition">⏹ Stop</button>
                </div>

                <div className="grid grid-cols-2 gap-3 pt-3">
                  <button onClick={() => triggerControl("Test Model", "test-model", { currentLearning: "Evaluating Model...", status: "TESTING..." }, "Testing initiated.")} className="p-3 bg-slate-800 hover:bg-slate-700 text-slate-200 rounded-xl text-xs font-bold transition border border-slate-700">🧪 Test Model</button>
                  <button onClick={() => triggerControl("Run Forgetting Test", "run-forgetting-test", { currentLearning: "Forgetting Detection...", status: "CHECKING..." }, "Forgetting test started.")} className="p-3 bg-slate-800 hover:bg-slate-700 text-slate-200 rounded-xl text-xs font-bold transition border border-slate-700">🔍 Forgetting Test</button>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* PAGE DELEGATIONS */}
        {activeTab === 'workspace' && <AIWorkspacePage />}
        {activeTab === 'memory' && <MemoryPage />}
        {activeTab === 'ingestion' && <DatasetManagerPage />}
        {activeTab === 'quality' && <QualityPoisoningPage />}
        {activeTab === 'gaps' && <GapResearchPage />}
        {activeTab === 'training' && <TrainingLabPage />}
        {activeTab === 'registry' && <RegistryPromotionPage />}
        {activeTab === 'testing' && <TestingLabPage />}
        {activeTab === 'observability' && <ObservabilityPage />}

      </main>
    </div>
  );
};
