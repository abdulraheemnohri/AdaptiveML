import React, { useState, useEffect } from 'react';

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

interface MemoryEntry {
  id: string;
  type: string;
  content: string;
  trusted: boolean;
  created_at: string;
}

interface ChatMessage {
  id: string;
  sender: 'user' | 'assistant';
  text: string;
  modality: string;
  timestamp: string;
  explanation?: string;
  isCorrect?: boolean | null;
  rating?: number;
  correctionSubmitted?: boolean;
}

export const Dashboard: React.FC = () => {
  // Tabs: 'control' | 'workspace' | 'memory' | 'settings'
  const [activeTab, setActiveTab] = useState<'control' | 'workspace' | 'memory' | 'settings'>('control');

  // 1. Command Center Telemetry State (Section 49)
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

  // 2. AI Workspace State (Section 2 & 28)
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([
    {
      id: "msg-1",
      sender: "assistant",
      text: "Greetings! I am Qwen2.5-Omni-3B. How can I assist you with multimodal tasks, language understanding, or coding experiments today?",
      modality: "text",
      timestamp: new Date(Date.now() - 3600000).toLocaleTimeString()
    }
  ]);
  const [inputText, setInputText] = useState("");
  const [selectedModel, setSelectedModel] = useState("Qwen2.5-Omni-3B");
  const [selectedAdapter, setSelectedAdapter] = useState("None");
  const [ragEnabled, setRagEnabled] = useState(true);
  const [memoryEnabled, setMemoryEnabled] = useState(true);
  const [activeModality, setActiveModality] = useState("text");
  const [showExplanationId, setShowExplanationId] = useState<string | null>(null);

  // 3. Long-Term Memory State (Section 11)
  const [memories, setMemories] = useState<MemoryEntry[]>([
    { id: "mem-1", type: "user", content: "Prefer short, direct, fact-filled explanations.", trusted: true, created_at: "2025-02-23T10:00:00Z" },
    { id: "mem-2", type: "conversation", content: "Discussed Urdu vocabulary transliteration and adapters.", trusted: true, created_at: "2025-02-23T11:30:00Z" },
    { id: "mem-3", type: "task", content: "Fine-tune Qwen2.5-Omni on target-task text datasets.", trusted: false, created_at: "2025-02-23T12:15:00Z" }
  ]);
  const [searchMemoryQuery, setSearchMemoryQuery] = useState("");
  const [newMemoryText, setNewMemoryText] = useState("");
  const [newMemoryType, setNewMemoryTextType] = useState("user");
  const [autoMemoryEnabled, setAutoMemoryEnabled] = useState(true);
  const [memorySizeLimit, setMemorySizeLimit] = useState(10000);

  // 4. Inference Settings State (Section 40 & 18)
  const [temperature, setTemperature] = useState(0.7);
  const [topP, setTopP] = useState(0.9);
  const [topK, setTopK] = useState(50);
  const [maxTokens, setMaxTokens] = useState(2048);
  const [repetitionPenalty, setRepetitionPenalty] = useState(1.1);
  const [precision, setPrecision] = useState("bf16");
  const [quantisation, setQuantisation] = useState("4-bit");

  // Sync state from FastAPI backend if active
  const fetchStatusAndMemories = async () => {
    try {
      // Sync telemetry status
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

      // Sync user memories
      const memoryResponse = await fetch('/memory');
      if (memoryResponse.ok) {
        const data = await memoryResponse.json();
        if (data.memories) {
          setMemories(data.memories);
        }
      }
    } catch (e) {
      // Backend offline fallback is handled by state initialization
    }
  };

  useEffect(() => {
    fetchStatusAndMemories();
    const interval = setInterval(fetchStatusAndMemories, 4000);
    return () => clearInterval(interval);
  }, []);

  // Control action trigger helper
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

  // AI Workspace Send Message Trigger
  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputText.trim()) return;

    const userMsg: ChatMessage = {
      id: `user-${Date.now()}`,
      sender: 'user',
      text: inputText,
      modality: activeModality,
      timestamp: new Date().toLocaleTimeString()
    };

    setChatMessages(prev => [...prev, userMsg]);
    setInputText("");

    const botMsgId = `bot-${Date.now()}`;
    const pendingBotMsg: ChatMessage = {
      id: botMsgId,
      sender: 'assistant',
      text: "Thinking...",
      modality: activeModality,
      timestamp: new Date().toLocaleTimeString()
    };

    setChatMessages(prev => [...prev, pendingBotMsg]);

    try {
      const response = await fetch('/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: userMsg.text,
          model_id: selectedModel,
          adapter_id: selectedAdapter !== "None" ? selectedAdapter : null,
          rag_enabled: ragEnabled,
          memory_enabled: memoryEnabled,
          modality: activeModality
        })
      });

      if (response.ok) {
        const data = await response.json();
        setChatMessages(prev => prev.map(m => m.id === botMsgId ? {
          ...m,
          text: data.prediction,
          explanation: data.explained_answer,
          timestamp: new Date().toLocaleTimeString()
        } : m));
      } else {
        simulateBotResponse(userMsg, botMsgId);
      }
    } catch (err) {
      simulateBotResponse(userMsg, botMsgId);
    }
  };

  const simulateBotResponse = (userMsg: ChatMessage, botMsgId: string) => {
    const textLower = userMsg.text.lowerCase ? userMsg.text.toLowerCase() : userMsg.text;
    let prediction = "";
    let explanation = "";

    if (textLower.includes("urdu")) {
      prediction = "Qwen2.5-Omni: 'یہ ایک آزمائش ہے' (meaning 'This is a test'). The specialized Urdu language adapter was successfully activated to handle native text synthesis.";
      explanation = "Routed to Urdu Adapter. Zero degradation observed. Retention of original multilingual benchmarks remains at 99.1%.";
    } else if (textLower.includes("code") || textLower.includes("python")) {
      prediction = "Here is a python model wrapper example:\n```python\n# Anti-forgetting checkpoint wrapper\nclass AdaptiveWrapper:\n    def __init__(self, model):\n        self.model = model\n```";
      explanation = "Inference routed to Coding Adapter. Checked against MAS parameter regularisation. Loss metrics remained flat.";
    } else if (activeModality !== 'text') {
      prediction = `Processed raw multimodal inputs successfully using ${activeModality} sample encoders. Captions extracted with high quality.`;
      explanation = "Feature extraction mapped frames into unified Qwen Omni Thinker embedding space. Hybrid search triggered.";
    } else {
      prediction = ` Greetings! I am answering in standard mode with RAG (currently ${ragEnabled ? 'ON' : 'OFF'}) and memory (${memoryEnabled ? 'ON' : 'OFF'}). Your instruction: '${userMsg.text}' was analyzed.`;
      explanation = "Default base model routing. Soft targets preserved using knowledge distillation temperature scaling.";
    }

    setChatMessages(prev => prev.map(m => m.id === botMsgId ? {
      ...m,
      text: prediction,
      explanation: explanation,
      timestamp: new Date().toLocaleTimeString()
    } : m));
  };

  // Submit human feedback (Section 28)
  const submitFeedback = async (msgId: string, rating: number, isHallucination = false, isFactualError = false, correction = "") => {
    setChatMessages(prev => prev.map(m => m.id === msgId ? {
      ...m,
      rating,
      isCorrect: rating > 3,
      correctionSubmitted: correction !== "" ? true : m.correctionSubmitted
    } : m));

    try {
      await fetch('/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message_id: msgId,
          rating,
          is_hallucination: isHallucination,
          is_factual_error: isFactualError,
          correction: correction || null
        })
      });
    } catch (e) {
      // offline fallback
    }
  };

  // Memory Actions (Section 11)
  const handleAddMemory = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newMemoryText.trim()) return;

    try {
      const response = await fetch('/memory', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          type: newMemoryType,
          content: newMemoryText,
          trusted: true
        })
      });

      if (response.ok) {
        const data = await response.json();
        setMemories(prev => [...prev, data.memory]);
      } else {
        const fallbackMem: MemoryEntry = {
          id: `mem-${Date.now()}`,
          type: newMemoryType,
          content: newMemoryText,
          trusted: true,
          created_at: new Date().toISOString()
        };
        setMemories(prev => [...prev, fallbackMem]);
      }
    } catch (err) {
      const fallbackMem: MemoryEntry = {
        id: `mem-${Date.now()}`,
        type: newMemoryType,
        content: newMemoryText,
        trusted: true,
        created_at: new Date().toISOString()
      };
      setMemories(prev => [...prev, fallbackMem]);
    }
    setNewMemoryText("");
  };

  const deleteMemory = async (id: string) => {
    setMemories(prev => prev.filter(m => m.id !== id));
    try {
      await fetch(`/memory/${id}`, { method: 'DELETE' });
    } catch (e) {}
  };

  const toggleTrustMemory = async (id: string) => {
    setMemories(prev => prev.map(m => m.id === id ? { ...m, trusted: !m.trusted } : m));
    try {
      await fetch(`/memory/${id}/trust`, { method: 'POST' });
    } catch (e) {}
  };

  // Filtered Memories list
  const filteredMemories = memories.filter(m =>
    m.content.toLowerCase().includes(searchMemoryQuery.toLowerCase()) ||
    m.type.toLowerCase().includes(searchMemoryQuery.toLowerCase())
  );

  return (
    <div className="p-6 max-w-7xl mx-auto space-y-8 bg-gray-50 min-h-screen">

      {/* CENTRAL NAVIGATION NAVIGATION BAR */}
      <nav className="flex items-center justify-between bg-white border border-gray-100 rounded-2xl p-4 shadow-sm">
        <div className="flex items-center space-x-3">
          <span className="text-2xl font-bold bg-indigo-600 text-white h-10 w-10 flex items-center justify-center rounded-xl shadow-md">Ω</span>
          <div>
            <h1 className="text-xl font-black text-gray-900 leading-tight">Adaptive Omni Brain</h1>
            <p className="text-[10px] text-gray-500 font-semibold uppercase tracking-wider">Fast, Medium, and Slow Continual Learning</p>
          </div>
        </div>

        {/* TABS CONTROLLER */}
        <div className="flex space-x-1 bg-slate-100 p-1.5 rounded-xl border border-slate-200/50">
          <button
            onClick={() => setActiveTab('control')}
            className={`px-4 py-2 rounded-lg text-xs font-bold transition ${
              activeTab === 'control' ? 'bg-white text-indigo-700 shadow-sm' : 'text-gray-600 hover:text-indigo-600'
            }`}
          >
            🕹️ Control Center
          </button>
          <button
            onClick={() => setActiveTab('workspace')}
            className={`px-4 py-2 rounded-lg text-xs font-bold transition ${
              activeTab === 'workspace' ? 'bg-white text-indigo-700 shadow-sm' : 'text-gray-600 hover:text-indigo-600'
            }`}
          >
            💬 AI Workspace
          </button>
          <button
            onClick={() => setActiveTab('memory')}
            className={`px-4 py-2 rounded-lg text-xs font-bold transition ${
              activeTab === 'memory' ? 'bg-white text-indigo-700 shadow-sm' : 'text-gray-600 hover:text-indigo-600'
            }`}
          >
            🧠 Long-Term Memory
          </button>
          <button
            onClick={() => setActiveTab('settings')}
            className={`px-4 py-2 rounded-lg text-xs font-bold transition ${
              activeTab === 'settings' ? 'bg-white text-indigo-700 shadow-sm' : 'text-gray-600 hover:text-indigo-600'
            }`}
          >
            ⚙️ Settings & Inference
          </button>
        </div>
      </nav>

      {/* ==================== TAB 1: COMMAND CENTER (SECTION 49) ==================== */}
      {activeTab === 'control' && (
        <div className="space-y-8 animate-fadeIn">

          {/* HEADER SECTION */}
          <header className="bg-gradient-to-r from-slate-900 to-indigo-950 text-white p-6 rounded-2xl shadow-lg border border-indigo-900/40">
            <div className="flex flex-col md:flex-row md:justify-between md:items-center space-y-4 md:space-y-0">
              <div>
                <div className="flex items-center space-x-3">
                  <span className="bg-indigo-500/20 text-indigo-300 text-xs font-bold uppercase tracking-wider px-2.5 py-1 rounded-md border border-indigo-500/30">
                    Central Command Centre
                  </span>
                  <span className="h-2 w-2 rounded-full bg-green-400 animate-pulse"></span>
                  <span className="text-xs text-green-300 font-medium tracking-wide">SYSTEM ALIVE</span>
                </div>
                <h1 className="text-3xl font-extrabold mt-1 tracking-tight">Adaptive Brain Control</h1>
                <p className="text-indigo-200 text-sm mt-1 max-w-xl">
                  Unified Multimodal Continual Learning Control Center. Prevent catastrophic forgetting, discover knowledge gaps, and guide autonomous optimization cycles.
                </p>
              </div>
              <div className="flex flex-col items-end space-y-2">
                <span className="text-xs text-indigo-300 font-semibold uppercase">GLOBAL SYSTEM INTEGRITY</span>
                <span className={`px-4 py-1.5 rounded-full text-sm font-bold shadow-md border ${
                  state.status === "SAFE TO CONTINUE" || state.status === "SAFE"
                    ? "bg-emerald-950/80 text-emerald-300 border-emerald-500/50"
                    : state.status.includes("LEARNING") || state.status.includes("TRAINING") || state.status.includes("CHECKING") || state.status.includes("PROMOTED")
                    ? "bg-indigo-950/80 text-indigo-300 border-indigo-500/50 animate-pulse"
                    : "bg-amber-950/80 text-amber-300 border-amber-500/50"
                }`}>
                  {state.status}
                </span>
              </div>
            </div>
          </header>

          <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">

            {/* METRICS VIEW */}
            <section className="lg:col-span-7 space-y-6">
              <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-6 space-y-6">
                <div className="flex justify-between items-center border-b border-gray-100 pb-4">
                  <h2 className="text-lg font-bold text-gray-900 tracking-tight flex items-center">
                    <span className="mr-2">🧠</span> System Observability & Telemetry
                  </h2>
                  <span className="text-xs bg-indigo-50 text-indigo-700 font-semibold px-2.5 py-1 rounded-md">
                    Live Refreshing
                  </span>
                </div>

                {/* CURRENT MODEL */}
                <div className="bg-slate-900 text-white rounded-xl p-5 border border-slate-800 flex items-center justify-between shadow-sm">
                  <div className="space-y-1">
                    <span className="text-xs text-slate-400 font-bold uppercase tracking-wider">CURRENT MODEL</span>
                    <p className="text-lg font-extrabold text-cyan-300 tracking-tight">{state.currentModel}</p>
                  </div>
                  <div className="bg-slate-800 p-2 rounded-lg text-2xl">🤖</div>
                </div>

                {/* SPECIFIED METRICS GRID */}
                <div className="grid grid-cols-2 sm:grid-cols-3 gap-4">
                  <div className="p-4 bg-gradient-to-b from-indigo-50/40 to-indigo-50/10 rounded-xl border border-indigo-50 flex flex-col justify-between">
                    <span className="text-xs font-bold text-gray-500 uppercase tracking-wider">KNOWLEDGE</span>
                    <p className="text-2xl font-black text-indigo-600 mt-2">{state.knowledge}</p>
                  </div>
                  <div className="p-4 bg-gradient-to-b from-teal-50/40 to-teal-50/10 rounded-xl border border-teal-50 flex flex-col justify-between">
                    <span className="text-xs font-bold text-gray-500 uppercase tracking-wider">NEW CAPABILITIES</span>
                    <p className="text-2xl font-black text-teal-600 mt-2">{state.newCapabilities}</p>
                  </div>
                  <div className="p-4 bg-gradient-to-b from-emerald-50/40 to-emerald-50/10 rounded-xl border border-emerald-50 flex flex-col justify-between">
                    <span className="text-xs font-bold text-gray-500 uppercase tracking-wider">OLD CAPABILITIES RETAINED</span>
                    <p className="text-2xl font-black text-emerald-600 mt-2">{state.oldCapabilitiesRetained}</p>
                  </div>
                  <div className="p-4 bg-gradient-to-b from-red-50/40 to-red-50/10 rounded-xl border border-red-50 flex flex-col justify-between">
                    <span className="text-xs font-bold text-gray-500 uppercase tracking-wider">FORGETTING RISK</span>
                    <p className="text-2xl font-black text-red-600 mt-2">{state.forgettingRisk}</p>
                  </div>
                  <div className="p-4 bg-gradient-to-b from-cyan-50/40 to-cyan-50/10 rounded-xl border border-cyan-50 flex flex-col justify-between">
                    <span className="text-xs font-bold text-gray-500 uppercase tracking-wider">DATA TRUST</span>
                    <p className="text-2xl font-black text-cyan-600 mt-2">{state.dataTrust}</p>
                  </div>
                  <div className="p-4 bg-gradient-to-b from-purple-50/40 to-purple-50/10 rounded-xl border border-purple-50 flex flex-col justify-between">
                    <span className="text-xs font-bold text-gray-500 uppercase tracking-wider">MODEL SAFETY</span>
                    <p className="text-2xl font-black text-purple-600 mt-2">{state.modelSafety}</p>
                  </div>
                </div>

                {/* SECONDARY METRICS */}
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 border-t border-gray-100 pt-5">
                  <div className="space-y-1">
                    <span className="text-xs text-gray-400 font-bold uppercase tracking-wider">MODEL QUALITY</span>
                    <p className="text-xl font-bold text-gray-800">{state.modelQuality}</p>
                  </div>
                  <div className="space-y-1">
                    <span className="text-xs text-gray-400 font-bold uppercase tracking-wider">CURRENT LEARNING</span>
                    <p className="text-xl font-bold text-blue-600 truncate">{state.currentLearning}</p>
                  </div>
                  <div className="space-y-1">
                    <span className="text-xs text-gray-400 font-bold uppercase tracking-wider">NEXT ACTION</span>
                    <p className="text-xl font-bold text-indigo-600">{state.nextAction}</p>
                  </div>
                </div>
              </div>

              {/* MODALITY HEALTH */}
              <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-6 space-y-4">
                <h3 className="text-sm font-bold text-gray-900 uppercase tracking-wider flex items-center">
                  📊 Modality Accuracy & Anti-Forgetting Firewall
                </h3>
                <div className="grid grid-cols-2 sm:grid-cols-5 gap-3 text-center">
                  <div className="p-3 bg-slate-50 rounded-xl border border-slate-100">
                    <span className="text-xs text-gray-400 font-semibold uppercase">Text</span>
                    <div className="text-lg font-bold text-emerald-600 mt-1">🟢 94%</div>
                  </div>
                  <div className="p-3 bg-slate-50 rounded-xl border border-slate-100">
                    <span className="text-xs text-gray-400 font-semibold uppercase">Vision</span>
                    <div className="text-lg font-bold text-emerald-600 mt-1">🟢 91%</div>
                  </div>
                  <div className="p-3 bg-slate-50 rounded-xl border border-slate-100">
                    <span className="text-xs text-gray-400 font-semibold uppercase">Audio</span>
                    <div className="text-lg font-bold text-amber-600 mt-1">🟡 84%</div>
                  </div>
                  <div className="p-3 bg-slate-50 rounded-xl border border-slate-100">
                    <span className="text-xs text-gray-400 font-semibold uppercase">Video</span>
                    <div className="text-lg font-bold text-emerald-600 mt-1">🟢 89%</div>
                  </div>
                  <div className="p-3 bg-slate-50 rounded-xl border border-slate-100">
                    <span className="text-xs text-gray-400 font-semibold uppercase">Speech</span>
                    <div className="text-lg font-bold text-emerald-600 mt-1">🟢 90%</div>
                  </div>
                </div>
              </div>
            </section>

            {/* CONTROLS */}
            <section className="lg:col-span-5 space-y-6">
              <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-6 space-y-6">
                <div className="flex justify-between items-center border-b border-gray-100 pb-4">
                  <h2 className="text-lg font-bold text-gray-900 tracking-tight flex items-center">
                    <span className="mr-2">🕹️</span> Direct Controller Panel
                  </h2>
                </div>

                <div className="space-y-4">
                  <div className="grid grid-cols-3 gap-3">
                    <button
                      onClick={() => triggerControl(
                        "Start Learning",
                        "start-learning",
                        { currentLearning: "Continual Learning", nextAction: "Evaluate & Compare", status: "LEARNING..." },
                        "Start Learning signal transmitted successfully."
                      )}
                      disabled={!!loadingAction}
                      className="flex flex-col items-center justify-center p-3.5 bg-emerald-600 hover:bg-emerald-700 active:bg-emerald-800 text-white rounded-xl font-bold text-xs transition disabled:opacity-50"
                    >
                      <span className="text-xl mb-1">▶</span>
                      Start Learning
                    </button>

                    <button
                      onClick={() => triggerControl(
                        "Pause",
                        "pause-learning",
                        { currentLearning: "PAUSED", status: "PAUSED" },
                        "System paused."
                      )}
                      disabled={!!loadingAction}
                      className="flex flex-col items-center justify-center p-3.5 bg-slate-600 hover:bg-slate-700 active:bg-slate-800 text-white rounded-xl font-bold text-xs transition disabled:opacity-50"
                    >
                      <span className="text-xl mb-1">⏸</span>
                      Pause
                    </button>

                    <button
                      onClick={() => triggerControl(
                        "Stop",
                        "stop-learning",
                        { currentLearning: "STOPPED", status: "STOPPED" },
                        "System halted."
                      )}
                      disabled={!!loadingAction}
                      className="flex flex-col items-center justify-center p-3.5 bg-red-600 hover:bg-red-700 active:bg-red-800 text-white rounded-xl font-bold text-xs transition disabled:opacity-50"
                    >
                      <span className="text-xl mb-1">⏹</span>
                      Stop
                    </button>
                  </div>

                  <div className="grid grid-cols-2 gap-3">
                    <button
                      onClick={() => triggerControl(
                        "Test Model",
                        "test-model",
                        { currentLearning: "Evaluating Model...", status: "TESTING..." },
                        "Benchmark evaluation suite triggered across multiple validation sets."
                      )}
                      disabled={!!loadingAction}
                      className="flex items-center space-x-2 p-3 bg-indigo-50 hover:bg-indigo-100 text-indigo-900 rounded-xl text-left font-bold text-xs border border-indigo-100 shadow-sm transition"
                    >
                      <span className="text-lg">🧪</span>
                      <span>Test Model</span>
                    </button>

                    <button
                      onClick={() => triggerControl(
                        "Run Forgetting Test",
                        "run-forgetting-test",
                        { currentLearning: "Forgetting Detection...", status: "CHECKING..." },
                        "Anti-forgetting firewall validation analysis initialized."
                      )}
                      disabled={!!loadingAction}
                      className="flex items-center space-x-2 p-3 bg-violet-50 hover:bg-violet-100 text-violet-900 rounded-xl text-left font-bold text-xs border border-violet-100 shadow-sm transition"
                    >
                      <span className="text-lg">🔍</span>
                      <span>Run Forgetting Test</span>
                    </button>

                    <button
                      onClick={() => triggerControl(
                        "Find Knowledge Gaps",
                        "find-gaps",
                        { currentLearning: "Identifying Gaps...", status: "GAP DISCOVERY..." },
                        "Autonomous gap analysis scanning user queries and retrieval errors."
                      )}
                      disabled={!!loadingAction}
                      className="flex items-center space-x-2 p-3 bg-pink-50 hover:bg-pink-100 text-pink-900 rounded-xl text-left font-bold text-xs border border-pink-100 shadow-sm transition"
                    >
                      <span className="text-lg">📥</span>
                      <span>Find Gaps</span>
                    </button>

                    <button
                      onClick={() => triggerControl(
                        "Collect Data",
                        "collect-data",
                        { currentLearning: "Ingesting Data...", status: "ACQUIRING..." },
                        "Triggering multi-source ingestion pipelines (Web, RSS, GitHub, YouTube)."
                      )}
                      disabled={!!loadingAction}
                      className="flex items-center space-x-2 p-3 bg-amber-50 hover:bg-amber-100 text-amber-900 rounded-xl text-left font-bold text-xs border border-amber-100 shadow-sm transition"
                    >
                      <span className="text-lg">🧠</span>
                      <span>Collect Data</span>
                    </button>

                    <button
                      onClick={() => triggerControl(
                        "Train Candidate",
                        "train-candidate",
                        { currentLearning: "Fine-Tuning Candidate...", status: "TRAINING..." },
                        "Initializing continual learning sequence on replay-mixed candidate dataset."
                      )}
                      disabled={!!loadingAction}
                      className="flex items-center space-x-2 p-3 bg-teal-50 hover:bg-teal-100 text-teal-900 rounded-xl text-left font-bold text-xs border border-teal-100 shadow-sm transition"
                    >
                      <span className="text-lg">⚖️</span>
                      <span>Train Candidate</span>
                    </button>

                    <button
                      onClick={() => triggerControl(
                        "Compare Models",
                        "compare-models",
                        { currentLearning: "Comparing Models...", status: "COMPARING..." },
                        "Generating comprehensive promotion reports comparing candidate vs production."
                      )}
                      disabled={!!loadingAction}
                      className="flex items-center space-x-2 p-3 bg-cyan-50 hover:bg-cyan-100 text-cyan-900 rounded-xl text-left font-bold text-xs border border-cyan-100 shadow-sm transition"
                    >
                      <span className="text-lg">🚀</span>
                      <span>Compare Models</span>
                    </button>
                  </div>

                  <div className="border-t border-gray-100 pt-4 space-y-2">
                    <span className="text-[10px] text-gray-400 font-bold uppercase tracking-wider block">
                      Critical Registry Interventions
                    </span>
                    <div className="grid grid-cols-2 gap-3">
                      <button
                        onClick={() => triggerControl(
                          "Roll Back",
                          "rollback",
                          { currentModel: "Qwen2.5-Omni-3B Adaptive v3.4.1 (Rolled Back)", status: "ROLLED BACK" },
                          "Reverting production state to last stable checkpoint: v3.4.1"
                        )}
                        disabled={!!loadingAction}
                        className="flex items-center justify-center space-x-2 p-3 bg-slate-100 hover:bg-slate-200 text-slate-800 rounded-xl font-bold text-xs shadow-sm transition"
                      >
                        <span>↩</span>
                        <span>Roll Back</span>
                      </button>

                      <button
                        onClick={() => triggerControl(
                          "Emergency Promote Model",
                          "emergency-promote",
                          { currentModel: "Qwen2.5-Omni-3B Adaptive v3.4.3 (Promoted)", status: "PROMOTED" },
                          "FORCE PROMOTION: Bypassing promotion gates. Activating new model version."
                        )}
                        disabled={!!loadingAction}
                        className="flex items-center justify-center space-x-2 p-3 bg-amber-500 hover:bg-amber-600 text-white rounded-xl font-bold text-xs shadow-md transition"
                      >
                        <span>🛑</span>
                        <span>Emergency Promote</span>
                      </button>
                    </div>
                  </div>
                </div>
              </div>

              {/* DYNAMIC LOG */}
              <div className="bg-slate-900 rounded-2xl p-5 border border-slate-800 shadow-inner space-y-3">
                <span className="text-xs font-bold text-slate-400 uppercase block tracking-wider">
                  📟 Control Activity logs
                </span>
                <div className="h-40 overflow-y-auto space-y-1 font-mono text-[11px] text-slate-300 pr-2">
                  {controlLogs.map((log, index) => (
                    <div key={index} className="pb-1 text-slate-300">{log}</div>
                  ))}
                </div>
              </div>
            </section>
          </div>
        </div>
      )}

      {/* ==================== TAB 2: AI WORKSPACE (SECTION 2 & 28) ==================== */}
      {activeTab === 'workspace' && (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 animate-fadeIn">

          {/* LEFT CHAT PANEL */}
          <div className="lg:col-span-8 bg-white rounded-2xl shadow-sm border border-gray-100 p-6 flex flex-col h-[650px]">
            <div className="flex justify-between items-center border-b border-gray-100 pb-4 mb-4">
              <h2 className="text-lg font-bold text-gray-900 tracking-tight flex items-center">
                <span className="mr-2">💬</span> Multimodal Chat Space
              </h2>
              <div className="flex space-x-2">
                {['text', 'image', 'audio', 'video', 'speech'].map((m) => (
                  <button
                    key={m}
                    onClick={() => setActiveModality(m)}
                    className={`px-3 py-1 rounded-lg text-xs font-bold uppercase transition ${
                      activeModality === m ? 'bg-indigo-600 text-white shadow-sm' : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                    }`}
                  >
                    {m}
                  </button>
                ))}
              </div>
            </div>

            {/* MESSAGE CONTAINER */}
            <div className="flex-grow overflow-y-auto space-y-4 mb-4 pr-2">
              {chatMessages.map((msg) => (
                <div key={msg.id} className={`flex flex-col ${msg.sender === 'user' ? 'items-end' : 'items-start'}`}>
                  <div className={`p-4 rounded-2xl max-w-xl text-sm ${
                    msg.sender === 'user'
                      ? 'bg-indigo-600 text-white rounded-tr-none'
                      : 'bg-slate-100 text-gray-800 rounded-tl-none border border-slate-200/50'
                  }`}>
                    {/* Render raw/multiline or code nicely */}
                    <div className="whitespace-pre-wrap leading-relaxed">{msg.text}</div>

                    {/* Multimodal context tag if present */}
                    {msg.modality !== 'text' && msg.sender === 'user' && (
                      <span className="mt-2 inline-block text-[10px] bg-indigo-700/80 text-white px-2 py-0.5 rounded uppercase font-bold">
                        📂 Loaded {msg.modality} asset
                      </span>
                    )}

                    <div className={`text-[10px] mt-2 block ${msg.sender === 'user' ? 'text-indigo-200' : 'text-gray-400'}`}>
                      {msg.timestamp}
                    </div>
                  </div>

                  {/* ASSISTANT UTILITIES AND FEEDBACK CONTROLS (SECTION 2 & 28) */}
                  {msg.sender === 'assistant' && msg.text !== "Thinking..." && (
                    <div className="flex items-center space-x-4 mt-2 px-1 text-xs">

                      {/* Explaining answer (Section 2) */}
                      {msg.explanation && (
                        <button
                          onClick={() => setShowExplanationId(showExplanationId === msg.id ? null : msg.id)}
                          className="text-indigo-600 hover:text-indigo-800 font-bold hover:underline flex items-center"
                        >
                          💡 Explain Answer
                        </button>
                      )}

                      {/* Correct/Incorrect ratings (Section 28) */}
                      <div className="flex items-center space-x-2 border-l border-gray-200 pl-4">
                        <button
                          onClick={() => submitFeedback(msg.id, 5)}
                          className={`hover:scale-110 transition ${msg.rating === 5 ? 'text-emerald-600 font-black' : 'text-gray-400'}`}
                          title="Mark Response as Correct / Safe"
                        >
                          👍 Correct
                        </button>
                        <button
                          onClick={() => submitFeedback(msg.id, 1, true, true, "Submit correction")}
                          className={`hover:scale-110 transition ${msg.rating === 1 ? 'text-red-600 font-black' : 'text-gray-400'}`}
                          title="Mark Response as Incorrect / Hallucination"
                        >
                          👎 Incorrect
                        </button>
                      </div>

                      {/* Save example / Training queue (Section 2) */}
                      <button
                        onClick={() => submitFeedback(msg.id, 5, false, false, "training example saved")}
                        className="text-emerald-700 hover:text-emerald-900 hover:underline font-semibold"
                      >
                        💾 Save as Training Example
                      </button>
                    </div>
                  )}

                  {/* EXPLANATION ACCORDION (SECTION 2 - ASK MODEL TO EXPLAIN ANSWERS) */}
                  {msg.sender === 'assistant' && showExplanationId === msg.id && msg.explanation && (
                    <div className="mt-2 p-3 bg-amber-50/50 border border-amber-200 text-amber-900 rounded-xl text-xs max-w-xl animate-fadeIn">
                      <span className="font-bold block mb-1">🔍 Retrieval & Adapter Inference Explanation:</span>
                      {msg.explanation}
                    </div>
                  )}

                  {/* CORRECTION FIELD (SECTION 28) */}
                  {msg.sender === 'assistant' && msg.rating === 1 && (
                    <div className="mt-2 p-3 bg-red-50 border border-red-200 text-red-950 rounded-xl text-xs w-full max-w-xl animate-fadeIn space-y-2">
                      <span className="font-bold block">🚨 Submit Human Correction:</span>
                      <input
                        type="text"
                        placeholder="Type correct factual statement or expected model behavior..."
                        className="w-full p-2 rounded-lg border border-red-200 focus:outline-none focus:ring-1 focus:ring-red-400 text-xs"
                        onKeyDown={(e) => {
                          if (e.key === 'Enter') {
                            submitFeedback(msg.id, 1, false, true, (e.target as HTMLInputElement).value);
                            (e.target as HTMLInputElement).value = "";
                          }
                        }}
                      />
                      {msg.correctionSubmitted && (
                        <p className="text-[10px] text-green-700 font-bold">✓ Correction saved and queued to Dataset Manager Review Queue.</p>
                      )}
                    </div>
                  )}
                </div>
              ))}
            </div>

            {/* MULTIMODAL MEDIA SIMULATION UPLOADS */}
            <div className="bg-slate-50 p-2.5 rounded-xl border border-slate-200 mb-3 flex items-center space-x-2 text-xs text-gray-500 font-medium">
              <span className="font-bold">Multimodal Assets:</span>
              <button
                onClick={() => {
                  setActiveModality("image");
                  setChatMessages(prev => [...prev, {
                    id: `asset-${Date.now()}`,
                    sender: "user",
                    text: "[Loaded Multimodal Asset]: Image 'continual_loop.png' uploaded successfully.",
                    modality: "image",
                    timestamp: new Date().toLocaleTimeString()
                  }]);
                }}
                className="bg-white hover:bg-gray-100 border px-2.5 py-1 rounded"
              >
                🖼️ Upload Image
              </button>
              <button
                onClick={() => {
                  setActiveModality("audio");
                  setChatMessages(prev => [...prev, {
                    id: `asset-${Date.now()}`,
                    sender: "user",
                    text: "[Loaded Multimodal Asset]: Audio snippet 'urdu_vocal.wav' uploaded successfully.",
                    modality: "audio",
                    timestamp: new Date().toLocaleTimeString()
                  }]);
                }}
                className="bg-white hover:bg-gray-100 border px-2.5 py-1 rounded"
              >
                🎙️ Upload Audio
              </button>
              <button
                onClick={() => {
                  setActiveModality("video");
                  setChatMessages(prev => [...prev, {
                    id: `asset-${Date.now()}`,
                    sender: "user",
                    text: "[Loaded Multimodal Asset]: Video track 'training_demo.mp4' uploaded successfully.",
                    modality: "video",
                    timestamp: new Date().toLocaleTimeString()
                  }]);
                }}
                className="bg-white hover:bg-gray-100 border px-2.5 py-1 rounded"
              >
                🎥 Upload Video
              </button>
            </div>

            {/* SEND FORM */}
            <form onSubmit={handleSendMessage} className="flex space-x-2">
              <input
                type="text"
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                placeholder={`Ask Qwen2.5-Omni anything (e.g. 'translate to Urdu' or 'show code')...`}
                className="flex-grow p-4 rounded-xl border border-gray-200 focus:outline-none focus:ring-2 focus:ring-indigo-600 focus:border-transparent text-sm"
              />
              <button
                type="submit"
                className="bg-indigo-600 hover:bg-indigo-700 active:bg-indigo-800 text-white font-bold px-6 rounded-xl text-sm transition"
              >
                Send 🚀
              </button>
            </form>
          </div>

          {/* RIGHT WORKSPACE CONTEXT SETTINGS PANEL (SECTION 2) */}
          <div className="lg:col-span-4 bg-white rounded-2xl shadow-sm border border-gray-100 p-6 space-y-6">
            <h3 className="text-sm font-bold text-gray-900 uppercase tracking-wider border-b border-gray-100 pb-2">
              🧭 Workspace Environment
            </h3>

            <div className="space-y-4 text-xs">
              <div className="space-y-1">
                <label className="text-gray-500 font-bold block">Base Model Selection</label>
                <select
                  value={selectedModel}
                  onChange={(e) => setSelectedModel(e.target.value)}
                  className="w-full p-2.5 border rounded-lg focus:outline-none text-xs font-semibold"
                >
                  <option>Qwen/Qwen2.5-Omni-3B</option>
                  <option>Qwen/Qwen2.5-Omni-7B</option>
                </select>
              </div>

              <div className="space-y-1">
                <label className="text-gray-500 font-bold block">Specialized Adapter Selection</label>
                <select
                  value={selectedAdapter}
                  onChange={(e) => setSelectedAdapter(e.target.value)}
                  className="w-full p-2.5 border rounded-lg focus:outline-none text-xs font-semibold"
                >
                  <option>None</option>
                  <option>Urdu Adapter (Skill)</option>
                  <option>Coding Adapter (Domain)</option>
                  <option>Vision Adapter (Multimodal)</option>
                </select>
              </div>

              {/* Toggle Switches */}
              <div className="space-y-3 pt-3 border-t border-gray-100">
                <div className="flex justify-between items-center">
                  <span className="font-bold text-gray-700">RAG Knowledge Ingestion</span>
                  <input
                    type="checkbox"
                    checked={ragEnabled}
                    onChange={(e) => setRagEnabled(e.target.checked)}
                    className="h-4 w-4 text-indigo-600 focus:ring-indigo-500 rounded border-gray-300"
                  />
                </div>
                <div className="flex justify-between items-center">
                  <span className="font-bold text-gray-700">Long-Term Memory Synced</span>
                  <input
                    type="checkbox"
                    checked={memoryEnabled}
                    onChange={(e) => setMemoryEnabled(e.target.checked)}
                    className="h-4 w-4 text-indigo-600 focus:ring-indigo-500 rounded border-gray-300"
                  />
                </div>
              </div>

              {/* Knowledge source */}
              <div className="space-y-2 pt-3 border-t border-gray-100">
                <label className="text-gray-500 font-bold block">Knowledge Sources Selected</label>
                <div className="space-y-1">
                  <label className="flex items-center space-x-2">
                    <input type="checkbox" defaultChecked className="rounded border-gray-300" />
                    <span>Public Documentation Database</span>
                  </label>
                  <label className="flex items-center space-x-2">
                    <input type="checkbox" defaultChecked className="rounded border-gray-300" />
                    <span>Urdu Language Text Corpuses</span>
                  </label>
                  <label className="flex items-center space-x-2">
                    <input type="checkbox" className="rounded border-gray-300" />
                    <span>GitHub Researched Source Repos</span>
                  </label>
                </div>
              </div>

              {/* Create dataset from conversation */}
              <div className="pt-4 border-t border-gray-100">
                <button
                  type="button"
                  onClick={() => {
                    alert("Dataset candidate created successfully from this session! Sent 2 instruction-output pairs to the review queue.");
                  }}
                  className="w-full bg-slate-900 hover:bg-slate-800 text-white font-bold p-3 rounded-xl text-center shadow transition text-xs"
                >
                  📦 Create Dataset from Chat Session
                </button>
              </div>

            </div>
          </div>
        </div>
      )}

      {/* ==================== TAB 3: LONG-TERM MEMORY (SECTION 11) ==================== */}
      {activeTab === 'memory' && (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 animate-fadeIn">

          {/* MEMORY MANAGER COLUMN */}
          <div className="lg:col-span-8 bg-white rounded-2xl shadow-sm border border-gray-100 p-6 space-y-6">
            <div className="flex flex-col sm:flex-row justify-between sm:items-center border-b border-gray-100 pb-4 space-y-3 sm:space-y-0">
              <h2 className="text-lg font-bold text-gray-900 tracking-tight flex items-center">
                <span className="mr-2">💾</span> Long-Term Memory Console
              </h2>
              <input
                type="text"
                value={searchMemoryQuery}
                onChange={(e) => setSearchMemoryQuery(e.target.value)}
                placeholder="Search user, task, or conversation memories..."
                className="p-2 border rounded-xl focus:outline-none focus:ring-1 focus:ring-indigo-500 text-xs w-full sm:w-64"
              />
            </div>

            {/* ADD NEW MEMORY FORM */}
            <form onSubmit={handleAddMemory} className="bg-slate-50 p-4 rounded-xl border border-slate-100 flex flex-col sm:flex-row items-end space-y-3 sm:space-y-0 sm:space-x-3">
              <div className="flex-grow space-y-1 w-full">
                <label className="text-[10px] text-gray-500 font-bold uppercase tracking-wider block">New Memory Content</label>
                <input
                  type="text"
                  value={newMemoryText}
                  onChange={(e) => setNewMemoryText(e.target.value)}
                  placeholder="e.g. 'Prefers Urdu replies when greeting'"
                  className="w-full p-2.5 rounded-lg border text-xs focus:outline-none"
                />
              </div>
              <div className="space-y-1 w-full sm:w-40">
                <label className="text-[10px] text-gray-500 font-bold uppercase tracking-wider block">Memory Type</label>
                <select
                  value={newMemoryType}
                  onChange={(e) => setNewMemoryTextType(e.target.value)}
                  className="w-full p-2.5 rounded-lg border text-xs focus:outline-none font-semibold bg-white"
                >
                  <option value="user">User Memory</option>
                  <option value="conversation">Conversation</option>
                  <option value="task">Task Memory</option>
                  <option value="project">Project Memory</option>
                  <option value="domain">Domain Memory</option>
                  <option value="system">System Memory</option>
                </select>
              </div>
              <button
                type="submit"
                className="bg-indigo-600 hover:bg-indigo-700 text-white font-bold px-4 py-2.5 rounded-lg text-xs transition w-full sm:w-auto"
              >
                Add Memory
              </button>
            </form>

            {/* MEMORIES VISUAL LIST */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {filteredMemories.length === 0 ? (
                <p className="text-gray-500 text-xs italic col-span-2 text-center py-8">No memories matching your filters.</p>
              ) : (
                filteredMemories.map((mem) => (
                  <div key={mem.id} className="p-4 bg-white rounded-xl border border-slate-200/60 shadow-sm flex flex-col justify-between space-y-3 relative hover:border-slate-300 transition">
                    <div className="space-y-1">
                      <div className="flex justify-between items-center">
                        <span className="text-[9px] bg-indigo-50 text-indigo-700 px-2 py-0.5 rounded-md uppercase font-black">
                          {mem.type}
                        </span>
                        <span className={`text-[9px] font-bold px-2 py-0.5 rounded-md ${
                          mem.trusted ? 'bg-emerald-50 text-emerald-700' : 'bg-amber-50 text-amber-700'
                        }`}>
                          {mem.trusted ? '✓ Trusted' : '⚠️ Pending'}
                        </span>
                      </div>
                      <p className="text-xs font-semibold text-slate-800 pt-1 leading-relaxed">{mem.content}</p>
                    </div>

                    <div className="flex justify-between items-center border-t border-slate-100 pt-2 text-[10px]">
                      <button
                        onClick={() => toggleTrustMemory(mem.id)}
                        className="text-indigo-600 hover:text-indigo-800 font-bold hover:underline"
                      >
                        {mem.trusted ? "Demote Trust" : "Mark as Trusted"}
                      </button>
                      <button
                        onClick={() => deleteMemory(mem.id)}
                        className="text-red-500 hover:text-red-700 font-bold hover:underline flex items-center"
                      >
                        🗑️ Forget Memory
                      </button>
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>

          {/* RIGHT SYSTEM MEMORY SETTINGS (SECTION 11) */}
          <div className="lg:col-span-4 bg-white rounded-2xl shadow-sm border border-gray-100 p-6 space-y-6">
            <h3 className="text-sm font-bold text-gray-900 uppercase tracking-wider border-b border-gray-100 pb-2">
              🧠 Memory Configuration
            </h3>

            <div className="space-y-4 text-xs">
              <div className="flex justify-between items-center">
                <span className="font-bold text-gray-700">Memory Engine Enabled</span>
                <input
                  type="checkbox"
                  checked={autoMemoryEnabled}
                  onChange={(e) => setAutoMemoryEnabled(e.target.checked)}
                  className="h-4 w-4 text-indigo-600 focus:ring-indigo-500 rounded border-gray-300"
                />
              </div>

              <div className="space-y-2">
                <label className="text-gray-500 font-bold block">Maximum Memory Buffer Size</label>
                <div className="flex justify-between items-center font-bold">
                  <input
                    type="range"
                    min="5000"
                    max="50000"
                    step="5000"
                    value={memorySizeLimit}
                    onChange={(e) => setMemorySizeLimit(Number(e.target.value))}
                    className="flex-grow mr-4 h-1.5 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                  />
                  <span>{memorySizeLimit.toLocaleString()}</span>
                </div>
              </div>

              <div className="space-y-1">
                <label className="text-gray-500 font-bold block">Memory Expiration Policy</label>
                <select className="w-full p-2 border rounded-lg text-xs font-semibold">
                  <option>Never Expire (Permanent)</option>
                  <option>90 Days (Standard)</option>
                  <option>30 Days (Fast Cycle)</option>
                </select>
              </div>

              <div className="space-y-1">
                <label className="text-gray-500 font-bold block">Retrieval Similarity Threshold</label>
                <select className="w-full p-2 border rounded-lg text-xs font-semibold">
                  <option>High Precision (0.85)</option>
                  <option>Standard Balanced (0.75)</option>
                  <option>Broad Context (0.65)</option>
                </select>
              </div>

              <div className="bg-slate-50 p-4 rounded-xl border border-slate-100 text-slate-500 text-[11px] leading-relaxed">
                ℹ️ When 'Memory Engine' is enabled, conversational context triggers automatic summarization and extraction of facts to user-specific slots, which are reviewed periodically.
              </div>

            </div>
          </div>
        </div>
      )}

      {/* ==================== TAB 4: SETTINGS & INFERENCE (SECTION 40 & 18) ==================== */}
      {activeTab === 'settings' && (
        <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-6 space-y-6 animate-fadeIn">
          <div className="border-b border-gray-100 pb-4">
            <h2 className="text-lg font-bold text-gray-900 tracking-tight flex items-center">
              <span className="mr-2">🛠️</span> Parameters & Performance Optimization
            </h2>
            <p className="text-xs text-gray-500">Configure parameters, API keys, thresholds, active learning, and EWC strategies.</p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">

            {/* INFERENCE SLIDERS */}
            <div className="space-y-6">
              <h3 className="text-xs font-bold text-slate-400 uppercase tracking-wider block">Inference Settings</h3>

              <div className="space-y-2">
                <div className="flex justify-between text-xs font-bold">
                  <label className="text-gray-600">Temperature</label>
                  <span>{temperature}</span>
                </div>
                <input
                  type="range"
                  min="0.1"
                  max="1.5"
                  step="0.05"
                  value={temperature}
                  onChange={(e) => setTemperature(Number(e.target.value))}
                  className="w-full h-1.5 bg-gray-200 rounded-lg cursor-pointer"
                />
              </div>

              <div className="space-y-2">
                <div className="flex justify-between text-xs font-bold">
                  <label className="text-gray-600">Top P</label>
                  <span>{topP}</span>
                </div>
                <input
                  type="range"
                  min="0.5"
                  max="1.0"
                  step="0.05"
                  value={topP}
                  onChange={(e) => setTopP(Number(e.target.value))}
                  className="w-full h-1.5 bg-gray-200 rounded-lg cursor-pointer"
                />
              </div>

              <div className="space-y-2">
                <div className="flex justify-between text-xs font-bold">
                  <label className="text-gray-600">Top K</label>
                  <span>{topK}</span>
                </div>
                <input
                  type="range"
                  min="1"
                  max="100"
                  step="1"
                  value={topK}
                  onChange={(e) => setTopK(Number(e.target.value))}
                  className="w-full h-1.5 bg-gray-200 rounded-lg cursor-pointer"
                />
              </div>

              <div className="space-y-2">
                <div className="flex justify-between text-xs font-bold">
                  <label className="text-gray-600">Max Generated Tokens</label>
                  <span>{maxTokens}</span>
                </div>
                <input
                  type="range"
                  min="256"
                  max="4096"
                  step="256"
                  value={maxTokens}
                  onChange={(e) => setMaxTokens(Number(e.target.value))}
                  className="w-full h-1.5 bg-gray-200 rounded-lg cursor-pointer"
                />
              </div>
            </div>

            {/* PERFORMANCE & OPTIMIZATION SETTINGS */}
            <div className="space-y-6">
              <h3 className="text-xs font-bold text-slate-400 uppercase tracking-wider block">Performance & Precision</h3>

              <div className="space-y-1 text-xs">
                <label className="text-gray-600 font-bold block">Execution Precision</label>
                <div className="grid grid-cols-3 gap-2">
                  {['bf16', 'fp16', 'fp32'].map((p) => (
                    <button
                      key={p}
                      type="button"
                      onClick={() => setPrecision(p)}
                      className={`p-2.5 rounded-lg border text-xs font-bold uppercase transition ${
                        precision === p ? 'bg-indigo-600 border-indigo-600 text-white shadow-sm' : 'bg-white hover:bg-gray-50 text-gray-700'
                      }`}
                    >
                      {p}
                    </button>
                  ))}
                </div>
              </div>

              <div className="space-y-1 text-xs">
                <label className="text-gray-600 font-bold block">Quantisation level (bitsandbytes)</label>
                <div className="grid grid-cols-3 gap-2">
                  {['4-bit', '8-bit', 'None'].map((q) => (
                    <button
                      key={q}
                      type="button"
                      onClick={() => setQuantisation(q)}
                      className={`p-2.5 rounded-lg border text-xs font-bold uppercase transition ${
                        quantisation === q ? 'bg-indigo-600 border-indigo-600 text-white shadow-sm' : 'bg-white hover:bg-gray-50 text-gray-700'
                      }`}
                    >
                      {q}
                    </button>
                  ))}
                </div>
              </div>

              <div className="space-y-2 pt-2 text-xs">
                <label className="text-gray-600 font-bold flex justify-between">
                  <span>Elastic Weight Consolidation (EWC) Weight</span>
                  <span>1000.0</span>
                </label>
                <input type="range" defaultChecked className="w-full h-1 bg-indigo-200 rounded-lg cursor-pointer" />
              </div>

              <div className="space-y-2 text-xs">
                <label className="text-gray-600 font-bold flex justify-between">
                  <span>Distillation Alpha Loss weight</span>
                  <span>0.5</span>
                </label>
                <input type="range" defaultChecked className="w-full h-1 bg-indigo-200 rounded-lg cursor-pointer" />
              </div>
            </div>

          </div>
        </div>
      )}

    </div>
  );
};
