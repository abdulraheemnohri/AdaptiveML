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

export const Dashboard: React.FC = () => {
  // Master control state matching Section 49 spec exactly
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

  const [logs, setLogs] = useState<string[]>([
    "Brain Core initialized.",
    "Experience replay memory load status: [1.2M entries].",
    "Anti-forgetting firewall: Standard monitoring level.",
    "System status: SAFE TO CONTINUE"
  ]);

  const [loadingAction, setLoadingAction] = useState<string | null>(null);

  // Sync state from FastAPI backend if active
  const fetchStatus = async () => {
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
      // Backend not running/reachable, fallback to local state is already set
    }
  };

  useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, 3000);
    return () => clearInterval(interval);
  }, []);

  const triggerControl = async (
    actionName: string,
    endpoint: string,
    localStateUpdate: Partial<SystemState>,
    logMessage: string
  ) => {
    setLoadingAction(actionName);
    const timeStr = new Date().toLocaleTimeString();

    // Add pending log
    setLogs(prev => [`[${timeStr}] ⚙️ Triggering: ${actionName}...`, ...prev]);

    // Apply local state update instantly for fast feedback
    setState(prev => ({ ...prev, ...localStateUpdate }));

    try {
      // Attempt backend API post
      const response = await fetch(`/control/${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });

      if (response.ok) {
        const data = await response.json();
        if (data.state) {
          // Sync exact state from response
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
        setLogs(prev => [`[${timeStr}] ✅ ${logMessage} (Synced with API Server)`, ...prev]);
      } else {
        setLogs(prev => [`[${timeStr}] ℹ️ ${logMessage} (Offline Demo Mode)`, ...prev]);
      }
    } catch (error) {
      // Backend offline, fallback gracefully
      setLogs(prev => [`[${timeStr}] ℹ️ ${logMessage} (Offline Demo Mode)`, ...prev]);
    } finally {
      setTimeout(() => {
        setLoadingAction(null);
      }, 600);
    }
  };

  return (
    <div className="p-6 max-w-7xl mx-auto space-y-8 bg-gray-50 min-h-screen">

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

      {/* TWO COLUMN GRID */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">

        {/* LEFT COLUMN: SPECIFIED KPI SCREEN */}
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

            {/* Spec section 49 - CURRENT MODEL */}
            <div className="bg-slate-900 text-white rounded-xl p-5 border border-slate-800 flex items-center justify-between shadow-sm">
              <div className="space-y-1">
                <span className="text-xs text-slate-400 font-bold uppercase tracking-wider">CURRENT MODEL</span>
                <p className="text-lg font-extrabold text-cyan-300 tracking-tight">{state.currentModel}</p>
              </div>
              <div className="bg-slate-800 p-2 rounded-lg text-2xl">🤖</div>
            </div>

            {/* MAIN SPECIED METRICS GRID */}
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

            {/* SECONDARY SPECIED METRICS AND NEXT ACTIONS */}
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

          {/* SYSTEM HARDWARE & EXPERIMENT METRICS FROM CORE SPECS */}
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

        {/* RIGHT COLUMN: CORE DIRECT CONTROLS SECTION */}
        <section className="lg:col-span-5 space-y-6">
          <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-6 space-y-6">
            <div className="flex justify-between items-center border-b border-gray-100 pb-4">
              <h2 className="text-lg font-bold text-gray-900 tracking-tight flex items-center">
                <span className="mr-2">🕹️</span> Direct Controller Panel
              </h2>
              {loadingAction && (
                <div className="flex items-center space-x-1">
                  <div className="w-2 h-2 bg-indigo-600 rounded-full animate-bounce"></div>
                  <div className="w-2 h-2 bg-indigo-600 rounded-full animate-bounce [animation-delay:0.2s]"></div>
                  <span className="text-xs text-indigo-600 font-semibold">Running...</span>
                </div>
              )}
            </div>

            {/* BUTTON CONTROLS ALIGNED WITH SPEC 49 */}
            <div className="space-y-4">

              {/* PRIMARY LEARN STATUS TRIGGERS */}
              <div className="grid grid-cols-3 gap-3">
                <button
                  onClick={() => triggerControl(
                    "Start Learning",
                    "start-learning",
                    { currentLearning: "Continual Learning", nextAction: "Evaluate & Compare", status: "LEARNING..." },
                    "Start Learning signal transmitted successfully."
                  )}
                  disabled={!!loadingAction}
                  className="flex flex-col items-center justify-center p-3.5 bg-emerald-600 hover:bg-emerald-700 active:bg-emerald-800 text-white rounded-xl font-bold text-xs shadow-sm hover:shadow transition duration-200 disabled:opacity-50"
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
                  className="flex flex-col items-center justify-center p-3.5 bg-slate-600 hover:bg-slate-700 active:bg-slate-800 text-white rounded-xl font-bold text-xs shadow-sm hover:shadow transition duration-200 disabled:opacity-50"
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
                  className="flex flex-col items-center justify-center p-3.5 bg-red-600 hover:bg-red-700 active:bg-red-800 text-white rounded-xl font-bold text-xs shadow-sm hover:shadow transition duration-200 disabled:opacity-50"
                >
                  <span className="text-xl mb-1">⏹</span>
                  Stop
                </button>
              </div>

              {/* ACTION COMMAND CONTROLS */}
              <div className="grid grid-cols-2 gap-3">

                <button
                  onClick={() => triggerControl(
                    "Test Model",
                    "test-model",
                    { currentLearning: "Evaluating Model...", status: "TESTING..." },
                    "Benchmark evaluation suite triggered across multiple validation sets."
                  )}
                  disabled={!!loadingAction}
                  className="flex items-center space-x-2 p-3 bg-indigo-50 hover:bg-indigo-100 active:bg-indigo-200 text-indigo-900 rounded-xl text-left font-bold text-xs border border-indigo-100 shadow-sm transition duration-150 disabled:opacity-50"
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
                  className="flex items-center space-x-2 p-3 bg-violet-50 hover:bg-violet-100 active:bg-violet-200 text-violet-900 rounded-xl text-left font-bold text-xs border border-violet-100 shadow-sm transition duration-150 disabled:opacity-50"
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
                  className="flex items-center space-x-2 p-3 bg-pink-50 hover:bg-pink-100 active:bg-pink-200 text-pink-900 rounded-xl text-left font-bold text-xs border border-pink-100 shadow-sm transition duration-150 disabled:opacity-50"
                >
                  <span className="text-lg">📥</span>
                  <span>Find Knowledge Gaps</span>
                </button>

                <button
                  onClick={() => triggerControl(
                    "Collect Data",
                    "collect-data",
                    { currentLearning: "Ingesting Data...", status: "ACQUIRING..." },
                    "Triggering multi-source ingestion pipelines (Web, RSS, GitHub, YouTube)."
                  )}
                  disabled={!!loadingAction}
                  className="flex items-center space-x-2 p-3 bg-amber-50 hover:bg-amber-100 active:bg-amber-200 text-amber-900 rounded-xl text-left font-bold text-xs border border-amber-100 shadow-sm transition duration-150 disabled:opacity-50"
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
                  className="flex items-center space-x-2 p-3 bg-teal-50 hover:bg-teal-100 active:bg-teal-200 text-teal-900 rounded-xl text-left font-bold text-xs border border-teal-100 shadow-sm transition duration-150 disabled:opacity-50"
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
                  className="flex items-center space-x-2 p-3 bg-cyan-50 hover:bg-cyan-100 active:bg-cyan-200 text-cyan-900 rounded-xl text-left font-bold text-xs border border-cyan-100 shadow-sm transition duration-150 disabled:opacity-50"
                >
                  <span className="text-lg">🚀</span>
                  <span>Compare Models</span>
                </button>

              </div>

              {/* CRITICAL ROLLBACK / EMERGENCY PROMOTIONS */}
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
                    className="flex items-center justify-center space-x-2 p-3 bg-slate-100 hover:bg-slate-200 active:bg-slate-300 text-slate-800 rounded-xl font-bold text-xs shadow-sm transition duration-150 disabled:opacity-50"
                  >
                    <span className="text-base">↩</span>
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
                    className="flex items-center justify-center space-x-2 p-3 bg-amber-500 hover:bg-amber-600 active:bg-amber-700 text-white rounded-xl font-bold text-xs shadow-md shadow-amber-500/10 hover:shadow-md transition duration-150 disabled:opacity-50"
                  >
                    <span className="text-base">🛑</span>
                    <span>Emergency Promote</span>
                  </button>
                </div>
              </div>

            </div>
          </div>

          {/* DYNAMIC FEEDBACK LOG */}
          <div className="bg-slate-900 rounded-2xl p-5 border border-slate-800 shadow-inner space-y-3">
            <div className="flex justify-between items-center border-b border-slate-800 pb-2">
              <span className="text-xs font-bold text-slate-400 tracking-wider uppercase flex items-center">
                <span className="w-2 h-2 rounded-full bg-cyan-400 mr-2 animate-ping"></span>
                Console & Feedback Stream
              </span>
              <button
                onClick={() => setLogs(["Console cleared."])}
                className="text-[10px] text-slate-500 hover:text-slate-300 underline uppercase font-bold"
              >
                Clear
              </button>
            </div>
            <div className="h-40 overflow-y-auto space-y-2 font-mono text-xs text-slate-300 pr-2 scrollbar-thin scrollbar-thumb-slate-700">
              {logs.map((log, index) => (
                <div key={index} className={`pb-1 ${
                  log.includes("✅") ? "text-emerald-400" :
                  log.includes("ℹ️") ? "text-cyan-400" :
                  log.includes("⚙️") ? "text-slate-400 font-semibold" : "text-slate-300"
                }`}>
                  {log}
                </div>
              ))}
            </div>
          </div>
        </section>

      </div>
    </div>
  );
};
