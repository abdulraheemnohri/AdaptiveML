import React, { useState } from 'react';

interface Agent {
  id: string;
  name: string;
  role: string;
  status: 'idle' | 'running' | 'paused' | 'disabled';
  autonomousLevel: string;
  lastTask: string;
}

export const AgentsView: React.FC = () => {
  const [agents, setAgents] = useState<Agent[]>([
    { id: '1', name: 'Supervisor Agent', role: 'Coordinates research tasks and assigns gaps', status: 'idle', autonomousLevel: 'autonomous', lastTask: 'Validated v3.4.2 deployment' },
    { id: '2', name: 'Research Agent', role: 'Discovers knowledge gaps & collects facts', status: 'running', autonomousLevel: 'semi-automatic', lastTask: 'Searching Urdu technical vocabularies' },
    { id: '3', name: 'Data Collector Agent', role: 'Ingests Web, RSS, and Git data sources', status: 'idle', autonomousLevel: 'automatic', lastTask: 'Ingested MMMU benchmarks repo' },
    { id: '4', name: 'Data Cleaner Agent', role: 'Deduplicates, extracts and formats data', status: 'idle', autonomousLevel: 'automatic', lastTask: 'Deduplicated Wiki text' },
    { id: '5', name: 'Verification Agent', role: 'Cross-verifies facts across multiple sources', status: 'idle', autonomousLevel: 'semi-automatic', lastTask: 'Verified Qwen Omni specs' },
    { id: '6', name: 'Teacher Agent', role: 'Generates conversational instruction-output pairs', status: 'idle', autonomousLevel: 'assisted', lastTask: 'Created 24 instruction pairs' },
    { id: '7', name: 'Critic Agent', role: 'Flags hallucinations & contradiction metrics', status: 'idle', autonomousLevel: 'assisted', lastTask: 'Audited candidate v3.4.3' },
    { id: '8', name: 'Training Agent', role: 'Triggers fine-tuning and protects parameters', status: 'idle', autonomousLevel: 'semi-automatic', lastTask: 'Ran EWC lambda training' },
    { id: '9', name: 'Evaluation Agent', role: 'Runs cross-modal benchmarks (MMLU, MMMU)', status: 'idle', autonomousLevel: 'automatic', lastTask: 'Completed LibriSpeech test suite' },
    { id: '10', name: 'Forgetting Detection Agent', role: 'Runs anti-forgetting firewall suites', status: 'idle', autonomousLevel: 'autonomous', lastTask: 'Calculated Urdu retention rate' },
    { id: '11', name: 'Safety Agent', role: 'Screens content for profanity & poisoning', status: 'running', autonomousLevel: 'autonomous', lastTask: 'Screened raw RSS data' },
    { id: '12', name: 'Monitoring Agent', role: 'Monitors hardware loads, metrics, and errors', status: 'running', autonomousLevel: 'autonomous', lastTask: 'Updated GPU/VRAM telemetry' }
  ]);

  const [logs, setLogs] = useState<string[]>([
    "[Supervisor Agent] All 12 autonomous system agents initialized.",
    "[Research Agent] High-confidence Urdu dialect research task registered.",
    "[Safety Agent] Prompt injection filters loaded successfully."
  ]);

  const toggleAgentStatus = (id: string) => {
    setAgents(prev => prev.map(a => {
      if (a.id === id) {
        const nextStatus: Record<string, 'idle' | 'running' | 'paused' | 'disabled'> = {
          'idle': 'disabled',
          'running': 'disabled',
          'paused': 'disabled',
          'disabled': 'idle'
        };
        const status = nextStatus[a.status] || 'idle';
        const time = new Date().toLocaleTimeString();
        setLogs(l => [`[${time}] 🔄 Toggled ${a.name} status to ${status.toUpperCase()}`, ...l]);
        return { ...a, status };
      }
      return a;
    }));
  };

  const cycleAutonomy = (id: string) => {
    const levels = ['assisted', 'semi-automatic', 'automatic', 'autonomous'];
    setAgents(prev => prev.map(a => {
      if (a.id === id) {
        const nextIdx = (levels.indexOf(a.autonomousLevel) + 1) % levels.length;
        const nextLevel = levels[nextIdx];
        const time = new Date().toLocaleTimeString();
        setLogs(l => [`[${time}] ⚙️ Changed ${a.name} autonomy level to ${nextLevel.toUpperCase()}`, ...l]);
        return { ...a, autonomousLevel: nextLevel };
      }
      return a;
    }));
  };

  return (
    <div className="p-6 max-w-7xl mx-auto space-y-6 text-slate-100 bg-slate-950 min-h-screen">
      <header className="border-b border-slate-800 pb-4">
        <h2 className="text-2xl font-black text-white">Autonomous Multi-Agent Research System</h2>
        <p className="text-xs text-slate-400 mt-1">Configure permission levels, assign research tasks, and monitor agents in the self-improving workspace.</p>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* AGENTS LIST */}
        <div className="lg:col-span-8 bg-slate-900 border border-slate-800 rounded-2xl p-6 space-y-4">
          <h3 className="text-sm font-black text-slate-400 uppercase tracking-wider block">Agent Registry</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {agents.map(agent => (
              <div key={agent.id} className="p-4 bg-slate-950 border border-slate-850 rounded-xl space-y-3">
                <div className="flex justify-between items-center text-xs">
                  <span className="font-extrabold text-white">{agent.name}</span>
                  <span className={`px-2 py-0.5 rounded text-[10px] font-bold uppercase ${
                    agent.status === 'running' ? 'bg-indigo-950 text-indigo-400 border border-indigo-900' :
                    agent.status === 'idle' ? 'bg-emerald-950 text-emerald-400 border border-emerald-900' :
                    'bg-slate-800 text-slate-400'
                  }`}>{agent.status}</span>
                </div>
                <p className="text-[11px] text-slate-400 leading-normal">{agent.role}</p>
                <div className="text-[10px] text-slate-500 font-mono">Last action: {agent.lastTask}</div>
                <div className="flex justify-between items-center pt-2 border-t border-slate-850/50 text-[10px] font-bold">
                  <button onClick={() => cycleAutonomy(agent.id)} className="text-indigo-400 hover:underline">Autonomy: {agent.autonomousLevel.toUpperCase()}</button>
                  <button onClick={() => toggleAgentStatus(agent.id)} className="text-red-400 hover:underline">
                    {agent.status === 'disabled' ? 'Enable' : 'Disable'}
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* LOGS */}
        <div className="lg:col-span-4 bg-slate-900 border border-slate-800 rounded-2xl p-6 flex flex-col h-fit space-y-4">
          <h3 className="text-sm font-black text-slate-400 uppercase tracking-wider block">Agent Logs</h3>
          <div className="h-[400px] overflow-y-auto space-y-2 font-mono text-[10px] text-slate-300 pr-1">
            {logs.map((log, i) => (
              <div key={i} className="p-2 bg-slate-950 rounded border border-slate-850/60 leading-relaxed">{log}</div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};
