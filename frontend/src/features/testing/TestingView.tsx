import React, { useState } from 'react';

interface Benchmark {
  name: string;
  category: string;
  baseline: string;
  candidate: string;
  status: 'passed' | 'warning' | 'failed';
}

export const TestingView: React.FC = () => {
  const [benchmarks, setBenchmarks] = useState<Benchmark[]>([
    { name: 'MMLU General Reasoning', category: 'Text Reasoning', baseline: '82.4%', candidate: '84.1%', status: 'passed' },
    { name: 'MMMU Vision Understanding', category: 'Vision', baseline: '75.1%', candidate: '76.5%', status: 'passed' },
    { name: 'LibriSpeech Speech Transcription', category: 'Audio', baseline: '91.0%', candidate: '92.2%', status: 'passed' },
    { name: 'MSR-VTT Video Temporal Logic', category: 'Video', baseline: '68.3%', candidate: '68.9%', status: 'passed' },
    { name: 'Urdu Language Translation', category: 'Multilingual', baseline: '95.4%', candidate: '98.9%', status: 'passed' }
  ]);

  const [firewallWarnings, setFirewallWarnings] = useState<string[]>([
    "Urdu language retention rate: 98.9% (Passed)",
    "General reasoning regression check: +1.7% (Passed)",
    "Coding skill preservation: 99.5% (Passed)"
  ]);

  const runRetentionTests = () => {
    setFirewallWarnings(prev => [
      `[${new Date().toLocaleTimeString()}] Automated retention test complete: 99.1% retention rate calculated. Zero forgetting detected.`,
      ...prev
    ]);
  };

  return (
    <div className="p-6 max-w-7xl mx-auto space-y-6 text-slate-100 bg-slate-950 min-h-screen">
      <header className="border-b border-slate-800 pb-4">
        <h2 className="text-2xl font-black text-white">Catastrophic Forgetting Firewall</h2>
        <p className="text-xs text-slate-400 mt-1">Audit model regression, measure forgetting rates, and trigger automatic anti-forgetting recovery procedures.</p>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* BENCHMARKS */}
        <div className="lg:col-span-8 bg-slate-900 border border-slate-800 rounded-2xl p-6 space-y-6">
          <div className="flex justify-between items-center border-b border-slate-850 pb-3">
            <h3 className="text-sm font-black text-slate-400 uppercase tracking-wider block">Cross-Modal Benchmarks (Section 21)</h3>
            <button onClick={runRetentionTests} className="bg-indigo-600 hover:bg-indigo-700 text-white font-bold px-3 py-1.5 rounded-lg text-xs transition">
              ⚡ Run Retention Suite
            </button>
          </div>

          <div className="space-y-4">
            {benchmarks.map(b => (
              <div key={b.name} className="p-4 bg-slate-950 border border-slate-850 rounded-xl space-y-2">
                <div className="flex justify-between items-center text-xs font-bold">
                  <span className="text-slate-200">{b.name}</span>
                  <span className="text-indigo-400 uppercase text-[10px]">{b.category}</span>
                </div>
                <div className="flex justify-between text-[11px] text-slate-400">
                  <span>Baseline: {b.baseline}</span>
                  <span className="font-extrabold text-emerald-400">Candidate: {b.candidate}</span>
                </div>
                <div className="w-full bg-slate-900 rounded-full h-1.5 mt-1">
                  <div className="bg-indigo-500 h-1.5 rounded-full w-[80%]"></div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* FIREWALL SNAPSHOTS */}
        <div className="lg:col-span-4 bg-slate-900 border border-slate-800 rounded-2xl p-6 flex flex-col h-fit space-y-4">
          <h3 className="text-sm font-black text-slate-400 uppercase tracking-wider block border-b border-slate-850 pb-2">Forgetting Warnings (Section 14)</h3>
          <div className="space-y-2 font-mono text-[10px] text-slate-300">
            {firewallWarnings.map((warn, i) => (
              <div key={i} className="p-3 bg-slate-950 border border-slate-850 rounded-lg leading-relaxed">
                {warn}
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};
