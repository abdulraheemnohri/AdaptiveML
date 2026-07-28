import React, { useState } from 'react';

export const SettingsView: React.FC = () => {
  const [temperature, setTemperature] = useState(0.7);
  const [topP, setTopP] = useState(0.9);
  const [topK, setTopK] = useState(50);
  const [maxTokens, setMaxTokens] = useState(2048);
  const [precision, setPrecision] = useState('bf16');
  const [quantisation, setQuantisation] = useState('4-bit');

  const [emergencyActions, setEmergencyActions] = useState({
    stopAll: false,
    stopTraining: false,
    stopCollection: false,
    lockSystem: false
  });

  const triggerEmergency = (key: keyof typeof emergencyActions, label: string) => {
    setEmergencyActions(prev => ({ ...prev, [key]: true }));
    alert(`EMERGENCY COMMAND ISSUED: ${label}`);
  };

  return (
    <div className="p-6 max-w-7xl mx-auto space-y-6 text-slate-100 bg-slate-950 min-h-screen">
      <header className="border-b border-slate-800 pb-4">
        <h2 className="text-2xl font-black text-white">Platform & Inference Settings</h2>
        <p className="text-xs text-slate-400 mt-1">Configure global generation parameters, compute precision levels, and execute critical safety overrides.</p>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* INFERENCE CONTROLS */}
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 space-y-6">
          <h3 className="text-sm font-black text-slate-400 uppercase tracking-wider block">Generation Parameters (Section 40)</h3>

          <div className="space-y-4">
            <div className="space-y-2">
              <div className="flex justify-between text-xs font-bold">
                <label className="text-slate-300">Temperature (Sampling creativity)</label>
                <span>{temperature}</span>
              </div>
              <input
                type="range"
                min="0.1"
                max="1.5"
                step="0.05"
                value={temperature}
                onChange={e => setTemperature(Number(e.target.value))}
                className="w-full h-1 bg-slate-800 rounded-lg cursor-pointer"
              />
            </div>

            <div className="space-y-2">
              <div className="flex justify-between text-xs font-bold">
                <label className="text-slate-300">Top P (Nucleus filter)</label>
                <span>{topP}</span>
              </div>
              <input
                type="range"
                min="0.5"
                max="1.0"
                step="0.05"
                value={topP}
                onChange={e => setTopP(Number(e.target.value))}
                className="w-full h-1 bg-slate-800 rounded-lg cursor-pointer"
              />
            </div>

            <div className="space-y-2">
              <div className="flex justify-between text-xs font-bold">
                <label className="text-slate-300">Max Output Length (Tokens)</label>
                <span>{maxTokens}</span>
              </div>
              <input
                type="range"
                min="256"
                max="4096"
                step="256"
                value={maxTokens}
                onChange={e => setMaxTokens(Number(e.target.value))}
                className="w-full h-1 bg-slate-800 rounded-lg cursor-pointer"
              />
            </div>
          </div>
        </div>

        {/* PERFORMANCE & EMERGENCY OVERRIDES */}
        <div className="space-y-6">
          {/* COMPUTE */}
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 space-y-4 text-xs">
            <h3 className="text-sm font-black text-slate-400 uppercase tracking-wider block border-b border-slate-850 pb-2">Compute & Precision</h3>
            <div className="space-y-4">
              <div>
                <span className="text-slate-400 font-bold block mb-1">Execution Precision</span>
                <div className="grid grid-cols-3 gap-2">
                  {['bf16', 'fp16', 'fp32'].map(p => (
                    <button
                      key={p}
                      onClick={() => setPrecision(p)}
                      className={`p-2 rounded text-xs font-bold uppercase transition ${
                        precision === p ? 'bg-indigo-600 text-white' : 'bg-slate-800 text-slate-400'
                      }`}
                    >
                      {p}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* EMERGENCY (Section 45) */}
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 space-y-4">
            <h3 className="text-sm font-black text-red-400 uppercase tracking-wider block border-b border-slate-850 pb-2">🚨 Emergency Safety Commands</h3>
            <div className="grid grid-cols-2 gap-3 text-xs">
              <button
                onClick={() => triggerEmergency('stopAll', 'STOP ALL')}
                className="p-3 bg-red-950/80 border border-red-500/50 hover:bg-red-900 text-red-200 rounded-xl font-black uppercase tracking-wider transition"
              >
                STOP ALL
              </button>
              <button
                onClick={() => triggerEmergency('stopTraining', 'STOP TRAINING')}
                className="p-3 bg-red-950/80 border border-red-500/50 hover:bg-red-900 text-red-200 rounded-xl font-black uppercase tracking-wider transition"
              >
                STOP TRAINING
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
