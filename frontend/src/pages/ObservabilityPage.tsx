import React, { useState } from 'react';

export const ObservabilityPage: React.FC = () => {
  const [cpu, setCpu] = useState(42);
  const [gpu, setGpu] = useState(78);

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 text-slate-100 space-y-6">
      <h2 className="text-sm font-black text-slate-400 uppercase tracking-wider border-b border-slate-800 pb-3">Observability & Telemetry</h2>

      <div className="grid grid-cols-2 gap-4">
        <div className="bg-slate-950 p-4 border border-slate-850 rounded-xl text-center">
          <span className="text-xs text-slate-500 font-bold block">CPU Load</span>
          <span className="text-lg font-black text-indigo-400 mt-2 block">{cpu}%</span>
        </div>
        <div className="bg-slate-950 p-4 border border-slate-850 rounded-xl text-center">
          <span className="text-xs text-slate-500 font-bold block">GPU Load</span>
          <span className="text-lg font-black text-teal-400 mt-2 block">{gpu}%</span>
        </div>
      </div>
    </div>
  );
};
