import React, { useState } from 'react';

export const RegistryPromotionPage: React.FC = () => {
  const [gates, setGates] = useState({
    improvement: true,
    forgetting: true,
    safety: true
  });

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 text-slate-100 space-y-6 text-xs">
      <h2 className="text-sm font-black text-slate-400 uppercase tracking-wider border-b border-slate-800 pb-3">Model Registry & Promotion gates</h2>

      <div className="space-y-3">
        <span className="text-slate-400 font-bold block">Model Promotion Gates Checklist (Section 30)</span>
        <label className="flex items-center space-x-3 text-slate-300">
          <input type="checkbox" checked={gates.improvement} onChange={e => setGates(prev => ({ ...prev, improvement: e.target.checked }))} className="rounded" />
          <span>New capability improvement &gt;= 5%</span>
        </label>
        <label className="flex items-center space-x-3 text-slate-300">
          <input type="checkbox" checked={gates.forgetting} onChange={e => setGates(prev => ({ ...prev, forgetting: e.target.checked }))} className="rounded" />
          <span>Catastrophic Forgetting &lt;= 3%</span>
        </label>
        <label className="flex items-center space-x-3 text-slate-300">
          <input type="checkbox" checked={gates.safety} onChange={e => setGates(prev => ({ ...prev, safety: e.target.checked }))} className="rounded" />
          <span>Model Safety score &gt;= 98%</span>
        </label>
      </div>

      <div className="flex space-x-2 pt-2">
        <button onClick={() => alert("Model Promoted Successfully")} className="bg-indigo-600 hover:bg-indigo-700 text-white font-bold p-2.5 rounded-lg transition-all text-xs">Promote Candidate Model</button>
        <button onClick={() => alert("Model Rolled Back")} className="bg-slate-800 hover:bg-slate-700 text-slate-300 font-bold p-2.5 rounded-lg transition-all border border-slate-700 text-xs">Trigger Rollback</button>
      </div>
    </div>
  );
};
