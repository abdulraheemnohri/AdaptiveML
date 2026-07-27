import React, { useState } from 'react';

interface ModelVersion {
  version: string;
  accuracy: string;
  forgetting: string;
  safety: string;
  status: 'production' | 'candidate' | 'archived' | 'rejected';
}

export const ModelsView: React.FC = () => {
  const [versions, setVersions] = useState<ModelVersion[]>([
    { version: 'v3.4.2', accuracy: '94.2%', forgetting: '0.3%', safety: '98.9%', status: 'production' },
    { version: 'v3.4.3', accuracy: '95.1%', forgetting: '0.5%', safety: '99.1%', status: 'candidate' },
    { version: 'v3.4.1', accuracy: '93.4%', forgetting: '0.1%', safety: '98.5%', status: 'archived' }
  ]);

  const [promotionGates, setPromotionGates] = useState({
    capabilityImprovement: true,
    forgettingLimit: true,
    safetyScore: true,
    regressionLimit: true,
    humanApproval: false
  });

  const handlePromoteCandidate = () => {
    // Turn v3.4.3 into production and demote v3.4.2 to archived
    setVersions(prev => prev.map(v => {
      if (v.version === 'v3.4.3') return { ...v, status: 'production' };
      if (v.version === 'v3.4.2') return { ...v, status: 'archived' };
      return v;
    }));
    alert("Candidate Model successfully promoted to production!");
  };

  const handleRollback = () => {
    // Turn v3.4.1 into production, archived for the rest
    setVersions(prev => prev.map(v => {
      if (v.version === 'v3.4.1') return { ...v, status: 'production' };
      if (v.version === 'v3.4.2' || v.version === 'v3.4.3') return { ...v, status: 'archived' };
      return v;
    }));
    alert("System rolled back to stable version v3.4.1.");
  };

  return (
    <div className="p-6 max-w-7xl mx-auto space-y-6 text-slate-100 bg-slate-950 min-h-screen">
      <header className="border-b border-slate-800 pb-4">
        <h2 className="text-2xl font-black text-white">Model & Dynamic Adapter Registry</h2>
        <p className="text-xs text-slate-400 mt-1">Manage base model versions, configure promotion thresholds, and trigger automated rollbacks.</p>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* MODEL VERSIONS TABLE */}
        <div className="lg:col-span-8 bg-slate-900 border border-slate-800 rounded-2xl p-6 space-y-6">
          <h3 className="text-sm font-black text-slate-400 uppercase tracking-wider block">Registered Model Versions</h3>

          <div className="overflow-x-auto">
            <table className="w-full text-xs text-left text-slate-400">
              <thead className="text-[10px] text-slate-500 uppercase tracking-wider border-b border-slate-850">
                <tr>
                  <th className="py-3">Model Version</th>
                  <th className="py-3">Overall Quality</th>
                  <th className="py-3">Forgetting</th>
                  <th className="py-3">Safety Rating</th>
                  <th className="py-3">Status</th>
                </tr>
              </thead>
              <tbody>
                {versions.map(v => (
                  <tr key={v.version} className="border-b border-slate-850/50 hover:bg-slate-850/20">
                    <td className="py-3.5 font-bold text-slate-200">{v.version}</td>
                    <td className="py-3.5 text-cyan-400 font-bold">{v.accuracy}</td>
                    <td className="py-3.5 text-red-400">{v.forgetting}</td>
                    <td className="py-3.5 text-indigo-400">{v.safety}</td>
                    <td className="py-3.5">
                      <span className={`px-2.5 py-1 rounded text-[9px] font-black uppercase ${
                        v.status === 'production' ? 'bg-emerald-950/80 text-emerald-400 border border-emerald-900/50' :
                        v.status === 'candidate' ? 'bg-indigo-950/80 text-indigo-400 border border-indigo-900/50' :
                        'bg-slate-800 text-slate-400'
                      }`}>{v.status}</span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="flex space-x-3 pt-3">
            <button onClick={handlePromoteCandidate} className="flex-grow bg-indigo-600 hover:bg-indigo-700 text-white font-bold py-2.5 rounded-xl transition text-xs">
              🚀 Promote Candidate Model
            </button>
            <button onClick={handleRollback} className="flex-grow bg-slate-800 hover:bg-slate-700 text-slate-200 font-bold py-2.5 rounded-xl transition border border-slate-700 text-xs">
              ↩ Trigger Rollback
            </button>
          </div>
        </div>

        {/* PROMOTION GATES */}
        <div className="lg:col-span-4 bg-slate-900 border border-slate-800 rounded-2xl p-6 space-y-6 text-xs">
          <h3 className="text-sm font-black text-slate-400 uppercase tracking-wider border-b border-slate-850 pb-2">Model Promotion Gates (Section 30)</h3>

          <div className="space-y-4">
            <label className="flex items-center space-x-3 text-slate-300">
              <input
                type="checkbox"
                checked={promotionGates.capabilityImprovement}
                onChange={e => setPromotionGates(prev => ({ ...prev, capabilityImprovement: e.target.checked }))}
                className="rounded bg-slate-950 border-slate-800 h-4 w-4 text-indigo-600 focus:ring-0"
              />
              <span>Capability Improvement &gt;= 5%</span>
            </label>

            <label className="flex items-center space-x-3 text-slate-300">
              <input
                type="checkbox"
                checked={promotionGates.forgettingLimit}
                onChange={e => setPromotionGates(prev => ({ ...prev, forgettingLimit: e.target.checked }))}
                className="rounded bg-slate-950 border-slate-800 h-4 w-4 text-indigo-600 focus:ring-0"
              />
              <span>Catastrophic Forgetting &lt;= 3%</span>
            </label>

            <label className="flex items-center space-x-3 text-slate-300">
              <input
                type="checkbox"
                checked={promotionGates.safetyScore}
                onChange={e => setPromotionGates(prev => ({ ...prev, safetyScore: e.target.checked }))}
                className="rounded bg-slate-950 border-slate-800 h-4 w-4 text-indigo-600 focus:ring-0"
              />
              <span>Safety Evaluation Score &gt;= 98%</span>
            </label>

            <label className="flex items-center space-x-3 text-slate-300">
              <input
                type="checkbox"
                checked={promotionGates.regressionLimit}
                onChange={e => setPromotionGates(prev => ({ ...prev, regressionLimit: e.target.checked }))}
                className="rounded bg-slate-950 border-slate-800 h-4 w-4 text-indigo-600 focus:ring-0"
              />
              <span>Zero Regression on Core Benchmarks</span>
            </label>
          </div>
        </div>
      </div>
    </div>
  );
};
