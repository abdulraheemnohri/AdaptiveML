import React, { useState } from 'react';

export const TrainingLabPage: React.FC = () => {
  const [replayRatio, setReplayRatio] = useState(0.3);
  const [distillAlpha, setDistillAlpha] = useState(0.5);
  const [ewcWeight, setEwcWeight] = useState(1000.0);

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 text-slate-100 space-y-6 text-xs">
      <h2 className="text-sm font-black text-slate-400 uppercase tracking-wider border-b border-slate-800 pb-3">Continual Learning Training Lab</h2>

      <div className="space-y-4">
        <div>
          <div className="flex justify-between font-bold">
            <span>Experience Replay Ratio</span>
            <span>{replayRatio}</span>
          </div>
          <input type="range" min="0.1" max="0.9" step="0.05" value={replayRatio} onChange={e => setReplayRatio(Number(e.target.value))} className="w-full mt-1" />
        </div>

        <div>
          <div className="flex justify-between font-bold">
            <span>Knowledge Distillation Temperature (Alpha)</span>
            <span>{distillAlpha}</span>
          </div>
          <input type="range" min="0.1" max="1.0" step="0.05" value={distillAlpha} onChange={e => setDistillAlpha(Number(e.target.value))} className="w-full mt-1" />
        </div>

        <div>
          <div className="flex justify-between font-bold">
            <span>Elastic Weight Consolidation (EWC) Weight</span>
            <span>{ewcWeight}</span>
          </div>
          <input type="range" min="100.0" max="5000.0" step="100.0" value={ewcWeight} onChange={e => setEwcWeight(Number(e.target.value))} className="w-full mt-1" />
        </div>
      </div>
    </div>
  );
};
