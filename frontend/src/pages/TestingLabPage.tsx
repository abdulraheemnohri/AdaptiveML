import React from 'react';

export const TestingLabPage: React.FC = () => {
  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 text-slate-100 space-y-6 text-xs">
      <h2 className="text-sm font-black text-slate-400 uppercase tracking-wider border-b border-slate-800 pb-3">Model Testing Lab & Benchmark Centre</h2>

      <div className="space-y-4">
        <div className="space-y-1">
          <div className="flex justify-between font-bold">
            <span>MMLU General Reasoning</span>
            <span className="text-cyan-400">84.1%</span>
          </div>
          <div className="w-full bg-slate-950 rounded-full h-1.5">
            <div className="bg-indigo-600 h-1.5 rounded-full w-[84.1%]"></div>
          </div>
        </div>

        <div className="space-y-1">
          <div className="flex justify-between font-bold">
            <span>MMMU Vision understanding</span>
            <span className="text-cyan-400">76.5%</span>
          </div>
          <div className="w-full bg-slate-950 rounded-full h-1.5">
            <div className="bg-emerald-600 h-1.5 rounded-full w-[76.5%]"></div>
          </div>
        </div>

        <div className="space-y-1">
          <div className="flex justify-between font-bold">
            <span>LibriSpeech Speech Transcription</span>
            <span className="text-cyan-400">92.2%</span>
          </div>
          <div className="w-full bg-slate-950 rounded-full h-1.5">
            <div className="bg-teal-600 h-1.5 rounded-full w-[92.2%]"></div>
          </div>
        </div>
      </div>
    </div>
  );
};
