import React from 'react';

export const Dashboard: React.FC = () => {
  return (
    <div className="p-6 max-w-7xl mx-auto space-y-6">
      <header className="flex justify-between items-center mb-8">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Adaptive Omni Brain</h1>
          <p className="text-gray-500">Learn Continuously. Remember Permanently. Forget Nothing Important.</p>
        </div>
        <div className="flex space-x-3">
          <span className="bg-green-100 text-green-800 text-sm font-semibold px-3 py-1 rounded-full flex items-center">
            <span className="w-2.5 h-2.5 bg-green-500 rounded-full mr-2"></span>
            Production: v2.4.1
          </span>
        </div>
      </header>

      {/* Main Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
          <h3 className="text-sm font-medium text-gray-400">Knowledge Coverage</h3>
          <p className="text-3xl font-semibold text-gray-900 mt-2">82% <span className="text-green-500 text-sm font-normal">↑ 1.4%</span></p>
        </div>
        <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
          <h3 className="text-sm font-medium text-gray-400">Knowledge Retention</h3>
          <p className="text-3xl font-semibold text-gray-900 mt-2">98.7%</p>
        </div>
        <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
          <h3 className="text-sm font-medium text-gray-400">Active Adapters</h3>
          <p className="text-3xl font-semibold text-gray-900 mt-2">18</p>
        </div>
        <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
          <h3 className="text-sm font-medium text-gray-400">Replay Memory</h3>
          <p className="text-3xl font-semibold text-gray-900 mt-2">1.2M</p>
        </div>
      </div>

      {/* Modality Health */}
      <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100 mt-8">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Modality Accuracy & Forgetting Risks</h3>
        <div className="grid grid-cols-2 md:grid-cols-5 gap-4 text-center">
          <div className="p-4 bg-gray-50 rounded-lg">
            <span className="text-sm font-medium text-gray-500">Text</span>
            <div className="text-2xl font-bold text-green-600 mt-1">🟢 94%</div>
          </div>
          <div className="p-4 bg-gray-50 rounded-lg">
            <span className="text-sm font-medium text-gray-500">Vision</span>
            <div className="text-2xl font-bold text-green-600 mt-1">🟢 91%</div>
          </div>
          <div className="p-4 bg-gray-50 rounded-lg">
            <span className="text-sm font-medium text-gray-500">Audio</span>
            <div className="text-2xl font-bold text-yellow-600 mt-1">🟡 84%</div>
          </div>
          <div className="p-4 bg-gray-50 rounded-lg">
            <span className="text-sm font-medium text-gray-500">Video</span>
            <div className="text-2xl font-bold text-green-600 mt-1">🟢 89%</div>
          </div>
          <div className="p-4 bg-gray-50 rounded-lg">
            <span className="text-sm font-medium text-gray-500">Speech</span>
            <div className="text-2xl font-bold text-green-600 mt-1">🟢 90%</div>
          </div>
        </div>
      </div>
    </div>
  );
};
