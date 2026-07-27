import React from 'react';

export const ModelsView: React.FC = () => {
  return (
    <div className="p-6">
      <h2 className="text-2xl font-bold mb-4">Model & Dynamic Adapter Registry</h2>
      <p className="text-gray-600">Model versions, LoRA/QLoRA adapter configurations, promotion gates, and rollback manager.</p>
    </div>
  );
};
