import React from 'react';

export const LearningView: React.FC = () => {
  return (
    <div className="p-6">
      <h2 className="text-2xl font-bold mb-4">Three-Speed Learning Architecture</h2>
      <p className="text-gray-600">Fast (RAG/KG), Medium (Adapters/Memory), and Slow (Continual Training with EWC/MAS/SI).</p>
    </div>
  );
};
