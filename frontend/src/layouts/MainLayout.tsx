import React from 'react';

export const MainLayout: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  return (
    <div className="min-h-screen bg-gray-50 flex flex-col">
      <nav className="bg-white border-b border-gray-200 px-6 py-4">
        <div className="flex justify-between items-center max-w-7xl mx-auto">
          <span className="text-xl font-bold text-gray-800">Adaptive Omni Brain</span>
          <span className="text-sm text-gray-500">Fast, Medium, and Slow Multimodal Continual Learning</span>
        </div>
      </nav>
      <main className="flex-grow">
        {children}
      </main>
    </div>
  );
};
