import React from 'react';

interface MetricCardProps {
  title: string;
  value: string | number;
  status?: string;
}

export const MetricCard: React.FC<MetricCardProps> = ({ title, value, status }) => {
  return (
    <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
      <h3 className="text-sm font-medium text-gray-400">{title}</h3>
      <p className="text-3xl font-semibold text-gray-900 mt-2">
        {value} {status && <span className="text-green-500 text-sm font-normal">{status}</span>}
      </p>
    </div>
  );
};
