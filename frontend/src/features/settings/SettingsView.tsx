import React, { useState } from 'react';

export const SettingsView: React.FC = () => {
  const [activeTab, setActiveTab] = useState('general');
  const [systemControls, setSystemControls] = useState({
    trainingEngine: true, dataCollection: true, adaptiveLearning: true,
    continualLearning: true, antiForgetting: true, modelTesting: true,
    autoPromotion: false, autoRollback: true, localServing: true,
    apiServing: true, aiRouter: true, rag: true, memory: true, autonomousAgents: false
  });

  const triggerEmergency = (label: string) => alert(`⚠️ EMERGENCY: ${label}`);
  const toggleControl = (key: string) => setSystemControls((p: any) => ({ ...p, [key]: !p[key] }));

  const tabs = [
    { id: 'general', label: 'General', icon: '⚙️' },
    { id: 'training', label: 'Training', icon: '🧠' },
    { id: 'continual', label: 'Continual Learning', icon: '🔄' },
    { id: 'evaluation', label: 'Evaluation', icon: '📊' },
    { id: 'promotion', label: 'Promotion', icon: '✅' },
    { id: 'serving', label: 'Serving', icon: '🚀' },
    { id: 'local', label: 'Local Model', icon: '🖥️' },
    { id: 'apis', label: 'AI Providers', icon: '☁️' },
    { id: 'privacy', label: 'Privacy', icon: '🔒' },
    { id: 'storage', label: 'Storage', icon: '💾' },
    { id: 'security', label: 'Security', icon: '🛡️' },
    { id: 'control', label: 'System Control', icon: '🎛️' },
  ];

  const Input = ({ label, value, onChange, type = 'text', options }: any) => (
    <div className="space-y-2">
      <label className="text-sm font-semibold text-slate-300">{label}</label>
      {options ? (
        <select value={value} onChange={onChange} className="w-full bg-slate-800 border border-slate-700 rounded-lg px-4 py-2 text-white">
          {options.map((o: any) => <option key={o.value || o} value={o.value || o}>{o.label || o}</option>)}
        </select>
      ) : type === 'range' ? (
        <input type="range" {...{ value, onChange }} className="w-full" />
      ) : (
        <input type={type} value={value} onChange={onChange} className="w-full bg-slate-800 border border-slate-700 rounded-lg px-4 py-2 text-white" />
      )}
    </div>
  );

  const Toggle = ({ label, checked, onChange }: any) => (
    <div className="space-y-2">
      <label className="text-sm font-semibold text-slate-300">{label}</label>
      <div className="flex items-center">
        <button onClick={() => onChange(!checked)} className={`relative inline-flex h-6 w-11 items-center rounded-full transition ${checked ? 'bg-indigo-600' : 'bg-slate-700'}`}>
          <span className={`inline-block h-4 w-4 transform rounded-full bg-white transition ${checked ? 'translate-x-6' : 'translate-x-1'}`} />
        </button>
        <span className="ml-3 text-sm text-slate-400">{checked ? 'Enabled' : 'Disabled'}</span>
      </div>
    </div>
  );

  return (
    <div className="p-6 max-w-[1600px] mx-auto space-y-6 text-slate-100 bg-slate-950 min-h-screen">
      <header className="border-b border-slate-800 pb-4">
        <h2 className="text-2xl font-black text-white">Platform Settings & Control Centre</h2>
        <p className="text-xs text-slate-400 mt-1">Configure all aspects of Adaptive Omni ML platform</p>
      </header>

      <div className="flex overflow-x-auto gap-2 pb-2 border-b border-slate-800">
        {tabs.map(tab => (
          <button key={tab.id} onClick={() => setActiveTab(tab.id)}
            className={`px-4 py-2 rounded-lg text-sm font-semibold whitespace-nowrap transition ${activeTab === tab.id ? 'bg-indigo-600 text-white' : 'bg-slate-800 text-slate-400 hover:bg-slate-700'}`}>
            {tab.icon} {tab.label}
          </button>
        ))}
      </div>

      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6">
        {activeTab === 'general' && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <Input label="Application Name" value="Adaptive Omni ML" onChange={()=>{}} />
            <Input label="Theme" value="dark" onChange={()=>{}} options={['dark', 'light', 'system']} />
            <Input label="Language" value="en" onChange={()=>{}} options={[{value:'en',label:'English'},{value:'ur',label:'Urdu'},{value:'zh',label:'Chinese'}]} />
            <Input label="Startup Mode" value="dashboard" onChange={()=>{}} options={[{value:'dashboard',label:'Dashboard'},{value:'training',label:'Training'},{value:'serving',label:'Serving'}]} />
            <Toggle label="Notifications" checked={true} onChange={()=>{}} />
          </div>
        )}

        {activeTab === 'training' && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <Input label="Base Model" value="Qwen/Qwen2.5-Omni-3B" onChange={()=>{}} />
            <Input label="Training Directory" value="/workspace/models" onChange={()=>{}} />
            <Input label="Dataset Directory" value="/workspace/datasets" onChange={()=>{}} />
            <Input label="Checkpoint Frequency" value={1000} onChange={()=>{}} type="number" />
            <Input label="Max Training Budget (hrs)" value={100} onChange={()=>{}} type="number" />
            <Toggle label="Auto Training" checked={false} onChange={()=>{}} />
            <Input label="Schedule" value="manual" onChange={()=>{}} options={['manual','daily','weekly','on_new_data']} />
          </div>
        )}

        {activeTab === 'continual' && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <Input label="Replay Ratio" value={0.2} onChange={()=>{}} type="range" />
            <Input label="Replay Buffer Size" value={10000} onChange={()=>{}} type="number" />
            <Input label="Distillation Weight" value={0.5} onChange={()=>{}} type="range" />
            <Input label="EWC Strength" value={5000} onChange={()=>{}} type="number" />
            <Input label="Protected Capability Weight" value={0.3} onChange={()=>{}} type="range" />
            <Input label="Forgetting Threshold (%)" value={2.0} onChange={()=>{}} type="number" />
          </div>
        )}

        {activeTab === 'evaluation' && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <Toggle label="Automatic Testing" checked={true} onChange={()=>{}} />
            <Input label="Test Frequency" value="after_training" onChange={()=>{}} options={['after_training','daily','weekly','on_demand']} />
            <Input label="Required Benchmark Score (%)" value={85} onChange={()=>{}} type="number" />
            <Input label="Regression Threshold (%)" value={1.0} onChange={()=>{}} type="number" />
            <Input label="Forgetting Threshold (%)" value={2.0} onChange={()=>{}} type="number" />
          </div>
        )}

        {activeTab === 'promotion' && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <Input label="Approval Mode" value="manual" onChange={()=>{}} options={[{value:'manual',label:'Manual'},{value:'automatic',label:'Automatic'},{value:'hybrid',label:'Hybrid'}]} />
            <Input label="Quality Gate (%)" value={90} onChange={()=>{}} type="number" />
            <Input label="Safety Gate (%)" value={95} onChange={()=>{}} type="number" />
            <Input label="Regression Gate (%)" value={1.0} onChange={()=>{}} type="number" />
            <Input label="Forgetting Gate (%)" value={2.0} onChange={()=>{}} type="number" />
            <Input label="Rollback Policy" value="auto" onChange={()=>{}} options={[{value:'auto',label:'Automatic'},{value:'manual',label:'Manual'},{value:'disabled',label:'Disabled'}]} />
          </div>
        )}

        {activeTab === 'serving' && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <Input label="Default Serving" value="local" onChange={()=>{}} options={[{value:'local',label:'Local Only'},{value:'api',label:'API Only'},{value:'auto',label:'Automatic'}]} />
            <Input label="AI Router Mode" value="automatic" onChange={()=>{}} options={[{value:'local-first',label:'Local First'},{value:'api-first',label:'API First'},{value:'automatic',label:'Automatic'},{value:'manual',label:'Manual'}]} />
          </div>
        )}

        {activeTab === 'local' && (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            <Input label="Model Version" value="production" onChange={()=>{}} options={['production','previous','candidate','archived']} />
            <Input label="GPU Device" value={0} onChange={()=>{}} type="number" />
            <Input label="CPU Threads" value={8} onChange={()=>{}} type="number" />
            <Input label="VRAM Limit (%)" value={80} onChange={()=>{}} type="range" />
            <Input label="Quantisation" value="4-bit" onChange={()=>{}} options={['none','8-bit','4-bit']} />
            <Input label="Context Length" value={4096} onChange={()=>{}} options={[2048,4096,8192,16384,32768]} />
            <Input label="Batch Size" value={1} onChange={()=>{}} type="number" />
            <Input label="Temperature" value={0.7} onChange={()=>{}} type="range" />
            <Input label="Top P" value={0.9} onChange={()=>{}} type="range" />
            <Input label="Top K" value={50} onChange={()=>{}} type="number" />
          </div>
        )}

        {activeTab === 'apis' && (
          <div className="space-y-4">
            {['OpenAI','Anthropic','Google Gemini','Qwen','DeepSeek','Mistral'].map(name => (
              <div key={name} className="bg-slate-800 border border-slate-700 rounded-xl p-4">
                <div className="flex justify-between mb-3">
                  <h4 className="font-semibold text-white">{name}</h4>
                  <button className="relative inline-flex h-6 w-11 items-center rounded-full bg-slate-700"><span className="inline-block h-4 w-4 rounded-full bg-white translate-x-1"/></button>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                  <input placeholder="API Key" className="bg-slate-900 border border-slate-700 rounded px-3 py-2 text-white text-sm"/>
                  <input placeholder="Endpoint" className="bg-slate-900 border border-slate-700 rounded px-3 py-2 text-white text-sm"/>
                  <input placeholder="Default Model" className="bg-slate-900 border border-slate-700 rounded px-3 py-2 text-white text-sm"/>
                </div>
              </div>
            ))}
            <button className="w-full py-3 border-2 border-dashed border-slate-700 rounded-xl text-slate-400 hover:border-indigo-500">+ Add Custom Provider</button>
          </div>
        )}

        {activeTab === 'privacy' && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {['Local-Only Mode','Allow API Access','Send Files','Send Images','Send Audio','Send Video','Send History'].map(label => (
              <Toggle key={label} label={label} checked={false} onChange={()=>{}} />
            ))}
          </div>
        )}

        {activeTab === 'storage' && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <Input label="Models Directory" value="/workspace/models" onChange={()=>{}} />
            <Input label="Datasets Directory" value="/workspace/datasets" onChange={()=>{}} />
            <Input label="Checkpoints Directory" value="/workspace/checkpoints" onChange={()=>{}} />
            <Input label="Logs Directory" value="/workspace/logs" onChange={()=>{}} />
            <Input label="Backups Directory" value="/workspace/backups" onChange={()=>{}} />
            <Input label="Maximum Storage (GB)" value={500} onChange={()=>{}} type="number" />
          </div>
        )}

        {activeTab === 'security' && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <Toggle label="Authentication" checked={true} onChange={()=>{}} />
            <Toggle label="Data Encryption" checked={true} onChange={()=>{}} />
            <Toggle label="API Key Encryption" checked={true} onChange={()=>{}} />
            <Toggle label="Audit Logs" checked={true} onChange={()=>{}} />
            <Input label="Session Timeout (min)" value={60} onChange={()=>{}} type="number" />
          </div>
        )}

        {activeTab === 'control' && (
          <div className="space-y-6">
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6">
              <h4 className="text-sm font-black text-slate-400 uppercase tracking-wider mb-4">Component Status</h4>
              <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
                {Object.entries(systemControls).map(([key, enabled]) => (
                  <button key={key} onClick={() => toggleControl(key)}
                    className={`p-4 rounded-xl border transition ${enabled ? 'bg-green-900/30 border-green-500/50 text-green-200' : 'bg-red-900/30 border-red-500/50 text-red-200'}`}>
                    <div className="text-xs font-bold uppercase mb-1">{key.replace(/([A-Z])/g, ' $1').trim()}</div>
                    <div className="text-xs opacity-75">{enabled ? '🟢 ACTIVE' : '🔴 DISABLED'}</div>
                  </button>
                ))}
              </div>
            </div>
            <div className="bg-slate-900 border border-red-900/50 rounded-2xl p-6">
              <h4 className="text-sm font-black text-red-400 uppercase tracking-wider mb-4">🚨 Emergency Commands</h4>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                {['STOP ALL','PAUSE TRAINING','STOP COLLECTION','FREEZE PRODUCTION','ROLLBACK','DISABLE API','DISABLE AUTONOMY'].map(cmd => (
                  <button key={cmd} onClick={() => triggerEmergency(cmd)}
                    className="p-4 bg-red-950/80 border border-red-500/50 hover:bg-red-900 text-red-200 rounded-xl font-black uppercase tracking-wider">
                    {cmd}
                  </button>
                ))}
              </div>
              <p className="text-xs text-red-400 mt-4">⚠️ Emergency commands are logged.</p>
            </div>
          </div>
        )}
      </div>

      <div className="flex justify-end gap-4">
        <button className="px-6 py-3 bg-slate-800 hover:bg-slate-700 text-white rounded-xl font-semibold">Cancel</button>
        <button className="px-6 py-3 bg-indigo-600 hover:bg-indigo-700 text-white rounded-xl font-semibold">💾 Save All Settings</button>
      </div>
    </div>
  );
};
