import { Routes, Route } from 'react-router-dom'
import { Toaster } from '@/components/ui/toaster'
import Layout from './layouts/Layout'
import DashboardPage from './pages/Dashboard'
import DataHubPage from './pages/DataHub'
import DataSourcesPage from './pages/DataSources'
import DatasetsPage from './pages/Datasets'
import KnowledgePage from './pages/Knowledge'
import KnowledgeGraphPage from './pages/KnowledgeGraph'
import LearningCenterPage from './pages/LearningCenter'
import TrainingPage from './pages/Training'
import ContinualLearningPage from './pages/ContinualLearning'
import AntiForgettingPage from './pages/AntiForgetting'
import ResearchAgentsPage from './pages/ResearchAgents'
import ExperimentsPage from './pages/Experiments'
import TestingLabPage from './pages/TestingLab'
import BenchmarkCenterPage from './pages/BenchmarkCenter'
import ModelRegistryPage from './pages/ModelRegistry'
import ModelComparisonPage from './pages/ModelComparison'
import DeploymentPage from './pages/Deployment'
import MonitoringPage from './pages/Monitoring'
import ObservabilityPage from './pages/Observability'
import AutomationPage from './pages/Automation'
import SecurityPage from './pages/Security'
import AuditLogsPage from './pages/AuditLogs'
import SettingsPage from './pages/Settings'
import NotFoundPage from './pages/NotFound'

function App() {
  return (
    <>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<DashboardPage />} />
          <Route path="data-hub" element={<DataHubPage />} />
          <Route path="data/sources" element={<DataSourcesPage />} />
          <Route path="data/datasets" element={<DatasetsPage />} />
          <Route path="knowledge" element={<KnowledgePage />} />
          <Route path="knowledge/graph" element={<KnowledgeGraphPage />} />
          <Route path="learning" element={<LearningCenterPage />} />
          <Route path="training" element={<TrainingPage />} />
          <Route path="continual-learning" element={<ContinualLearningPage />} />
          <Route path="anti-forgetting" element={<AntiForgettingPage />} />
          <Route path="agents" element={<ResearchAgentsPage />} />
          <Route path="experiments" element={<ExperimentsPage />} />
          <Route path="testing" element={<TestingLabPage />} />
          <Route path="benchmarks" element={<BenchmarkCenterPage />} />
          <Route path="models" element={<ModelRegistryPage />} />
          <Route path="models/compare" element={<ModelComparisonPage />} />
          <Route path="deployment" element={<DeploymentPage />} />
          <Route path="monitoring" element={<MonitoringPage />} />
          <Route path="observability" element={<ObservabilityPage />} />
          <Route path="automation" element={<AutomationPage />} />
          <Route path="security" element={<SecurityPage />} />
          <Route path="audit-logs" element={<AuditLogsPage />} />
          <Route path="settings" element={<SettingsPage />} />
          <Route path="*" element={<NotFoundPage />} />
        </Route>
      </Routes>
      <Toaster />
    </>
  )
}

export default App