import { Routes, Route } from 'react-router-dom'
import { Layout } from '@/layouts/MainLayout'
import { DashboardPage } from '@/pages/Dashboard'
import { DataHubPage } from '@/pages/DataHub'
import { LearningCentrePage } from '@/pages/LearningCentre'
import { TestingLabPage } from '@/pages/TestingLab'
import { ModelRegistryPage } from '@/pages/ModelRegistry'
import { SettingsPage } from '@/pages/Settings'

export default function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<DashboardPage />} />
        <Route path="/data-hub" element={<DataHubPage />} />
        <Route path="/learning" element={<LearningCentrePage />} />
        <Route path="/testing" element={<TestingLabPage />} />
        <Route path="/models" element={<ModelRegistryPage />} />
        <Route path="/settings" element={<SettingsPage />} />
      </Routes>
    </Layout>
  )
}
