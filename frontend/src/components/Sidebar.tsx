import { Link, useLocation } from 'react-router-dom'
import { cn } from '@/lib/utils'
import {
  LayoutDashboard,
  Database,
  FileText,
  BookOpen,
  Brain,
  GraduationCap,
  ShieldCheck,
  BarChart3,
  Package,
  Rocket,
  Eye,
  Settings,
  Users,
  Logs,
  Bot,
  FlaskConical,
  GitBranch,
  CPU,
  HardDrive,
  Network,
} from 'lucide-react'

const navigationItems = [
  { title: 'Dashboard', url: '/', icon: <LayoutDashboard className="h-4 w-4" /> },
  { title: 'AI Workspace', url: '#', icon: <Brain className="h-4 w-4" /> },
]

const dataItems = [
  { title: 'Data Hub', url: '/data-hub', icon: <Database className="h-4 w-4" /> },
  { title: 'Data Sources', url: '/data/sources', icon: <GitBranch className="h-4 w-4" /> },
  { title: 'Datasets', url: '/data/datasets', icon: <FileText className="h-4 w-4" /> },
]

const knowledgeItems = [
  { title: 'Knowledge', url: '/knowledge', icon: <BookOpen className="h-4 w-4" /> },
  { title: 'Knowledge Graph', url: '/knowledge/graph', icon: <Network className="h-4 w-4" /> },
]

const learningItems = [
  { title: 'Learning Centre', url: '/learning', icon: <GraduationCap className="h-4 w-4" /> },
  { title: 'Training', url: '/training', icon: <FlaskConical className="h-4 w-4" /> },
  { title: 'Continual Learning', url: '/continual-learning', icon: <CPU className="h-4 w-4" /> },
  { title: 'Anti-Forgetting', url: '/anti-forgetting', icon: <ShieldCheck className="h-4 w-4" /> },
  { title: 'Research Agents', url: '/agents', icon: <Bot className="h-4 w-4" /> },
  { title: 'Experiments', url: '/experiments', icon: <Package className="h-4 w-4" /> },
]

const testingItems = [
  { title: 'Testing Lab', url: '/testing', icon: <FlaskConical className="h-4 w-4" /> },
  { title: 'Benchmark Centre', url: '/benchmarks', icon: <BarChart3 className="h-4 w-4" /> },
]

const modelItems = [
  { title: 'Model Registry', url: '/models', icon: <HardDrive className="h-4 w-4" /> },
  { title: 'Model Comparison', url: '/models/compare', icon: <BarChart3 className="h-4 w-4" /> },
  { title: 'Deployment', url: '/deployment', icon: <Rocket className="h-4 w-4" /> },
]

const monitoringItems = [
  { title: 'Monitoring', url: '/monitoring', icon: <Eye className="h-4 w-4" /> },
  { title: 'Observability', url: '/observability', icon: <BarChart3 className="h-4 w-4" /> },
]

const adminItems = [
  { title: 'Automation', url: '/automation', icon: <CPU className="h-4 w-4" /> },
  { title: 'Security', url: '/security', icon: <ShieldCheck className="h-4 w-4" /> },
  { title: 'Audit Logs', url: '/audit-logs', icon: <Logs className="h-4 w-4" /> },
  { title: 'Settings', url: '/settings', icon: <Settings className="h-4 w-4" /> },
]

function Sidebar() {
  const location = useLocation()
  const isActive = (url: string) => {
    return location.pathname === url || location.pathname.startsWith(url + '/')
  }

  const renderNavItem = (item: any) => (
    <Link
      key={item.url}
      to={item.url}
      className={cn(
        'flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition-colors',
        isActive(item.url)
          ? 'bg-accent text-accent-foreground'
          : 'text-muted-foreground hover:bg-accent hover:text-accent-foreground',
      )}
    >
      {item.icon}
      <span>{item.title}</span>
    </Link>
  )

  const renderNavSection = (title: string, items: any[]) => (
    <div className="space-y-1">
      <h3 className="px-3 text-xs font-semibold text-muted-foreground/70 uppercase tracking-wider">
        {title}
      </h3>
      <div className="space-y-1">{items.map(renderNavItem)}</div>
    </div>
  )

  return (
    <aside className="w-64 h-screen border-r bg-background sticky top-0 left-0 z-10">
      <div className="h-full flex flex-col">
        <div className="border-b p-4">
          <Link to="/" className="flex items-center gap-2">
            <div className="w-8 h-8 bg-primary rounded-lg flex items-center justify-center">
              <Brain className="h-5 w-5 text-primary-foreground" />
            </div>
            <span className="font-bold text-lg">Adaptive Omni ML</span>
          </Link>
        </div>
        <div className="flex-1 overflow-y-auto p-4 space-y-6">
          {renderNavSection('Navigation', navigationItems)}
          {renderNavSection('Data Hub', dataItems)}
          {renderNavSection('Knowledge', knowledgeItems)}
          {renderNavSection('Learning Centre', learningItems)}
          {renderNavSection('Testing Lab', testingItems)}
          {renderNavSection('Model Registry', modelItems)}
          {renderNavSection('Monitoring', monitoringItems)}
          {renderNavSection('Administration', adminItems)}
        </div>
        <div className="border-t p-4">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 bg-muted rounded-full flex items-center justify-center">
              <Users className="h-4 w-4" />
            </div>
            <div>
              <p className="text-sm font-medium">Abdulraheem Nohari</p>
              <p className="text-xs text-muted-foreground">Admin</p>
            </div>
          </div>
        </div>
      </div>
    </aside>
  )
}

export default Sidebar