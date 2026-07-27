import { useState } from 'react'
import { Outlet, Link, useLocation } from 'react-router-dom'
import { 
  LayoutDashboard, 
  Database, 
  Brain, 
  TestTube2, 
  HardDrive, 
  Settings,
  Menu,
  X,
  Cpu,
  Activity,
  Bell
} from 'lucide-react'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import { useDashboardStore } from '@/stores/dashboard'

const navigation = [
  { name: 'Dashboard', href: '/', icon: LayoutDashboard },
  { name: 'Data Hub', href: '/data-hub', icon: Database },
  { name: 'Learning Centre', href: '/learning', icon: Brain },
  { name: 'Testing Lab', href: '/testing', icon: TestTube2 },
  { name: 'Model Registry', href: '/models', icon: HardDrive },
  { name: 'Settings', href: '/settings', icon: Settings },
]

export function Layout({ children }: { children: React.ReactNode }) {
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const location = useLocation()
  const { currentModel, systemHealth } = useDashboardStore()

  return (
    <div className="min-h-screen bg-background">
      {/* Mobile sidebar */}
      <div className={cn('lg:hidden', sidebarOpen ? 'block' : 'hidden')}>
        <div className="fixed inset-0 z-40 flex">
          <div className="fixed inset-0 bg-black/50" onClick={() => setSidebarOpen(false)} />
          <div className="relative flex w-full max-w-xs flex-1 flex-col bg-card pt-5 pb-4">
            <div className="flex shrink-0 items-center px-4">
              <Brain className="h-8 w-8 text-primary" />
              <span className="ml-2 text-lg font-bold">Adaptive Omni ML</span>
            </div>
            <nav className="mt-5 space-y-1 px-2">
              {navigation.map((item) => (
                <Link
                  key={item.name}
                  to={item.href}
                  onClick={() => setSidebarOpen(false)}
                  className={cn(
                    'group flex items-center rounded-md px-2 py-2 text-sm font-medium',
                    location.pathname === item.href
                      ? 'bg-primary text-primary-foreground'
                      : 'text-foreground hover:bg-accent'
                  )}
                >
                  <item.icon className="mr-3 h-5 w-5" />
                  {item.name}
                </Link>
              ))}
            </nav>
          </div>
        </div>
      </div>

      {/* Desktop sidebar */}
      <div className="hidden lg:fixed lg:inset-y-0 lg:flex lg:w-64 lg:flex-col">
        <div className="flex min-h-0 flex-1 flex-col bg-card border-r">
          <div className="flex h-16 shrink-0 items-center px-6 border-b">
            <Brain className="h-8 w-8 text-primary" />
            <span className="ml-2 text-lg font-bold">Adaptive Omni ML</span>
          </div>
          <nav className="flex-1 space-y-1 px-4 py-4">
            {navigation.map((item) => (
              <Link
                key={item.name}
                to={item.href}
                className={cn(
                  'group flex items-center rounded-md px-3 py-2 text-sm font-medium',
                  location.pathname === item.href
                    ? 'bg-primary text-primary-foreground'
                    : 'text-foreground hover:bg-accent'
                )}
              >
                <item.icon className="mr-3 h-5 w-5" />
                {item.name}
              </Link>
            ))}
          </nav>
          <div className="p-4 border-t">
            <div className="flex items-center gap-2 text-xs text-muted-foreground">
              <Cpu className="h-4 w-4" />
              <span>CPU: {systemHealth.cpu}%</span>
              <Activity className="h-4 w-4 ml-2" />
              <span>GPU: {systemHealth.gpu}%</span>
            </div>
          </div>
        </div>
      </div>

      {/* Main content */}
      <div className="lg:pl-64">
        {/* Top bar */}
        <div className="sticky top-0 z-30 flex h-16 shrink-0 items-center gap-x-4 border-b bg-background px-4 shadow-sm sm:gap-x-6 sm:px-6 lg:px-8">
          <Button
            variant="ghost"
            size="icon"
            className="lg:hidden"
            onClick={() => setSidebarOpen(true)}
          >
            <Menu className="h-6 w-6" />
          </Button>

          <div className="flex flex-1 gap-x-4 self-stretch lg:gap-x-6">
            <div className="flex flex-1 items-center">
              <div className="text-sm text-muted-foreground">
                Current Model: <span className="font-medium text-foreground">{currentModel.name}</span>
                <span className="mx-2">•</span>
                <span className="font-medium text-foreground">{currentModel.version}</span>
                <span className="mx-2">•</span>
                <span className={cn(
                  'px-2 py-0.5 rounded-full text-xs',
                  currentModel.status === 'production' ? 'bg-green-500/10 text-green-500' :
                  currentModel.status === 'training' ? 'bg-blue-500/10 text-blue-500' :
                  'bg-gray-500/10 text-gray-500'
                )}>
                  {currentModel.status}
                </span>
              </div>
            </div>
            <div className="flex items-center gap-x-4 lg:gap-x-6">
              <Button variant="ghost" size="icon">
                <Bell className="h-5 w-5" />
              </Button>
            </div>
          </div>
        </div>

        {/* Page content */}
        <main className="py-6">
          <div className="px-4 sm:px-6 lg:px-8">
            {children}
          </div>
        </main>
      </div>
    </div>
  )
}
