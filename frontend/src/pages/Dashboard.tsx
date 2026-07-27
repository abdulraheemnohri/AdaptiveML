import { DashboardOverview } from '@/features/dashboard/Overview'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Play, Pause, RefreshCw, Brain, Database, TestTube, HardDrive } from 'lucide-react'

export function DashboardPage() {
  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold tracking-tight">Dashboard</h1>
        <div className="flex gap-2">
          <Button>
            <Play className="mr-2 h-4 w-4" />
            Start Learning
          </Button>
          <Button variant="outline">
            <Pause className="mr-2 h-4 w-4" />
            Pause
          </Button>
          <Button variant="outline">
            <RefreshCw className="mr-2 h-4 w-4" />
            Refresh
          </Button>
        </div>
      </div>

      <DashboardOverview />

      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Learning Tasks</CardTitle>
            <Brain className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">34</div>
            <p className="text-xs text-muted-foreground">+12 from last cycle</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Data Sources</CardTitle>
            <Database className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">8</div>
            <p className="text-xs text-muted-foreground">5 active collectors</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Active Experiments</CardTitle>
            <TestTube className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">3</div>
            <p className="text-xs text-muted-foreground">1 completed today</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Model Versions</CardTitle>
            <HardDrive className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">12</div>
            <p className="text-xs text-muted-foreground">v1.3.0 latest</p>
          </CardContent>
        </Card>
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Recent Activity</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {[
                { action: 'Data collection completed', time: '2 minutes ago', type: 'success' },
                { action: 'Model evaluation passed', time: '15 minutes ago', type: 'success' },
                { action: 'New adapter created: Urdu-v2', time: '1 hour ago', type: 'info' },
                { action: 'Knowledge gap detected: Quantum Physics', time: '2 hours ago', type: 'warning' },
                { action: 'Forgetting score below threshold', time: '3 hours ago', type: 'success' },
              ].map((activity, i) => (
                <div key={i} className="flex items-center justify-between border-b pb-2 last:border-0">
                  <div>
                    <p className="text-sm font-medium">{activity.action}</p>
                    <p className="text-xs text-muted-foreground">{activity.time}</p>
                  </div>
                  <span className={cn(
                    'px-2 py-1 rounded-full text-xs',
                    activity.type === 'success' ? 'bg-green-500/10 text-green-500' :
                    activity.type === 'warning' ? 'bg-yellow-500/10 text-yellow-500' :
                    'bg-blue-500/10 text-blue-500'
                  )}>
                    {activity.type}
                  </span>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Quick Actions</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid gap-2">
              <Button variant="outline" className="justify-start">
                <Play className="mr-2 h-4 w-4" />
                Start New Training Cycle
              </Button>
              <Button variant="outline" className="justify-start">
                <TestTube className="mr-2 h-4 w-4" />
                Run Evaluation Suite
              </Button>
              <Button variant="outline" className="justify-start">
                <Database className="mr-2 h-4 w-4" />
                Import New Dataset
              </Button>
              <Button variant="outline" className="justify-start">
                <Brain className="mr-2 h-4 w-4" />
                Discover Knowledge Gaps
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}

function cn(...classes: string[]) {
  return classes.filter(Boolean).join(' ')
}
