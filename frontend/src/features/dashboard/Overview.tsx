import { useDashboardStore } from '@/stores/dashboard'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Activity, Brain, Database, Cpu, HardDrive, Zap } from 'lucide-react'

export function DashboardOverview() {
  const { systemHealth, learningProgress, currentModel } = useDashboardStore()

  const stats = [
    {
      title: 'Knowledge Growth',
      value: `+${learningProgress.knowledgeGrowth}%`,
      icon: Brain,
      color: 'text-blue-500',
    },
    {
      title: 'Skills Growth',
      value: `+${learningProgress.skillsGrowth}%`,
      icon: Activity,
      color: 'text-green-500',
    },
    {
      title: 'New Data',
      value: learningProgress.newDataCount.toLocaleString(),
      icon: Database,
      color: 'text-purple-500',
    },
    {
      title: 'Verified Facts',
      value: learningProgress.verifiedFacts.toLocaleString(),
      icon: Zap,
      color: 'text-yellow-500',
    },
  ]

  return (
    <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
      {stats.map((stat) => (
        <Card key={stat.title}>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">{stat.title}</CardTitle>
            <stat.icon className={`h-4 w-4 ${stat.color}`} />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{stat.value}</div>
          </CardContent>
        </Card>
      ))}

      <Card className="md:col-span-2 lg:col-span-4">
        <CardHeader>
          <CardTitle>System Health</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4 md:grid-cols-3">
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <Cpu className="h-4 w-4" />
                <span className="text-sm font-medium">CPU Usage</span>
              </div>
              <div className="h-2 w-full bg-secondary rounded-full overflow-hidden">
                <div
                  className="h-full bg-primary transition-all"
                  style={{ width: `${systemHealth.cpu}%` }}
                />
              </div>
              <p className="text-xs text-muted-foreground">{systemHealth.cpu}%</p>
            </div>

            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <HardDrive className="h-4 w-4" />
                <span className="text-sm font-medium">Disk Usage</span>
              </div>
              <div className="h-2 w-full bg-secondary rounded-full overflow-hidden">
                <div
                  className="h-full bg-primary transition-all"
                  style={{ width: `${systemHealth.disk}%` }}
                />
              </div>
              <p className="text-xs text-muted-foreground">{systemHealth.disk}%</p>
            </div>

            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <Activity className="h-4 w-4" />
                <span className="text-sm font-medium">Forgetting Score</span>
              </div>
              <div className="h-2 w-full bg-secondary rounded-full overflow-hidden">
                <div
                  className="h-full bg-destructive transition-all"
                  style={{ width: `${learningProgress.forgettingScore * 100}%` }}
                />
              </div>
              <p className="text-xs text-muted-foreground">
                {(learningProgress.forgettingScore * 100).toFixed(1)}%
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
