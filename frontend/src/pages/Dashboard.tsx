import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Progress } from '@/components/ui/progress'
import { Button } from '@/components/ui/button'
import { Brain, Database, TrendingUp, ShieldCheck, Activity, Clock } from 'lucide-react'
import { LineChart, BarChart, PieChart } from '@/components/Charts'

const mockStats = {
  currentModel: 'Qwen2.5-Omni-3B v1.2.0',
  learningProgress: 78,
  knowledgeGrowth: '+12%',
  forgettingScore: '0.4%',
  modelQuality: 92,
}

const mockBrainEvolution = {
  knowledge: '+12%',
  skills: '+8%',
  languages: '+3',
  newData: 24830,
  verifiedFacts: 12400,
  learningTasks: 34,
  forgetting: '0.4%',
}

const mockChartData = {
  learningProgress: [
    { name: 'Jan', value: 65 },
    { name: 'Feb', value: 70 },
    { name: 'Mar', value: 75 },
    { name: 'Apr', value: 78 },
    { name: 'May', value: 82 },
    { name: 'Jun', value: 85 },
  ],
  knowledgeDistribution: [
    { name: 'Text', value: 45 },
    { name: 'Image', value: 25 },
    { name: 'Audio', value: 15 },
    { name: 'Video', value: 10 },
    { name: 'Code', value: 5 },
  ],
  modelPerformance: [
    { name: 'Reasoning', value: 92 },
    { name: 'Coding', value: 88 },
    { name: 'Multimodal', value: 85 },
    { name: 'Language', value: 90 },
    { name: 'Safety', value: 95 },
  ],
}

function StatCard({ title, value, icon, trend }: { title: string; value: string; icon: React.ReactNode; trend?: string }) {
  return (
    <Card className="transition-shadow hover:shadow-md">
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-sm font-medium">{title}</CardTitle>
        {icon}
      </CardHeader>
      <CardContent>
        <div className="text-2xl font-bold">{value}</div>
        {trend && <p className="text-xs text-muted-foreground mt-1">{trend}</p>}
      </CardContent>
    </Card>
  )
}

function DashboardPage() {
  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold">Dashboard</h1>
          <p className="text-muted-foreground">Welcome to Adaptive Omni ML Platform</p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline">Export Report</Button>
          <Button>Start Learning Cycle</Button>
        </div>
      </div>

      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <StatCard title="Current Model" value={mockStats.currentModel} icon={<Brain className="h-4 w-4 text-muted-foreground" />} />
        <StatCard title="Learning Progress" value={`${mockStats.learningProgress}%`} icon={<TrendingUp className="h-4 w-4 text-muted-foreground" />} trend="+5% this week" />
        <StatCard title="Knowledge Growth" value={mockStats.knowledgeGrowth} icon={<Database className="h-4 w-4 text-muted-foreground" />} />
        <StatCard title="Forgetting Score" value={mockStats.forgettingScore} icon={<ShieldCheck className="h-4 w-4 text-muted-foreground" />} />
      </div>

      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
        <Card>
          <CardHeader><CardTitle>Learning Progress</CardTitle></CardHeader>
          <CardContent><LineChart data={mockChartData.learningProgress} /></CardContent>
        </Card>
        <Card>
          <CardHeader><CardTitle>Knowledge Distribution</CardTitle></CardHeader>
          <CardContent><PieChart data={mockChartData.knowledgeDistribution} /></CardContent>
        </Card>
        <Card>
          <CardHeader><CardTitle>Model Performance</CardTitle></CardHeader>
          <CardContent><BarChart data={mockChartData.modelPerformance} /></CardContent>
        </Card>
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card>
          <CardHeader><CardTitle>Live Brain Evolution</CardTitle></CardHeader>
          <CardContent>
            <div className="grid gap-4 md:grid-cols-2">
              <div className="space-y-4">
                <div className="flex items-center gap-4"><span className="text-green-500">↑</span><div><p className="text-sm text-muted-foreground">Knowledge</p><p className="text-xl font-bold">{mockBrainEvolution.knowledge}</p></div></div>
                <div className="flex items-center gap-4"><span className="text-green-500">↑</span><div><p className="text-sm text-muted-foreground">Skills</p><p className="text-xl font-bold">{mockBrainEvolution.skills}</p></div></div>
                <div className="flex items-center gap-4"><span className="text-green-500">↑</span><div><p className="text-sm text-muted-foreground">Languages</p><p className="text-xl font-bold">{mockBrainEvolution.languages}</p></div></div>
              </div>
              <div className="space-y-4">
                <div className="flex items-center gap-4"><Database className="h-5 w-5" /><div><p className="text-sm text-muted-foreground">New Data</p><p className="text-xl font-bold">{mockBrainEvolution.newData.toLocaleString()}</p></div></div>
                <div className="flex items-center gap-4"><ShieldCheck className="h-5 w-5" /><div><p className="text-sm text-muted-foreground">Verified Facts</p><p className="text-xl font-bold">{mockBrainEvolution.verifiedFacts.toLocaleString()}</p></div></div>
                <div className="flex items-center gap-4"><Activity className="h-5 w-5" /><div><p className="text-sm text-muted-foreground">Learning Tasks</p><p className="text-xl font-bold">{mockBrainEvolution.learningTasks}</p></div></div>
              </div>
            </div>
            <div className="mt-6 pt-4 border-t"><div className="flex items-center gap-4"><Clock className="h-5 w-5" /><div><p className="text-sm text-muted-foreground">Forgetting</p><p className="text-xl font-bold">{mockBrainEvolution.forgetting}</p></div></div><Progress value={4} className="mt-2 h-2" /></div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader><CardTitle>Forgetting Firewall</CardTitle></CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="flex justify-between items-center"><span className="text-sm font-medium">Old Knowledge</span><span className="text-sm font-bold">98%</span></div>
              <Progress value={98} className="h-3" />
              <div className="flex justify-between items-center"><span className="text-sm font-medium">New Knowledge</span><span className="text-sm font-bold">91%</span></div>
              <Progress value={91} className="h-3" />
              <div className="flex justify-between items-center"><span className="text-sm font-medium text-red-500">Forgetting Risk</span><span className="text-sm font-bold text-red-500">4%</span></div>
              <Progress value={4} className="h-3" />
              <div className="pt-4 text-center"><p className="text-sm text-muted-foreground">Anti-Forgetting Protection: <span className="text-green-500 font-semibold">ACTIVE</span></p></div>
            </div>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader><CardTitle>Quick Actions</CardTitle></CardHeader>
        <CardContent><div className="flex flex-wrap gap-2"><Button variant="outline" size="sm">Collect New Data</Button><Button variant="outline" size="sm">Start Training</Button><Button variant="outline" size="sm">Run Evaluation</Button><Button variant="outline" size="sm">Check Forgetting</Button><Button variant="outline" size="sm">Deploy Model</Button></div></CardContent>
      </Card>
    </div>
  )
}

export default DashboardPage
