import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Progress } from '@/components/ui/progress'
import { PlayCircle, PauseCircle, StopCircle, Brain, Clock, CheckCircle, XCircle, TrendingUp } from 'lucide-react'
import { Badge } from '@/components/ui/badge'

const mockTrainingSessions = [
  { id: '1', name: 'Urdu Language Adapter', model: 'Qwen2.5-Omni-3B', dataset: 'Urdu Text Corpus', status: 'completed', progress: 100, epochs: 3, batchSize: 4, learningRate: 0.00002, startedAt: '2024-01-15 10:30', completedAt: '2024-01-15 14:45', duration: '4h 15m' },
  { id: '2', name: 'Coding Fine-tuning', model: 'Qwen2.5-Omni-3B', dataset: 'Code Examples', status: 'training', progress: 67, epochs: 5, batchSize: 8, learningRate: 0.00001, startedAt: '2024-01-16 09:00', completedAt: '-', duration: '2h 30m' },
  { id: '3', name: 'Multimodal Continual', model: 'Qwen2.5-Omni-3B', dataset: 'Mixed Modality', status: 'paused', progress: 34, epochs: 10, batchSize: 4, learningRate: 0.00001, startedAt: '2024-01-16 14:00', completedAt: '-', duration: '1h 15m' },
]

const statusConfig = {
  completed: { icon: <CheckCircle className="h-4 w-4 text-green-500" />, badge: 'default' },
  training: { icon: <TrendingUp className="h-4 w-4 text-blue-500" />, badge: 'default' },
  paused: { icon: <PauseCircle className="h-4 w-4 text-yellow-500" />, badge: 'secondary' },
  stopped: { icon: <StopCircle className="h-4 w-4 text-red-500" />, badge: 'destructive' },
}

function TrainingPage() {
  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <div><h1 className="text-3xl font-bold">Training</h1><p className="text-muted-foreground">Manage and monitor your training sessions</p></div>
        <div className="flex gap-2"><Button variant="outline"><PauseCircle className="h-4 w-4 mr-2" />Pause All</Button><Button><PlayCircle className="h-4 w-4 mr-2" />Start Training</Button></div>
      </div>
      <div className="grid gap-4 md:grid-cols-4">
        <Card><CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2"><CardTitle className="text-sm font-medium">Active Sessions</CardTitle><Brain className="h-4 w-4 text-muted-foreground" /></CardHeader><CardContent><div className="text-2xl font-bold">1</div><p className="text-xs text-muted-foreground">Running</p></CardContent></Card>
        <Card><CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2"><CardTitle className="text-sm font-medium">Completed</CardTitle><CheckCircle className="h-4 w-4 text-muted-foreground" /></CardHeader><CardContent><div className="text-2xl font-bold">12</div><p className="text-xs text-muted-foreground">This month</p></CardContent></Card>
        <Card><CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2"><CardTitle className="text-sm font-medium">Total Duration</CardTitle><Clock className="h-4 w-4 text-muted-foreground" /></CardHeader><CardContent><div className="text-2xl font-bold">45h 30m</div><p className="text-xs text-muted-foreground">This month</p></CardContent></Card>
        <Card><CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2"><CardTitle className="text-sm font-medium">GPU Usage</CardTitle><TrendingUp className="h-4 w-4 text-muted-foreground" /></CardHeader><CardContent><div className="text-2xl font-bold">78%</div><p className="text-xs text-muted-foreground">VRAM: 12GB/16GB</p></CardContent></Card>
      </div>
      <Card>
        <CardHeader><CardTitle>Training Sessions ({mockTrainingSessions.length})</CardTitle><CardDescription>Monitor and manage your training sessions</CardDescription></CardHeader>
        <CardContent>
          <div className="space-y-4">
            {mockTrainingSessions.map((session) => {
              const config = statusConfig[session.status as keyof typeof statusConfig]
              return (
                <div key={session.id} className="border rounded-lg p-4">
                  <div className="flex justify-between items-start">
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-1"><h3 className="font-semibold">{session.name}</h3><Badge variant={config.badge}>{session.status}</Badge></div>
                      <div className="flex flex-wrap gap-4 text-sm text-muted-foreground"><span>Model: {session.model}</span><span>Dataset: {session.dataset}</span><span>Epochs: {session.epochs}</span><span>Batch: {session.batchSize}</span><span>LR: {session.learningRate}</span></div>
                      <div className="mt-3"><Progress value={session.progress} className="h-2" /><p className="text-xs text-muted-foreground mt-1">Progress: {session.progress}%</p></div>
                    </div>
                    <div className="flex flex-col items-end gap-2">
                      <div className="flex gap-1"><Button variant="ghost" size="icon"><PauseCircle className="h-4 w-4" /></Button><Button variant="ghost" size="icon"><StopCircle className="h-4 w-4" /></Button></div>
                      <div className="text-xs text-muted-foreground"><p>{session.duration}</p><p>{session.startedAt}</p></div>
                    </div>
                  </div>
                </div>
              )
            })}
          </div>
        </CardContent>
      </Card>
      <Card><CardHeader><CardTitle>Quick Actions</CardTitle></CardHeader><CardContent><div className="flex flex-wrap gap-2"><Button variant="outline" size="sm"><PlayCircle className="h-4 w-4 mr-2" />Start New Session</Button><Button variant="outline" size="sm"><PauseCircle className="h-4 w-4 mr-2" />Pause All</Button><Button variant="outline" size="sm"><StopCircle className="h-4 w-4 mr-2" />Stop All</Button><Button variant="outline" size="sm"><Clock className="h-4 w-4 mr-2" />View History</Button></div></CardContent></Card>
    </div>
  )
}

export default TrainingPage