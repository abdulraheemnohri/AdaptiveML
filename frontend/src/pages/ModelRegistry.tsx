import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Brain, Crown, Clock, CheckCircle, XCircle, TrendingUp, Package, Rocket, HardDrive } from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { Link } from 'react-router-dom'

const mockModels = [
  { id: '1', name: 'Qwen2.5-Omni-3B', version: 'v1.0.0', type: 'base', status: 'production', baseModel: '-', createdAt: '2024-01-01', size: '14 GB', parameters: '3B', overallScore: 85 },
  { id: '2', name: 'Qwen-Omni-Urdu', version: 'v1.1.0', type: 'adapter', status: 'approved', baseModel: 'Qwen2.5-Omni-3B', createdAt: '2024-01-15', size: '1.2 GB', parameters: '3B + LoRA', overallScore: 88 },
  { id: '3', name: 'Qwen-Omni-Coding', version: 'v1.2.0', type: 'fine_tuned', status: 'candidate', baseModel: 'Qwen2.5-Omni-3B', createdAt: '2024-01-16', size: '14 GB', parameters: '3B', overallScore: 91 },
  { id: '4', name: 'Qwen-Omni-Multimodal', version: 'v1.3.0', type: 'fine_tuned', status: 'testing', baseModel: 'Qwen2.5-Omni-3B', createdAt: '2024-01-17', size: '14 GB', parameters: '3B', overallScore: 89 },
]

const statusIcons = { production: <Crown className="h-4 w-4 text-yellow-500" />, approved: <CheckCircle className="h-4 w-4 text-green-500" />, candidate: <Rocket className="h-4 w-4 text-blue-500" />, testing: <Clock className="h-4 w-4 text-yellow-500" />, draft: <Package className="h-4 w-4 text-gray-500" />, rejected: <XCircle className="h-4 w-4 text-red-500" /> }
const typeBadges = { base: 'default', adapter: 'secondary', fine_tuned: 'outline' }

function ModelRegistryPage() {
  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <div><h1 className="text-3xl font-bold">Model Registry</h1><p className="text-muted-foreground">View and manage all your AI models</p></div>
        <div className="flex gap-2"><Button variant="outline" asChild><Link to="/models/compare">Compare Models</Link></Button><Button><Plus className="h-4 w-4 mr-2" />Register Model</Button></div>
      </div>
      <div className="grid gap-4 md:grid-cols-5">
        <Card><CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2"><CardTitle className="text-sm font-medium">Total Models</CardTitle><Brain className="h-4 w-4 text-muted-foreground" /></CardHeader><CardContent><div className="text-2xl font-bold">{mockModels.length}</div><p className="text-xs text-muted-foreground">Active: 3</p></CardContent></Card>
        <Card><CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2"><CardTitle className="text-sm font-medium">Production</CardTitle><Crown className="h-4 w-4 text-muted-foreground" /></CardHeader><CardContent><div className="text-2xl font-bold">1</div><p className="text-xs text-muted-foreground">Qwen2.5-Omni-3B</p></CardContent></Card>
        <Card><CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2"><CardTitle className="text-sm font-medium">Avg Score</CardTitle><TrendingUp className="h-4 w-4 text-muted-foreground" /></CardHeader><CardContent><div className="text-2xl font-bold">88%</div><p className="text-xs text-muted-foreground">Overall</p></CardContent></Card>
        <Card><CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2"><CardTitle className="text-sm font-medium">Retention</CardTitle><CheckCircle className="h-4 w-4 text-muted-foreground" /></CardHeader><CardContent><div className="text-2xl font-bold">98%</div><p className="text-xs text-muted-foreground">Anti-Forgetting</p></CardContent></Card>
        <Card><CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2"><CardTitle className="text-sm font-medium">Storage</CardTitle><HardDrive className="h-4 w-4 text-muted-foreground" /></CardHeader><CardContent><div className="text-2xl font-bold">42 GB</div><p className="text-xs text-muted-foreground">Used</p></CardContent></Card>
      </div>
      <Card>
        <CardHeader><CardTitle>All Models ({mockModels.length})</CardTitle><CardDescription>Complete model registry with version history</CardDescription></CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead><tr className="border-b"><th className="text-left p-4 font-semibold">Name</th><th className="text-left p-4 font-semibold">Version</th><th className="text-left p-4 font-semibold">Type</th><th className="text-left p-4 font-semibold">Status</th><th className="text-left p-4 font-semibold">Base Model</th><th className="text-left p-4 font-semibold">Score</th><th className="text-left p-4 font-semibold">Actions</th></tr></thead>
              <tbody>
                {mockModels.map((model) => (
                  <tr key={model.id} className="border-b">
                    <td className="p-4"><div className="font-medium">{model.name}</div><div className="text-xs text-muted-foreground">{model.createdAt}</div></td>
                    <td className="p-4 text-sm text-muted-foreground">{model.version}</td>
                    <td className="p-4"><Badge variant={typeBadges[model.type as keyof typeof typeBadges]}>{model.type.replace('_', ' ')}</Badge></td>
                    <td className="p-4"><div className="flex items-center gap-2">{statusIcons[model.status as keyof typeof statusIcons]}<Badge variant={model.status === 'production' ? 'default' : 'outline'}>{model.status}</Badge></div></td>
                    <td className="p-4 text-sm text-muted-foreground">{model.baseModel}</td>
                    <td className="p-4"><Badge variant={model.overallScore >= 90 ? 'default' : 'secondary'}>{model.overallScore}%</Badge></td>
                    <td className="p-4"><div className="flex gap-2"><Button variant="ghost" size="icon"><Rocket className="h-4 w-4" /></Button><Button variant="ghost" size="icon"><TrendingUp className="h-4 w-4" /></Button></div></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>
      <Card>
        <CardHeader><CardTitle>Production Model</CardTitle><CardDescription>Currently deployed model serving user requests</CardDescription></CardHeader>
        <CardContent>
          <div className="flex items-center gap-6">
            <div className="w-20 h-20 bg-primary rounded-lg flex items-center justify-center"><Brain className="h-10 w-10 text-primary-foreground" /></div>
            <div className="flex-1">
              <h3 className="text-xl font-bold">Qwen2.5-Omni-3B</h3>
              <p className="text-muted-foreground">v1.0.0 - Base Model</p>
              <div className="flex flex-wrap gap-4 mt-4 text-sm"><div><p className="font-semibold">Status</p><p className="text-muted-foreground">Production</p></div><div><p className="font-semibold">Parameters</p><p className="text-muted-foreground">3B</p></div><div><p className="font-semibold">Size</p><p className="text-muted-foreground">14 GB</p></div><div><p className="font-semibold">Overall Score</p><p className="text-muted-foreground">85%</p></div></div>
            </div>
            <div className="flex gap-2"><Button variant="outline">View Details</Button><Button>Deploy</Button></div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

export default ModelRegistryPage