import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Plus, Database, FileText, CheckCircle, XCircle, Clock } from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { Link } from 'react-router-dom'

const mockDatasets = [
  { id: '1', name: 'Wikipedia Articles', description: 'Collection of Wikipedia articles', status: 'ready', source: 'Web', samples: 12450, size: '2.4 GB', quality: 92, lastUpdated: '2 hours ago' },
  { id: '2', name: 'ArXiv Papers', description: 'Scientific papers from ArXiv', status: 'processing', source: 'RSS', samples: 8920, size: '1.8 GB', quality: 88, lastUpdated: '5 hours ago' },
  { id: '3', name: 'Code Examples', description: 'Python code examples', status: 'validating', source: 'GitHub', samples: 5430, size: '800 MB', quality: 95, lastUpdated: '1 day ago' },
  { id: '4', name: 'Urdu Text', description: 'Urdu language text corpus', status: 'quarantined', source: 'Local', samples: 2340, size: '500 MB', quality: 65, lastUpdated: '3 days ago' },
]

const statusIcons = { ready: <CheckCircle className="h-4 w-4 text-green-500" />, processing: <Clock className="h-4 w-4 text-blue-500" />, validating: <Clock className="h-4 w-4 text-yellow-500" />, quarantined: <XCircle className="h-4 w-4 text-red-500" /> }

function DatasetsPage() {
  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <div><h1 className="text-3xl font-bold">Datasets</h1><p className="text-muted-foreground">View and manage all your datasets</p></div>
        <Button asChild><Link to="#"><Plus className="h-4 w-4 mr-2" />Add Dataset</Link></Button>
      </div>
      <div className="grid gap-4 md:grid-cols-4">
        <Card><CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2"><CardTitle className="text-sm font-medium">Total Datasets</CardTitle><Database className="h-4 w-4 text-muted-foreground" /></CardHeader><CardContent><div className="text-2xl font-bold">45</div><p className="text-xs text-muted-foreground">Active: 38</p></CardContent></Card>
        <Card><CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2"><CardTitle className="text-sm font-medium">Total Samples</CardTitle><FileText className="h-4 w-4 text-muted-foreground" /></CardHeader><CardContent><div className="text-2xl font-bold">128,430</div><p className="text-xs text-muted-foreground">Validated: 115,670</p></CardContent></Card>
        <Card><CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2"><CardTitle className="text-sm font-medium">Storage Used</CardTitle><Database className="h-4 w-4 text-muted-foreground" /></CardHeader><CardContent><div className="text-2xl font-bold">12.4 GB</div><p className="text-xs text-muted-foreground">Available: 87.6 GB</p></CardContent></Card>
        <Card><CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2"><CardTitle className="text-sm font-medium">Avg Quality</CardTitle><CheckCircle className="h-4 w-4 text-muted-foreground" /></CardHeader><CardContent><div className="text-2xl font-bold">88%</div><p className="text-xs text-muted-foreground">Trust: 92%</p></CardContent></Card>
      </div>
      <Card>
        <CardHeader className="flex justify-between items-center"><CardTitle>All Datasets ({mockDatasets.length})</CardTitle><Button variant="outline" size="sm" asChild><Link to="#">View All</Link></Button></CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead><tr className="border-b"><th className="text-left p-4 font-semibold">Name</th><th className="text-left p-4 font-semibold">Status</th><th className="text-left p-4 font-semibold">Source</th><th className="text-left p-4 font-semibold">Samples</th><th className="text-left p-4 font-semibold">Size</th><th className="text-left p-4 font-semibold">Quality</th><th className="text-left p-4 font-semibold">Last Updated</th><th className="text-left p-4 font-semibold">Actions</th></tr></thead>
              <tbody>
                {mockDatasets.map((dataset) => (
                  <tr key={dataset.id} className="border-b">
                    <td className="p-4"><div className="font-medium">{dataset.name}</div><div className="text-sm text-muted-foreground truncate max-w-[200px]">{dataset.description}</div></td>
                    <td className="p-4"><div className="flex items-center gap-2">{statusIcons[dataset.status as keyof typeof statusIcons]}<Badge variant={dataset.status === 'ready' ? 'default' : dataset.status === 'quarantined' ? 'destructive' : 'outline'}>{dataset.status}</Badge></div></td>
                    <td className="p-4 text-sm text-muted-foreground">{dataset.source}</td>
                    <td className="p-4 text-sm text-muted-foreground">{dataset.samples.toLocaleString()}</td>
                    <td className="p-4 text-sm text-muted-foreground">{dataset.size}</td>
                    <td className="p-4"><Badge variant={dataset.quality >= 90 ? 'default' : dataset.quality >= 70 ? 'secondary' : 'destructive'}>{dataset.quality}%</Badge></td>
                    <td className="p-4 text-sm text-muted-foreground">{dataset.lastUpdated}</td>
                    <td className="p-4"><div className="flex gap-2"><Button variant="ghost" size="icon"><Edit className="h-4 w-4" /></Button><Button variant="ghost" size="icon"><MoreVertical className="h-4 w-4" /></Button></div></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

export default DatasetsPage