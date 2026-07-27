import { useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Badge } from '@/components/ui/badge'
import { Search, Plus, Globe, Rss, PlayCircle, FileText, Database, GitBranch, MoreVertical, Edit, Trash2 } from 'lucide-react'

const mockSources = [
  { id: '1', name: 'Wikipedia', type: 'web', url: 'https://en.wikipedia.org', status: 'active', lastCollected: '2 hours ago', totalCollected: 12450 },
  { id: '2', name: 'ArXiv Papers', type: 'rss', url: 'https://arxiv.org/rss', status: 'active', lastCollected: '5 hours ago', totalCollected: 8920 },
  { id: '3', name: 'GitHub Repo', type: 'github', url: 'https://github.com/example/repo', status: 'paused', lastCollected: '1 day ago', totalCollected: 5430 },
  { id: '4', name: 'Local Documents', type: 'local_folder', path: '/data/documents', status: 'disabled', lastCollected: '3 days ago', totalCollected: 2340 },
]

const sourceTypes = [
  { value: 'web', label: 'Web', icon: <Globe className="h-4 w-4" /> },
  { value: 'rss', label: 'RSS Feed', icon: <Rss className="h-4 w-4" /> },
  { value: 'youtube', label: 'YouTube', icon: <PlayCircle className="h-4 w-4" /> },
  { value: 'pdf', label: 'PDF', icon: <FileText className="h-4 w-4" /> },
  { value: 'csv', label: 'CSV', icon: <Database className="h-4 w-4" /> },
  { value: 'github', label: 'GitHub', icon: <GitBranch className="h-4 w-4" /> },
]

function DataSourcesPage() {
  const [searchQuery, setSearchQuery] = useState('')
  const filteredSources = mockSources.filter(source => source.name.toLowerCase().includes(searchQuery.toLowerCase()) || source.url?.toLowerCase().includes(searchQuery.toLowerCase()))

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <div><h1 className="text-3xl font-bold">Data Sources</h1><p className="text-muted-foreground">Configure and manage your data sources</p></div>
        <Button><Plus className="h-4 w-4 mr-2" />Add Data Source</Button>
      </div>
      <Card><CardContent className="pt-6"><div className="relative max-w-md"><Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" /><Input type="search" placeholder="Search data sources..." className="pl-10" value={searchQuery} onChange={(e) => setSearchQuery(e.target.value)} /></div></CardContent></Card>
      <Card>
        <CardHeader><CardTitle>All Data Sources ({filteredSources.length})</CardTitle><CardDescription>Configure automatic data collection from various sources</CardDescription></CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead><tr className="border-b"><th className="text-left p-4 font-semibold">Name</th><th className="text-left p-4 font-semibold">Type</th><th className="text-left p-4 font-semibold">Status</th><th className="text-left p-4 font-semibold">Last Collected</th><th className="text-left p-4 font-semibold">Total Collected</th><th className="text-left p-4 font-semibold">Actions</th></tr></thead>
              <tbody>
                {filteredSources.map((source) => (
                  <tr key={source.id} className="border-b">
                    <td className="p-4"><div className="font-medium">{source.name}</div><div className="text-sm text-muted-foreground">{source.url || source.path}</div></td>
                    <td className="p-4"><Badge variant="outline">{sourceTypes.find(t => t.value === source.type)?.label || source.type}</Badge></td>
                    <td className="p-4"><Badge variant={source.status === 'active' ? 'default' : source.status === 'paused' ? 'secondary' : 'outline'}>{source.status}</Badge></td>
                    <td className="p-4 text-sm text-muted-foreground">{source.lastCollected}</td>
                    <td className="p-4 text-sm text-muted-foreground">{source.totalCollected.toLocaleString()}</td>
                    <td className="p-4"><div className="flex gap-2"><Button variant="ghost" size="icon"><PlayCircle className="h-4 w-4" /></Button><Button variant="ghost" size="icon"><Edit className="h-4 w-4" /></Button><Button variant="ghost" size="icon"><MoreVertical className="h-4 w-4" /></Button></div></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {filteredSources.length === 0 && <div className="text-center py-8 text-muted-foreground">No data sources found matching your search.</div>}
        </CardContent>
      </Card>
    </div>
  )
}

export default DataSourcesPage