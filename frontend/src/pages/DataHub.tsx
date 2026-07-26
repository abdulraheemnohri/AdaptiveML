import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Database, Plus } from 'lucide-react'
import { Link } from 'react-router-dom'

function DataHubPage() {
  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold">Data Hub</h1>
          <p className="text-muted-foreground">Manage all your data sources and datasets</p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" asChild><Link to="/data/sources"><Plus className="h-4 w-4 mr-2" />Add Source</Link></Button>
          <Button asChild><Link to="/data/datasets"><Plus className="h-4 w-4 mr-2" />Add Dataset</Link></Button>
        </div>
      </div>
      <div className="grid gap-4 md:grid-cols-4">
        <Card><CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2"><CardTitle className="text-sm font-medium">Total Sources</CardTitle><Database className="h-4 w-4 text-muted-foreground" /></CardHeader><CardContent><div className="text-2xl font-bold">12</div><p className="text-xs text-muted-foreground">Active: 8</p></CardContent></Card>
        <Card><CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2"><CardTitle className="text-sm font-medium">Total Datasets</CardTitle><Database className="h-4 w-4 text-muted-foreground" /></CardHeader><CardContent><div className="text-2xl font-bold">45</div><p className="text-xs text-muted-foreground">Validated: 38</p></CardContent></Card>
        <Card><CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2"><CardTitle className="text-sm font-medium">Total Samples</CardTitle><Database className="h-4 w-4 text-muted-foreground" /></CardHeader><CardContent><div className="text-2xl font-bold">128,430</div><p className="text-xs text-muted-foreground">Processed: 115,670</p></CardContent></Card>
        <Card><CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2"><CardTitle className="text-sm font-medium">Storage Used</CardTitle><Database className="h-4 w-4 text-muted-foreground" /></CardHeader><CardContent><div className="text-2xl font-bold">12.4 GB</div><p className="text-xs text-muted-foreground">Available: 87.6 GB</p></CardContent></Card>
      </div>
      <Card><CardHeader><CardTitle>Quick Actions</CardTitle></CardHeader><CardContent><div className="flex flex-wrap gap-2"><Button variant="outline" size="sm" asChild><Link to="/data/sources"><Plus className="h-4 w-4 mr-2" />Add Data Source</Link></Button><Button variant="outline" size="sm" asChild><Link to="/data/datasets"><Plus className="h-4 w-4 mr-2" />Add Dataset</Link></Button></div></CardContent></Card>
      <Card><CardHeader className="flex justify-between items-center"><CardTitle>Data Sources</CardTitle><Button variant="outline" size="sm" asChild><Link to="/data/sources">View All</Link></Button></CardHeader><CardContent><p className="text-muted-foreground text-center py-8">No data sources configured. Add your first data source to get started.</p></CardContent></Card>
      <Card><CardHeader className="flex justify-between items-center"><CardTitle>Recent Datasets</CardTitle><Button variant="outline" size="sm" asChild><Link to="/data/datasets">View All</Link></Button></CardHeader><CardContent><p className="text-muted-foreground text-center py-8">No datasets available. Create your first dataset to get started.</p></CardContent></Card>
    </div>
  )
}

export default DataHubPage