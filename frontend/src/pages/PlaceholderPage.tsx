import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Construction } from 'lucide-react'

interface PlaceholderPageProps {
  title: string
  description: string
  icon?: React.ReactNode
  comingSoon?: boolean
}

function PlaceholderPage({ title, description, icon, comingSoon = true }: PlaceholderPageProps) {
  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <div><h1 className="text-3xl font-bold">{title}</h1><p className="text-muted-foreground">{description}</p></div>
        <Button variant="outline">Learn More</Button>
      </div>
      <Card><CardContent className="pt-8 pb-8"><div className="text-center"><div className="inline-flex items-center justify-center w-20 h-20 bg-muted rounded-full mb-6 mx-auto">{icon || <Construction className="h-10 w-10 text-muted-foreground" />}</div><h2 className="text-xl font-semibold mb-2">{comingSoon ? 'Coming Soon' : 'Under Development'}</h2><p className="text-muted-foreground">{comingSoon ? 'This feature is currently under development and will be available soon.' : 'This feature is being actively developed.'}</p></div></CardContent></Card>
      <div className="grid gap-4 md:grid-cols-3">
        <Card><CardHeader><CardTitle>Feature 1</CardTitle></CardHeader><CardContent><CardDescription>Description of feature 1 and how it will enhance your experience.</CardDescription></CardContent></Card>
        <Card><CardHeader><CardTitle>Feature 2</CardTitle></CardHeader><CardContent><CardDescription>Description of feature 2 and its benefits for your workflow.</CardDescription></CardContent></Card>
        <Card><CardHeader><CardTitle>Feature 3</CardTitle></CardHeader><CardContent><CardDescription>Description of feature 3 and what it enables you to do.</CardDescription></CardContent></Card>
      </div>
    </div>
  )
}

export default PlaceholderPage