import { Link } from 'react-router-dom'
import { Button } from '@/components/ui/button'
import { AlertTriangle } from 'lucide-react'

function NotFoundPage() {
  return (
    <div className="flex flex-col items-center justify-center min-h-[calc(100vh-200px)]">
      <div className="text-center">
        <div className="inline-flex items-center justify-center w-20 h-20 bg-muted rounded-full mb-6">
          <AlertTriangle className="h-10 w-10 text-muted-foreground" />
        </div>
        <h1 className="text-4xl font-bold mb-2">404 - Page Not Found</h1>
        <p className="text-muted-foreground mb-6">The page you're looking for doesn't exist or has been moved.</p>
        <div className="flex gap-4 justify-center">
          <Button asChild><Link to="/">Go to Dashboard</Link></Button>
          <Button variant="outline" asChild><Link to="/settings">Settings</Link></Button>
        </div>
      </div>
    </div>
  )
}

export default NotFoundPage