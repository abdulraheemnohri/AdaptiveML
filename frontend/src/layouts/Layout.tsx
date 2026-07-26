import { Outlet } from 'react-router-dom'
import Sidebar from '@/components/Sidebar'
import TopBar from '@/components/TopBar'

function Layout() {
  return (
    <div className="min-h-screen bg-background antialiased">
      <div className="flex">
        <Sidebar />
        <div className="flex-1 flex flex-col">
          <TopBar />
          <main className="flex-1 p-4 md:p-6 lg:p-8">
            <Outlet />
          </main>
          <footer className="border-t py-4 px-6 text-sm text-muted-foreground">
            <div className="flex justify-between items-center">
              <p>Adaptive Omni ML Platform v0.1.0</p>
              <p>Built with Qwen2.5-Omni-3B</p>
            </div>
          </footer>
        </div>
      </div>
    </div>
  )
}

export default Layout