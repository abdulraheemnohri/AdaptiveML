import { useEffect, useState } from 'react'
import { THEME_KEY, Theme, getFromLocalStorage, setToLocalStorage } from '@/lib/utils'

export function useTheme() {
  const [theme, setThemeState] = useState<Theme>(() => {
    const savedTheme = getFromLocalStorage<Theme>(THEME_KEY, 'system')
    return savedTheme
  })

  useEffect(() => {
    const root = window.document.documentElement
    root.classList.remove('light', 'dark')
    if (theme === 'system') {
      const systemTheme = window.matchMedia('(prefers-color-scheme: dark)').matches 
        ? 'dark' 
        : 'light'
      root.classList.add(systemTheme)
    } else {
      root.classList.add(theme)
    }
    setToLocalStorage(THEME_KEY, theme)
  }, [theme])

  const setTheme = (newTheme: Theme) => {
    setThemeState(newTheme)
  }

  return { theme, setTheme }
}