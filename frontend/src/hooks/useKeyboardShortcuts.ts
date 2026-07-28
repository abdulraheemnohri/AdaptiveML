import { useEffect, useState } from 'react';

export const useKeyboardShortcuts = () => {
  const [shortcut, setShortcut] = useState<string | null>(null);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Command/Ctrl + K - Open command palette
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault();
        setShortcut('command-palette');
      }

      // Command/Ctrl + B - Toggle sidebar
      if ((e.metaKey || e.ctrlKey) && e.key === 'b') {
        e.preventDefault();
        setShortcut('toggle-sidebar');
      }

      // Command/Ctrl + T - Switch to training mode
      if ((e.metaKey || e.ctrlKey) && e.key === 't') {
        e.preventDefault();
        setShortcut('training-mode');
      }

      // Command/Ctrl + S - Switch to serving mode
      if ((e.metaKey || e.ctrlKey) && e.key === 's') {
        e.preventDefault();
        setShortcut('serving-mode');
      }

      // Escape - Close modals/palettes
      if (e.key === 'Escape') {
        setShortcut('close');
      }

      // ? - Help
      if (e.key === '?' && !e.metaKey && !e.ctrlKey) {
        e.preventDefault();
        setShortcut('help');
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  return shortcut;
};
