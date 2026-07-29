import { useEffect, useRef, useCallback } from 'react';
import { useAppStore } from '@/stores/appStore';

export const useWebSocket = (url: string) => {
  const wsRef = useRef<WebSocket | null>(null);
  const { setWsConnected, addNotification } = useAppStore();

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return;
    }

    try {
      const wsUrl = url.startsWith('ws') ? url : `ws://${window.location.host}${url}`;
      wsRef.current = new WebSocket(wsUrl);

      wsRef.current.onopen = () => {
        setWsConnected(true);
        addNotification({
          title: 'Connected',
          message: 'Real-time updates enabled',
          type: 'success',
          timestamp: new Date().toISOString(),
        });
      };

      wsRef.current.onclose = () => {
        setWsConnected(false);
        // Attempt reconnection after 5 seconds
        setTimeout(connect, 5000);
      };

      wsRef.current.onerror = () => {
        addNotification({
          title: 'Connection Error',
          message: 'WebSocket connection failed',
          type: 'error',
          timestamp: new Date().toISOString(),
        });
      };
    } catch (error) {
      console.error('WebSocket connection error:', error);
    }
  }, [url, setWsConnected, addNotification]);

  const disconnect = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
      setWsConnected(false);
    }
  }, [setWsConnected]);

  const send = useCallback((data: unknown) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(data));
    }
  }, []);

  useEffect(() => {
    connect();
    return () => disconnect();
  }, [connect, disconnect]);

  return { connect, disconnect, send, ws: wsRef.current };
};
