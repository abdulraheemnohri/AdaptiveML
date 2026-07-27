"use client"

import { useCallback, useState } from "react"

type ToasterToast = {
  id: string
  title?: React.ReactNode
  description?: React.ReactNode
  action?: {
    label: string
    altText: string
    onClick: () => void
  }
}

let toastId = 0

function generateId() {
  return ++toastId
}

export function useToast() {
  const [toasts, setToasts] = useState<ToasterToast[]>([])

  const toast = useCallback(
    ({
      title,
      description,
      action,
    }: {
      title?: React.ReactNode
      description?: React.ReactNode
      action?: {
        label: string
        altText: string
        onClick: () => void
      }
    }) => {
      const id = String(generateId())
      setToasts((toasts) => [
        ...toasts,
        {
          id,
          title,
          description,
          action,
        },
      ])
      return id
    },
    []
  )

  const dismiss = useCallback((toastId?: string) => {
    if (toastId) {
      setToasts((toasts) => toasts.filter((t) => t.id !== toastId))
    } else {
      setToasts([])
    }
  }, [])

  return {
    toast,
    toasts,
    dismiss,
  }
}