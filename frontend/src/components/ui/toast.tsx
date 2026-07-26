"use client"

import * as React from "react"
import * as ToastPrimitives from "@radix-ui/react-toast"
import { cva, type VariantProps } from "class-variance-authority"
import { X } from "lucide-react"
import { cn } from '@/lib/utils'

const ToastProvider = ToastPrimitives.ToastProvider

const ToastViewport = React.forwardRef<
  React.ElementRef<typeof ToastPrimitives.ToastViewport>,
  React.ComponentPropsWithoutRef<typeof ToastPrimitives.ToastViewport>
>(({ className, ...props }, ref) => (
  <ToastPrimitives.ToastViewport
    ref={ref}
    className={cn(
      "fixed top-0 z-[100] flex max-h-screen w-full flex-col-reverse p-4 sm:bottom-0 sm:right-0 sm:top-auto sm:flex-col md:max-w-[420px] lg:max-w-[460px]",
      className
    )}
    {...props}
  />
))
ToastViewport.displayName = ToastPrimitives.ToastViewport.displayName

const toastVariants = cva(
  "group pointer-events-auto relative flex w-full max-w-[80vw] items-center justify-between space-x-4 overflow-hidden rounded-md border p-6 pr-8 shadow-lg transition-all data-[swipe=cancel]:translate-x-0 data-[swipe=end]:translate-x-[var(--radix-toast-swipe-end-x)] data-[swipe=move]:translate-x-[var(--radix-toast-swipe-move-x)] data-[swipe=start]:translate-x-[var(--radix-toast-swipe-start-x)] data-[state=closed]:fade-out-0 data-[state=closed]:slide-out-to-right-full data-[state=open]:animate-in data-[state=open]:slide-in-from-right-full data-[swipe=end]:fade-out-0 data-[swipe=move]:transition-none data-[swipe=start]:slide-in-from-left-full",
  {
    variants: {
      variant: {
        default: "bg-background text-foreground",
        destructive: "bg-destructive text-destructive-foreground",
      },
    },
    defaultVariants: {
      variant: "default",
    },
  }
)

const Toast = React.forwardRef<
  React.ElementRef<typeof ToastPrimitives.Toast>,
  React.ComponentPropsWithoutRef<typeof ToastPrimitives.Toast> & VariantProps<typeof toastVariants>
>(({ className, variant, ...props }, ref) => (
  <ToastPrimitives.Toast
    ref={ref}
    className={cn(toastVariants({ variant }), className)}
    {...props}
  />
))
Toast.displayName = ToastPrimitives.Toast.displayName

const ToastAction = React.forwardRef<
  React.ElementRef<typeof ToastPrimitives.ToastAction>,
  React.ComponentPropsWithoutRef<typeof ToastPrimitives.ToastAction>
>(({ className, ...props }, ref) => (
  <ToastPrimitives.ToastAction
    ref={ref}
    className={cn(
      "inline-flex h-8 shrink-0 items-center justify-center rounded-md bg-primary px-3 text-sm font-medium text-primary-foreground transition-colors hover:bg-primary/90 focus:outline-none focus:ring-2 focus:ring-primary focus:ring-offset-2",
      className
    )}
    {...props}
  />
))
ToastAction.displayName = ToastPrimitives.ToastAction.displayName

const ToastClose = React.forwardRef<
  React.ElementRef<typeof ToastPrimitives.ToastClose>,
  React.ComponentPropsWithoutRef<typeof ToastPrimitives.ToastClose>
>(({ className, ...props }, ref) => (
  <ToastPrimitives.ToastClose
    ref={ref}
    className={cn(
      "absolute top-1 right-1 rounded-md p-1 text-foreground/70 opacity-0 transition-opacity hover:text-foreground focus:opacity-100 focus:outline-none focus:ring-2 group-hover:opacity-100 group-focus:opacity-100",
      className
    )}
    toast-close=""
    {...props}
  >
    <X className="h-4 w-4" />
  </ToastPrimitives.ToastClose>
))
ToastClose.displayName = ToastPrimitives.ToastClose.displayName

const ToastTitle = React.forwardRef<
  React.ElementRef<typeof ToastPrimitives.ToastTitle>,
  React.ComponentPropsWithoutRef<typeof ToastPrimitives.ToastTitle>
>(({ className, ...props }, ref) => (
  <ToastPrimitives.ToastTitle
    ref={ref}
    className={cn("font-semibold", className)}
    {...props}
  />
))
ToastTitle.displayName = ToastPrimitives.ToastTitle.displayName

const ToastDescription = React.forwardRef<
  React.ElementRef<typeof ToastPrimitives.ToastDescription>,
  React.ComponentPropsWithoutRef<typeof ToastPrimitives.ToastDescription>
>(({ className, ...props }, ref) => (
  <ToastPrimitives.ToastDescription
    ref={ref}
    className={cn("text-sm opacity-90", className)}
    {...props}
  />
))
ToastDescription.displayName = ToastPrimitives.ToastDescription.displayName

export {
  ToastProvider,
  ToastViewport,
  Toast,
  ToastAction,
  ToastClose,
  ToastTitle,
  ToastDescription,
}