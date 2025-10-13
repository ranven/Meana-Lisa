import type React from "react"
import type { Metadata } from "next"
import { Inter } from "next/font/google"
import { Analytics } from "@vercel/analytics/next"
import "./globals.css"
import { Suspense } from "react"  
import { Title } from "@/components/title"
import CornerArt from "@/components/cornerArt"

const inter = Inter({ 
  subsets: ["latin"],
  variable: "--font-inter",
  display: "swap",
})
export const metadata: Metadata = {
  title: "Meana-Lisa",
  description: "Upload and analyze paintings",
  generator: "v0.app",
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en">
      <body className={`font-sans ${inter.variable}`}>
        <div className="min-h-screen flex">
          {/* Left Column - 10% */}
          <div className="w-[20%] bg-muted/30 border-r border-border">
            <Title />
          </div>
          
          {/* Center Column - 80% */}
          <div className="w-[60%] flex flex-col">
            <main className="">
              {children}
            </main>
          </div>
          
          {/* Right Column - 10% */}
          <div className="w-[20%] bg-muted/30 border-l border-border">
            <CornerArt />
          </div>
        </div>
        <Analytics />
      </body>
    </html>
  )
}
