"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { TradingDashboard } from "@/components/trading-dashboard"
import { OrderBookView } from "@/components/order-book-view"
import { SignalGenerator } from "@/components/signal-generator"
import { BacktestInterface } from "@/components/backtest-interface"
import { PerformanceMonitor } from "@/components/performance-monitor"
import { IVSurfaceView } from "@/components/iv-surface-view"
import { Activity, TrendingUp, Zap, Target, BarChart3, Settings } from "lucide-react"

export default function QuantXEngine() {
  const [isConnected, setIsConnected] = useState(false)
  const [systemStatus, setSystemStatus] = useState("Initializing...")

  useEffect(() => {
    // Simulate system initialization
    const timer = setTimeout(() => {
      setIsConnected(true)
      setSystemStatus("Live Trading Ready")
    }, 2000)

    return () => clearTimeout(timer)
  }, [])

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900">
      <div className="container mx-auto p-6">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-4xl font-bold text-white mb-2">QuantX Engine</h1>
              <p className="text-slate-300">Integrated High-Frequency Trading Platform</p>
            </div>
            <div className="flex items-center gap-4">
              <Badge
                variant={isConnected ? "default" : "secondary"}
                className={isConnected ? "bg-green-600" : "bg-yellow-600"}
              >
                <Activity className="w-4 h-4 mr-1" />
                {systemStatus}
              </Badge>
              <Button variant="outline" size="sm">
                <Settings className="w-4 h-4 mr-2" />
                Settings
              </Button>
            </div>
          </div>
        </div>

        {/* System Overview Cards */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
          <Card className="bg-slate-800 border-slate-700">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-slate-300">Latency</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-green-400">8.3ms</div>
              <p className="text-xs text-slate-400">End-to-end execution</p>
            </CardContent>
          </Card>

          <Card className="bg-slate-800 border-slate-700">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-slate-300">Prediction Accuracy</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-blue-400">64.2%</div>
              <p className="text-xs text-slate-400">1-5 sec directional</p>
            </CardContent>
          </Card>

          <Card className="bg-slate-800 border-slate-700">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-slate-300">Daily P&L</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-green-400">+₹12,450</div>
              <p className="text-xs text-slate-400">Risk-adjusted return</p>
            </CardContent>
          </Card>

          <Card className="bg-slate-800 border-slate-700">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-slate-300">Active Signals</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-yellow-400">23</div>
              <p className="text-xs text-slate-400">LOB + IV combined</p>
            </CardContent>
          </Card>
        </div>

        {/* Main Interface */}
        <Tabs defaultValue="dashboard" className="space-y-6">
          <TabsList className="grid w-full grid-cols-6 bg-slate-800">
            <TabsTrigger value="dashboard" className="data-[state=active]:bg-slate-700">
              <BarChart3 className="w-4 h-4 mr-2" />
              Dashboard
            </TabsTrigger>
            <TabsTrigger value="orderbook" className="data-[state=active]:bg-slate-700">
              <Activity className="w-4 h-4 mr-2" />
              Order Book
            </TabsTrigger>
            <TabsTrigger value="signals" className="data-[state=active]:bg-slate-700">
              <Zap className="w-4 h-4 mr-2" />
              Signals
            </TabsTrigger>
            <TabsTrigger value="iv-surface" className="data-[state=active]:bg-slate-700">
              <TrendingUp className="w-4 h-4 mr-2" />
              IV Surface
            </TabsTrigger>
            <TabsTrigger value="backtest" className="data-[state=active]:bg-slate-700">
              <Target className="w-4 h-4 mr-2" />
              Backtest
            </TabsTrigger>
            <TabsTrigger value="performance" className="data-[state=active]:bg-slate-700">
              <Activity className="w-4 h-4 mr-2" />
              Performance
            </TabsTrigger>
          </TabsList>

          <TabsContent value="dashboard">
            <TradingDashboard />
          </TabsContent>

          <TabsContent value="orderbook">
            <OrderBookView />
          </TabsContent>

          <TabsContent value="signals">
            <SignalGenerator />
          </TabsContent>

          <TabsContent value="iv-surface">
            <IVSurfaceView />
          </TabsContent>

          <TabsContent value="backtest">
            <BacktestInterface />
          </TabsContent>

          <TabsContent value="performance">
            <PerformanceMonitor />
          </TabsContent>
        </Tabs>
      </div>
    </div>
  )
}
