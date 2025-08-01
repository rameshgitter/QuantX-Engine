"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ScatterChart,
  Scatter,
} from "recharts"
import { Play, Pause, RefreshCw, Zap } from "lucide-react"

interface Signal {
  id: string
  name: string
  type: "LOB" | "IV" | "COMBINED"
  strength: number
  confidence: number
  direction: "BUY" | "SELL" | "NEUTRAL"
  timestamp: Date
  features: Record<string, number>
}

interface ModelPerformance {
  accuracy: number
  precision: number
  recall: number
  sharpe: number
  maxDrawdown: number
}

export function SignalGenerator() {
  const [signals, setSignals] = useState<Signal[]>([])
  const [isRunning, setIsRunning] = useState(true)
  const [modelPerformance, setModelPerformance] = useState<ModelPerformance>({
    accuracy: 64.2,
    precision: 67.8,
    recall: 61.5,
    sharpe: 1.85,
    maxDrawdown: 3.2,
  })

  const [signalHistory, setSignalHistory] = useState<
    Array<{
      time: string
      lobSignal: number
      ivSignal: number
      combinedSignal: number
      actualReturn: number
    }>
  >([])

  useEffect(() => {
    const generateSignal = () => {
      // LOB-based signal
      const lobFeatures = {
        imbalance: Math.random() * 2 - 1,
        spread: Math.random() * 0.5,
        volumeRatio: Math.random() * 3,
        priceImpact: Math.random() * 0.1,
        microPrice: Math.random() * 0.02 - 0.01,
      }

      const lobStrength = Math.tanh(
        lobFeatures.imbalance * 0.3 + lobFeatures.volumeRatio * 0.2 - lobFeatures.spread * 0.5,
      )

      // IV-based signal
      const ivFeatures = {
        ivSkew: Math.random() * 0.1 - 0.05,
        termStructure: Math.random() * 0.2 - 0.1,
        volOfVol: Math.random() * 0.3,
        putCallRatio: Math.random() * 2,
        gamma: Math.random() * 0.001,
      }

      const ivStrength = Math.tanh(
        ivFeatures.ivSkew * 2 + ivFeatures.termStructure * 1.5 + ivFeatures.putCallRatio * 0.3,
      )

      // Combined signal
      const combinedStrength = (lobStrength * 0.6 + ivStrength * 0.4) * (0.8 + Math.random() * 0.4)

      const newSignals: Signal[] = [
        {
          id: `lob_${Date.now()}`,
          name: "LOB Predictor",
          type: "LOB",
          strength: Math.abs(lobStrength),
          confidence: 0.6 + Math.random() * 0.3,
          direction: lobStrength > 0.1 ? "BUY" : lobStrength < -0.1 ? "SELL" : "NEUTRAL",
          timestamp: new Date(),
          features: lobFeatures,
        },
        {
          id: `iv_${Date.now()}`,
          name: "IV Surface",
          type: "IV",
          strength: Math.abs(ivStrength),
          confidence: 0.5 + Math.random() * 0.4,
          direction: ivStrength > 0.1 ? "BUY" : ivStrength < -0.1 ? "SELL" : "NEUTRAL",
          timestamp: new Date(),
          features: ivFeatures,
        },
        {
          id: `combined_${Date.now()}`,
          name: "Ensemble Model",
          type: "COMBINED",
          strength: Math.abs(combinedStrength),
          confidence: 0.7 + Math.random() * 0.2,
          direction: combinedStrength > 0.15 ? "BUY" : combinedStrength < -0.15 ? "SELL" : "NEUTRAL",
          timestamp: new Date(),
          features: { ...lobFeatures, ...ivFeatures },
        },
      ]

      setSignals(newSignals)

      // Update signal history
      const time = new Date().toLocaleTimeString()
      const actualReturn = (Math.random() - 0.5) * 0.02 // Simulated actual return

      setSignalHistory((prev) => [
        ...prev.slice(-19),
        {
          time,
          lobSignal: lobStrength,
          ivSignal: ivStrength,
          combinedSignal: combinedStrength,
          actualReturn,
        },
      ])
    }

    let interval: NodeJS.Timeout
    if (isRunning) {
      generateSignal()
      interval = setInterval(generateSignal, 2000)
    }

    return () => {
      if (interval) clearInterval(interval)
    }
  }, [isRunning])

  const getSignalColor = (direction: string) => {
    switch (direction) {
      case "BUY":
        return "bg-green-600"
      case "SELL":
        return "bg-red-600"
      default:
        return "bg-gray-600"
    }
  }

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
      {/* Signal Controls */}
      <Card className="bg-slate-800 border-slate-700">
        <CardHeader>
          <CardTitle className="text-white flex items-center gap-2">
            <Zap className="w-5 h-5" />
            Signal Engine
          </CardTitle>
          <CardDescription className="text-slate-400">ML-powered prediction system</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex gap-2">
            <Button onClick={() => setIsRunning(!isRunning)} variant={isRunning ? "destructive" : "default"} size="sm">
              {isRunning ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
              {isRunning ? "Pause" : "Start"}
            </Button>
            <Button variant="outline" size="sm">
              <RefreshCw className="w-4 h-4" />
              Reset
            </Button>
          </div>

          <div className="space-y-3">
            <div>
              <div className="flex justify-between mb-1">
                <span className="text-sm text-slate-400">Model Accuracy</span>
                <span className="text-sm text-white">{modelPerformance.accuracy}%</span>
              </div>
              <Progress value={modelPerformance.accuracy} className="h-2" />
            </div>

            <div>
              <div className="flex justify-between mb-1">
                <span className="text-sm text-slate-400">Precision</span>
                <span className="text-sm text-white">{modelPerformance.precision}%</span>
              </div>
              <Progress value={modelPerformance.precision} className="h-2" />
            </div>

            <div>
              <div className="flex justify-between mb-1">
                <span className="text-sm text-slate-400">Sharpe Ratio</span>
                <span className="text-sm text-white">{modelPerformance.sharpe}</span>
              </div>
            </div>

            <div>
              <div className="flex justify-between mb-1">
                <span className="text-sm text-slate-400">Max Drawdown</span>
                <span className="text-sm text-white">{modelPerformance.maxDrawdown}%</span>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Active Signals */}
      <Card className="bg-slate-800 border-slate-700">
        <CardHeader>
          <CardTitle className="text-white">Active Signals</CardTitle>
          <CardDescription className="text-slate-400">Real-time predictions from all models</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {signals.map((signal) => (
            <div key={signal.id} className="border border-slate-600 rounded-lg p-3">
              <div className="flex justify-between items-center mb-2">
                <span className="text-sm font-medium text-white">{signal.name}</span>
                <Badge className={getSignalColor(signal.direction)}>{signal.direction}</Badge>
              </div>

              <div className="space-y-2">
                <div className="flex justify-between text-xs">
                  <span className="text-slate-400">Strength</span>
                  <span className="text-white">{(signal.strength * 100).toFixed(1)}%</span>
                </div>
                <Progress value={signal.strength * 100} className="h-1" />

                <div className="flex justify-between text-xs">
                  <span className="text-slate-400">Confidence</span>
                  <span className="text-white">{(signal.confidence * 100).toFixed(1)}%</span>
                </div>
                <Progress value={signal.confidence * 100} className="h-1" />
              </div>
            </div>
          ))}
        </CardContent>
      </Card>

      {/* Signal Performance */}
      <Card className="bg-slate-800 border-slate-700">
        <CardHeader>
          <CardTitle className="text-white">Signal Performance</CardTitle>
          <CardDescription className="text-slate-400">Historical accuracy and correlation</CardDescription>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={signalHistory}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="time" stroke="#9CA3AF" tick={{ fontSize: 10 }} />
              <YAxis stroke="#9CA3AF" tick={{ fontSize: 12 }} domain={[-1, 1]} />
              <Tooltip
                contentStyle={{
                  backgroundColor: "#1F2937",
                  border: "1px solid #374151",
                  borderRadius: "8px",
                }}
              />
              <Line
                type="monotone"
                dataKey="lobSignal"
                stroke="#3B82F6"
                strokeWidth={1}
                dot={false}
                name="LOB Signal"
              />
              <Line type="monotone" dataKey="ivSignal" stroke="#8B5CF6" strokeWidth={1} dot={false} name="IV Signal" />
              <Line
                type="monotone"
                dataKey="combinedSignal"
                stroke="#10B981"
                strokeWidth={2}
                dot={false}
                name="Combined"
              />
            </LineChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      {/* Signal vs Reality Scatter */}
      <Card className="bg-slate-800 border-slate-700 lg:col-span-3">
        <CardHeader>
          <CardTitle className="text-white">Signal vs Actual Returns</CardTitle>
          <CardDescription className="text-slate-400">
            Correlation analysis between predictions and market movements
          </CardDescription>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={300}>
            <ScatterChart data={signalHistory}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis type="number" dataKey="combinedSignal" name="Signal Strength" stroke="#9CA3AF" domain={[-1, 1]} />
              <YAxis
                type="number"
                dataKey="actualReturn"
                name="Actual Return"
                stroke="#9CA3AF"
                domain={[-0.02, 0.02]}
              />
              <Tooltip
                cursor={{ strokeDasharray: "3 3" }}
                contentStyle={{
                  backgroundColor: "#1F2937",
                  border: "1px solid #374151",
                  borderRadius: "8px",
                }}
              />
              <Scatter dataKey="actualReturn" fill="#10B981" fillOpacity={0.7} />
            </ScatterChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>
    </div>
  )
}
