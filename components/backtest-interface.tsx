"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Progress } from "@/components/ui/progress"
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar } from "recharts"
import { Play, Download, Settings, TrendingUp } from "lucide-react"

interface BacktestResult {
  totalReturn: number
  sharpeRatio: number
  maxDrawdown: number
  winRate: number
  avgTrade: number
  totalTrades: number
  volatility: number
}

interface BacktestConfig {
  startDate: string
  endDate: string
  initialCapital: number
  strategy: string
  riskModel: string
  maxPosition: number
}

export function BacktestInterface() {
  const [config, setConfig] = useState<BacktestConfig>({
    startDate: "2024-01-01",
    endDate: "2024-12-31",
    initialCapital: 1000000,
    strategy: "combined",
    riskModel: "var",
    maxPosition: 0.1,
  })

  const [isRunning, setIsRunning] = useState(false)
  const [progress, setProgress] = useState(0)
  const [results, setResults] = useState<BacktestResult | null>(null)

  const [equityCurve, setEquityCurve] = useState<
    Array<{
      date: string
      equity: number
      benchmark: number
      drawdown: number
    }>
  >([])

  const [tradeAnalysis, setTradeAnalysis] = useState<
    Array<{
      month: string
      trades: number
      pnl: number
      winRate: number
    }>
  >([])

  const runBacktest = async () => {
    setIsRunning(true)
    setProgress(0)

    // Simulate backtest execution
    const interval = setInterval(() => {
      setProgress((prev) => {
        if (prev >= 100) {
          clearInterval(interval)
          setIsRunning(false)

          // Generate mock results
          const mockResults: BacktestResult = {
            totalReturn: 15.7 + Math.random() * 10,
            sharpeRatio: 1.85 + Math.random() * 0.5,
            maxDrawdown: 3.2 + Math.random() * 2,
            winRate: 62.5 + Math.random() * 10,
            avgTrade: 125 + Math.random() * 50,
            totalTrades: Math.floor(Math.random() * 500) + 1000,
            volatility: 12.3 + Math.random() * 3,
          }
          setResults(mockResults)

          // Generate equity curve
          const equity = []
          const benchmark = []
          let equityValue = config.initialCapital
          let benchmarkValue = config.initialCapital

          for (let i = 0; i < 252; i++) {
            const date = new Date(2024, 0, i + 1).toISOString().split("T")[0]
            const strategyReturn = (Math.random() - 0.48) * 0.02 // Slight positive bias
            const benchmarkReturn = (Math.random() - 0.5) * 0.015

            equityValue *= 1 + strategyReturn
            benchmarkValue *= 1 + benchmarkReturn

            const drawdown = Math.max(
              0,
              ((Math.max(...equity.map((e) => e.equity), equityValue) - equityValue) /
                Math.max(...equity.map((e) => e.equity), equityValue)) *
                100,
            )

            equity.push({
              date,
              equity: equityValue,
              benchmark: benchmarkValue,
              drawdown,
            })
          }
          setEquityCurve(equity)

          // Generate trade analysis
          const trades = []
          for (let i = 0; i < 12; i++) {
            const month = new Date(2024, i, 1).toLocaleString("default", { month: "short" })
            trades.push({
              month,
              trades: Math.floor(Math.random() * 100) + 50,
              pnl: (Math.random() - 0.4) * 5000,
              winRate: 50 + Math.random() * 30,
            })
          }
          setTradeAnalysis(trades)

          return 100
        }
        return prev + 2
      })
    }, 100)
  }

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
      {/* Backtest Configuration */}
      <Card className="bg-slate-800 border-slate-700">
        <CardHeader>
          <CardTitle className="text-white flex items-center gap-2">
            <Settings className="w-5 h-5" />
            Backtest Configuration
          </CardTitle>
          <CardDescription className="text-slate-400">Configure strategy parameters</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-2 gap-4">
            <div>
              <Label htmlFor="startDate" className="text-slate-300">
                Start Date
              </Label>
              <Input
                id="startDate"
                type="date"
                value={config.startDate}
                onChange={(e) => setConfig({ ...config, startDate: e.target.value })}
                className="bg-slate-700 border-slate-600"
              />
            </div>
            <div>
              <Label htmlFor="endDate" className="text-slate-300">
                End Date
              </Label>
              <Input
                id="endDate"
                type="date"
                value={config.endDate}
                onChange={(e) => setConfig({ ...config, endDate: e.target.value })}
                className="bg-slate-700 border-slate-600"
              />
            </div>
          </div>

          <div>
            <Label htmlFor="capital" className="text-slate-300">
              Initial Capital
            </Label>
            <Input
              id="capital"
              type="number"
              value={config.initialCapital}
              onChange={(e) => setConfig({ ...config, initialCapital: Number(e.target.value) })}
              className="bg-slate-700 border-slate-600"
            />
          </div>

          <div>
            <Label className="text-slate-300">Strategy</Label>
            <Select value={config.strategy} onValueChange={(value) => setConfig({ ...config, strategy: value })}>
              <SelectTrigger className="bg-slate-700 border-slate-600">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="lob">LOB Only</SelectItem>
                <SelectItem value="iv">IV Only</SelectItem>
                <SelectItem value="combined">Combined LOB + IV</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div>
            <Label className="text-slate-300">Risk Model</Label>
            <Select value={config.riskModel} onValueChange={(value) => setConfig({ ...config, riskModel: value })}>
              <SelectTrigger className="bg-slate-700 border-slate-600">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="var">Value at Risk</SelectItem>
                <SelectItem value="kelly">Kelly Criterion</SelectItem>
                <SelectItem value="fixed">Fixed Size</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div>
            <Label htmlFor="maxPosition" className="text-slate-300">
              Max Position Size
            </Label>
            <Input
              id="maxPosition"
              type="number"
              step="0.01"
              value={config.maxPosition}
              onChange={(e) => setConfig({ ...config, maxPosition: Number(e.target.value) })}
              className="bg-slate-700 border-slate-600"
            />
          </div>

          <Button onClick={runBacktest} disabled={isRunning} className="w-full">
            <Play className="w-4 h-4 mr-2" />
            {isRunning ? "Running..." : "Run Backtest"}
          </Button>

          {isRunning && (
            <div>
              <div className="flex justify-between mb-2">
                <span className="text-sm text-slate-400">Progress</span>
                <span className="text-sm text-white">{progress}%</span>
              </div>
              <Progress value={progress} className="h-2" />
            </div>
          )}
        </CardContent>
      </Card>

      {/* Results Summary */}
      <Card className="bg-slate-800 border-slate-700">
        <CardHeader>
          <CardTitle className="text-white flex items-center gap-2">
            <TrendingUp className="w-5 h-5" />
            Results Summary
          </CardTitle>
          <CardDescription className="text-slate-400">Key performance metrics</CardDescription>
        </CardHeader>
        <CardContent>
          {results ? (
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <div className="text-2xl font-bold text-green-400">{results.totalReturn.toFixed(1)}%</div>
                  <div className="text-xs text-slate-400">Total Return</div>
                </div>
                <div>
                  <div className="text-2xl font-bold text-blue-400">{results.sharpeRatio.toFixed(2)}</div>
                  <div className="text-xs text-slate-400">Sharpe Ratio</div>
                </div>
              </div>

              <div className="space-y-3">
                <div className="flex justify-between">
                  <span className="text-sm text-slate-400">Max Drawdown</span>
                  <span className="text-sm text-white">{results.maxDrawdown.toFixed(1)}%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-slate-400">Win Rate</span>
                  <span className="text-sm text-white">{results.winRate.toFixed(1)}%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-slate-400">Avg Trade</span>
                  <span className="text-sm text-white">₹{results.avgTrade.toFixed(0)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-slate-400">Total Trades</span>
                  <span className="text-sm text-white">{results.totalTrades.toLocaleString()}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-slate-400">Volatility</span>
                  <span className="text-sm text-white">{results.volatility.toFixed(1)}%</span>
                </div>
              </div>

              <Button variant="outline" size="sm" className="w-full bg-transparent">
                <Download className="w-4 h-4 mr-2" />
                Export Report
              </Button>
            </div>
          ) : (
            <div className="text-center py-8">
              <p className="text-slate-400">Run a backtest to see results</p>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Trade Analysis */}
      <Card className="bg-slate-800 border-slate-700">
        <CardHeader>
          <CardTitle className="text-white">Monthly Analysis</CardTitle>
          <CardDescription className="text-slate-400">Trade distribution and performance</CardDescription>
        </CardHeader>
        <CardContent>
          {tradeAnalysis.length > 0 ? (
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={tradeAnalysis}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis dataKey="month" stroke="#9CA3AF" tick={{ fontSize: 10 }} />
                <YAxis stroke="#9CA3AF" tick={{ fontSize: 12 }} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "#1F2937",
                    border: "1px solid #374151",
                    borderRadius: "8px",
                  }}
                />
                <Bar dataKey="pnl" fill="#3B82F6" opacity={0.8} />
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <div className="text-center py-8">
              <p className="text-slate-400">No trade data available</p>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Equity Curve */}
      <Card className="bg-slate-800 border-slate-700 lg:col-span-3">
        <CardHeader>
          <CardTitle className="text-white">Equity Curve</CardTitle>
          <CardDescription className="text-slate-400">Strategy performance vs benchmark over time</CardDescription>
        </CardHeader>
        <CardContent>
          {equityCurve.length > 0 ? (
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={equityCurve}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis
                  dataKey="date"
                  stroke="#9CA3AF"
                  tick={{ fontSize: 10 }}
                  tickFormatter={(value) =>
                    new Date(value).toLocaleDateString("en-US", { month: "short", day: "numeric" })
                  }
                />
                <YAxis
                  stroke="#9CA3AF"
                  tick={{ fontSize: 12 }}
                  tickFormatter={(value) => `₹${(value / 1000).toFixed(0)}K`}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "#1F2937",
                    border: "1px solid #374151",
                    borderRadius: "8px",
                  }}
                  labelFormatter={(value) => new Date(value).toLocaleDateString()}
                  formatter={(value: number, name: string) => [
                    `₹${value.toLocaleString()}`,
                    name === "equity" ? "Strategy" : "Benchmark",
                  ]}
                />
                <Line type="monotone" dataKey="equity" stroke="#10B981" strokeWidth={2} dot={false} name="Strategy" />
                <Line
                  type="monotone"
                  dataKey="benchmark"
                  stroke="#6B7280"
                  strokeWidth={1}
                  strokeDasharray="5 5"
                  dot={false}
                  name="Benchmark"
                />
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <div className="text-center py-16">
              <p className="text-slate-400">Run a backtest to see equity curve</p>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
