"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area } from "recharts"
import { Activity, Cpu, HardDrive, Wifi, Zap, AlertCircle } from "lucide-react"

interface SystemMetrics {
  latency: number
  throughput: number
  cpuUsage: number
  memoryUsage: number
  networkLatency: number
  orderFillRate: number
}

interface LatencyData {
  time: string
  p50: number
  p95: number
  p99: number
}

export function PerformanceMonitor() {
  const [metrics, setMetrics] = useState<SystemMetrics>({
    latency: 8.3,
    throughput: 1250,
    cpuUsage: 45,
    memoryUsage: 62,
    networkLatency: 2.1,
    orderFillRate: 98.7,
  })

  const [latencyHistory, setLatencyHistory] = useState<LatencyData[]>([])
  const [alerts, setAlerts] = useState<string[]>([])

  useEffect(() => {
    const updateMetrics = () => {
      const newMetrics: SystemMetrics = {
        latency: 8 + Math.random() * 4,
        throughput: 1200 + Math.random() * 200,
        cpuUsage: 40 + Math.random() * 30,
        memoryUsage: 55 + Math.random() * 20,
        networkLatency: 1.5 + Math.random() * 2,
        orderFillRate: 97 + Math.random() * 3,
      }

      setMetrics(newMetrics)

      // Update latency history
      const time = new Date().toLocaleTimeString()
      const newLatencyData: LatencyData = {
        time,
        p50: newMetrics.latency * 0.7,
        p95: newMetrics.latency * 1.2,
        p99: newMetrics.latency * 1.8,
      }

      setLatencyHistory((prev) => [...prev.slice(-19), newLatencyData])

      // Generate alerts
      const newAlerts = []
      if (newMetrics.latency > 12) {
        newAlerts.push("High latency detected")
      }
      if (newMetrics.cpuUsage > 80) {
        newAlerts.push("CPU usage critical")
      }
      if (newMetrics.memoryUsage > 85) {
        newAlerts.push("Memory usage high")
      }
      if (newMetrics.orderFillRate < 95) {
        newAlerts.push("Low order fill rate")
      }
      setAlerts(newAlerts)
    }

    updateMetrics()
    const interval = setInterval(updateMetrics, 1000)

    return () => clearInterval(interval)
  }, [])

  const getLatencyColor = (latency: number) => {
    if (latency > 12) return "text-red-400"
    if (latency > 10) return "text-yellow-400"
    return "text-green-400"
  }

  const getUsageColor = (usage: number) => {
    if (usage > 80) return "bg-red-600"
    if (usage > 60) return "bg-yellow-600"
    return "bg-green-600"
  }

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
      {/* System Overview */}
      <Card className="bg-slate-800 border-slate-700">
        <CardHeader>
          <CardTitle className="text-white flex items-center gap-2">
            <Activity className="w-5 h-5" />
            System Health
          </CardTitle>
          <CardDescription className="text-slate-400">Real-time performance metrics</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Zap className="w-4 h-4 text-blue-400" />
              <span className="text-sm text-slate-400">Latency</span>
            </div>
            <span className={`text-lg font-mono ${getLatencyColor(metrics.latency)}`}>
              {metrics.latency.toFixed(1)}ms
            </span>
          </div>

          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Activity className="w-4 h-4 text-green-400" />
              <span className="text-sm text-slate-400">Throughput</span>
            </div>
            <span className="text-lg font-mono text-white">{metrics.throughput.toLocaleString()}/s</span>
          </div>

          <div>
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <Cpu className="w-4 h-4 text-purple-400" />
                <span className="text-sm text-slate-400">CPU Usage</span>
              </div>
              <span className="text-sm text-white">{metrics.cpuUsage.toFixed(0)}%</span>
            </div>
            <Progress value={metrics.cpuUsage} className="h-2" />
          </div>

          <div>
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <HardDrive className="w-4 h-4 text-orange-400" />
                <span className="text-sm text-slate-400">Memory Usage</span>
              </div>
              <span className="text-sm text-white">{metrics.memoryUsage.toFixed(0)}%</span>
            </div>
            <Progress value={metrics.memoryUsage} className="h-2" />
          </div>

          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Wifi className="w-4 h-4 text-cyan-400" />
              <span className="text-sm text-slate-400">Network Latency</span>
            </div>
            <span className="text-lg font-mono text-white">{metrics.networkLatency.toFixed(1)}ms</span>
          </div>

          <div className="flex items-center justify-between">
            <span className="text-sm text-slate-400">Order Fill Rate</span>
            <Badge className={getUsageColor(metrics.orderFillRate)}>{metrics.orderFillRate.toFixed(1)}%</Badge>
          </div>
        </CardContent>
      </Card>

      {/* Alerts */}
      <Card className="bg-slate-800 border-slate-700">
        <CardHeader>
          <CardTitle className="text-white flex items-center gap-2">
            <AlertCircle className="w-5 h-5" />
            System Alerts
          </CardTitle>
          <CardDescription className="text-slate-400">Performance warnings and issues</CardDescription>
        </CardHeader>
        <CardContent>
          {alerts.length > 0 ? (
            <div className="space-y-2">
              {alerts.map((alert, index) => (
                <div key={index} className="p-3 bg-red-900/20 border border-red-600/30 rounded-lg">
                  <div className="flex items-center gap-2">
                    <AlertCircle className="w-4 h-4 text-red-400" />
                    <span className="text-sm text-red-400">{alert}</span>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-8">
              <div className="w-12 h-12 bg-green-900/20 rounded-full flex items-center justify-center mx-auto mb-3">
                <Activity className="w-6 h-6 text-green-400" />
              </div>
              <p className="text-sm text-slate-400">All systems operational</p>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Component Status */}
      <Card className="bg-slate-800 border-slate-700">
        <CardHeader>
          <CardTitle className="text-white">Component Status</CardTitle>
          <CardDescription className="text-slate-400">Individual service health</CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          {[
            { name: "Feed Handler", status: "healthy", uptime: "99.9%" },
            { name: "Signal Generator", status: "healthy", uptime: "99.7%" },
            { name: "Order Manager", status: "healthy", uptime: "99.8%" },
            { name: "Risk Engine", status: "healthy", uptime: "100%" },
            { name: "IV Service", status: "warning", uptime: "98.2%" },
            { name: "Database", status: "healthy", uptime: "99.9%" },
          ].map((component, index) => (
            <div key={index} className="flex items-center justify-between p-2 border border-slate-600 rounded">
              <div className="flex items-center gap-3">
                <div
                  className={`w-2 h-2 rounded-full ${
                    component.status === "healthy"
                      ? "bg-green-400"
                      : component.status === "warning"
                        ? "bg-yellow-400"
                        : "bg-red-400"
                  }`}
                />
                <span className="text-sm text-white">{component.name}</span>
              </div>
              <span className="text-xs text-slate-400">{component.uptime}</span>
            </div>
          ))}
        </CardContent>
      </Card>

      {/* Latency Distribution */}
      <Card className="bg-slate-800 border-slate-700 lg:col-span-2">
        <CardHeader>
          <CardTitle className="text-white">Latency Distribution</CardTitle>
          <CardDescription className="text-slate-400">Real-time latency percentiles over time</CardDescription>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={300}>
            <AreaChart data={latencyHistory}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="time" stroke="#9CA3AF" tick={{ fontSize: 10 }} />
              <YAxis
                stroke="#9CA3AF"
                tick={{ fontSize: 12 }}
                label={{ value: "Latency (ms)", angle: -90, position: "insideLeft" }}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: "#1F2937",
                  border: "1px solid #374151",
                  borderRadius: "8px",
                }}
              />
              <Area
                type="monotone"
                dataKey="p99"
                stackId="1"
                stroke="#EF4444"
                fill="#EF4444"
                fillOpacity={0.3}
                name="P99"
              />
              <Area
                type="monotone"
                dataKey="p95"
                stackId="1"
                stroke="#F59E0B"
                fill="#F59E0B"
                fillOpacity={0.3}
                name="P95"
              />
              <Area
                type="monotone"
                dataKey="p50"
                stackId="1"
                stroke="#10B981"
                fill="#10B981"
                fillOpacity={0.3}
                name="P50"
              />
            </AreaChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      {/* Resource Usage */}
      <Card className="bg-slate-800 border-slate-700">
        <CardHeader>
          <CardTitle className="text-white">Resource Usage</CardTitle>
          <CardDescription className="text-slate-400">System resource consumption</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div>
            <div className="flex justify-between mb-2">
              <span className="text-sm text-slate-400">CPU Cores</span>
              <span className="text-sm text-white">16/32</span>
            </div>
            <Progress value={50} className="h-2" />
          </div>

          <div>
            <div className="flex justify-between mb-2">
              <span className="text-sm text-slate-400">Memory</span>
              <span className="text-sm text-white">24.8GB/64GB</span>
            </div>
            <Progress value={38.75} className="h-2" />
          </div>

          <div>
            <div className="flex justify-between mb-2">
              <span className="text-sm text-slate-400">Disk I/O</span>
              <span className="text-sm text-white">2.1GB/s</span>
            </div>
            <Progress value={65} className="h-2" />
          </div>

          <div>
            <div className="flex justify-between mb-2">
              <span className="text-sm text-slate-400">Network</span>
              <span className="text-sm text-white">850MB/s</span>
            </div>
            <Progress value={42.5} className="h-2" />
          </div>

          <div className="pt-2 border-t border-slate-600">
            <div className="text-xs text-slate-400 space-y-1">
              <div>Uptime: 15d 7h 23m</div>
              <div>Last restart: 2024-12-17 09:15</div>
              <div>Version: v2.1.3</div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
