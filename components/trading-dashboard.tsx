"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  AreaChart,
  Area,
  BarChart,
  Bar,
} from "recharts"

interface MarketData {
  time: string
  price: number
  volume: number
  signal: number
  pnl: number
}

export function TradingDashboard() {
  const [marketData, setMarketData] = useState<MarketData[]>([])
  const [currentPrice, setCurrentPrice] = useState(18450.75)

  useEffect(() => {
    // Generate initial data
    const initialData: MarketData[] = []
    const basePrice = 18450
    let cumulativePnL = 0

    for (let i = 0; i < 100; i++) {
      const time = new Date(Date.now() - (100 - i) * 1000).toLocaleTimeString()
      const price = basePrice + Math.sin(i * 0.1) * 50 + Math.random() * 20 - 10
      const volume = Math.floor(Math.random() * 1000) + 500
      const signal = Math.sin(i * 0.15) * 0.8 + Math.random() * 0.4 - 0.2
      const pnl = Math.random() * 200 - 100
      cumulativePnL += pnl

      initialData.push({
        time,
        price,
        volume,
        signal,
        pnl: cumulativePnL,
      })
    }

    setMarketData(initialData)

    // Simulate real-time updates
    const interval = setInterval(() => {
      const now = new Date().toLocaleTimeString()
      const lastPrice = marketData[marketData.length - 1]?.price || basePrice
      const newPrice = lastPrice + (Math.random() - 0.5) * 5
      const volume = Math.floor(Math.random() * 1000) + 500
      const signal = Math.sin(Date.now() * 0.001) * 0.8 + Math.random() * 0.4 - 0.2
      const pnl = Math.random() * 200 - 100

      setCurrentPrice(newPrice)

      setMarketData((prev) => {
        const newData = [
          ...prev.slice(-99),
          {
            time: now,
            price: newPrice,
            volume,
            signal,
            pnl: (prev[prev.length - 1]?.pnl || 0) + pnl,
          },
        ]
        return newData
      })
    }, 1000)

    return () => clearInterval(interval)
  }, [])

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      {/* Price Chart */}
      <Card className="bg-slate-800 border-slate-700 lg:col-span-2">
        <CardHeader>
          <CardTitle className="text-white">NIFTY50 Real-Time Price</CardTitle>
          <CardDescription className="text-slate-400">
            Current: ₹{currentPrice.toFixed(2)} | Live feed with ML predictions
          </CardDescription>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={marketData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="time" stroke="#9CA3AF" tick={{ fontSize: 12 }} />
              <YAxis stroke="#9CA3AF" tick={{ fontSize: 12 }} domain={["dataMin - 10", "dataMax + 10"]} />
              <Tooltip
                contentStyle={{
                  backgroundColor: "#1F2937",
                  border: "1px solid #374151",
                  borderRadius: "8px",
                }}
              />
              <Line type="monotone" dataKey="price" stroke="#3B82F6" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      {/* Signal Strength */}
      <Card className="bg-slate-800 border-slate-700">
        <CardHeader>
          <CardTitle className="text-white">ML Signal Strength</CardTitle>
          <CardDescription className="text-slate-400">Combined LOB + IV predictions</CardDescription>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={200}>
            <AreaChart data={marketData.slice(-20)}>
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
              <Area type="monotone" dataKey="signal" stroke="#10B981" fill="#10B981" fillOpacity={0.3} />
            </AreaChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      {/* P&L Chart */}
      <Card className="bg-slate-800 border-slate-700">
        <CardHeader>
          <CardTitle className="text-white">Cumulative P&L</CardTitle>
          <CardDescription className="text-slate-400">Real-time profit & loss tracking</CardDescription>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={marketData.slice(-20)}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="time" stroke="#9CA3AF" tick={{ fontSize: 10 }} />
              <YAxis stroke="#9CA3AF" tick={{ fontSize: 12 }} />
              <Tooltip
                contentStyle={{
                  backgroundColor: "#1F2937",
                  border: "1px solid #374151",
                  borderRadius: "8px",
                }}
              />
              <Line type="monotone" dataKey="pnl" stroke="#F59E0B" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      {/* Volume Analysis */}
      <Card className="bg-slate-800 border-slate-700 lg:col-span-2">
        <CardHeader>
          <CardTitle className="text-white">Volume Analysis</CardTitle>
          <CardDescription className="text-slate-400">Order flow and market microstructure</CardDescription>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={marketData.slice(-20)}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="time" stroke="#9CA3AF" tick={{ fontSize: 10 }} />
              <YAxis stroke="#9CA3AF" tick={{ fontSize: 12 }} />
              <Tooltip
                contentStyle={{
                  backgroundColor: "#1F2937",
                  border: "1px solid #374151",
                  borderRadius: "8px",
                }}
              />
              <Bar dataKey="volume" fill="#8B5CF6" opacity={0.8} />
            </BarChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>
    </div>
  )
}
