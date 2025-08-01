"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts"
import { TrendingUp, AlertTriangle } from "lucide-react"

interface IVPoint {
  strike: number
  expiry: string
  iv: number
  delta: number
  gamma: number
  theta: number
  vega: number
}

interface IVSurfaceData {
  strike: number
  "1W": number
  "2W": number
  "1M": number
  "2M": number
  "3M": number
}

export function IVSurfaceView() {
  const [ivData, setIvData] = useState<IVSurfaceData[]>([])
  const [selectedExpiry, setSelectedExpiry] = useState("1M")
  const [ivSkew, setIvSkew] = useState(0)
  const [termStructure, setTermStructure] = useState<Array<{ expiry: string; atm_iv: number }>>([])
  const [alerts, setAlerts] = useState<string[]>([])

  useEffect(() => {
    const generateIVSurface = () => {
      const baseIV = 0.18 + Math.sin(Date.now() * 0.0001) * 0.02
      const currentPrice = 18450

      const strikes = []
      for (let i = -10; i <= 10; i++) {
        strikes.push(currentPrice + i * 50)
      }

      const surfaceData: IVSurfaceData[] = strikes.map((strike) => {
        const moneyness = Math.log(strike / currentPrice)
        const skewEffect = -0.1 * moneyness + 0.05 * moneyness * moneyness

        return {
          strike,
          "1W": Math.max(0.05, baseIV + skewEffect + Math.random() * 0.02),
          "2W": Math.max(0.05, baseIV + skewEffect * 0.8 + Math.random() * 0.02),
          "1M": Math.max(0.05, baseIV + skewEffect * 0.6 + Math.random() * 0.02),
          "2M": Math.max(0.05, baseIV + skewEffect * 0.4 + Math.random() * 0.02),
          "3M": Math.max(0.05, baseIV + skewEffect * 0.3 + Math.random() * 0.02),
        }
      })

      setIvData(surfaceData)

      // Calculate IV skew (25-delta put vs call)
      const atmIndex = Math.floor(surfaceData.length / 2)
      const putSkew = (surfaceData[atmIndex - 2]?.[selectedExpiry as keyof IVSurfaceData] as number) || 0
      const callSkew = (surfaceData[atmIndex + 2]?.[selectedExpiry as keyof IVSurfaceData] as number) || 0
      setIvSkew((putSkew - callSkew) * 100)

      // Generate term structure
      const termData = [
        { expiry: "1W", atm_iv: surfaceData[atmIndex]["1W"] },
        { expiry: "2W", atm_iv: surfaceData[atmIndex]["2W"] },
        { expiry: "1M", atm_iv: surfaceData[atmIndex]["1M"] },
        { expiry: "2M", atm_iv: surfaceData[atmIndex]["2M"] },
        { expiry: "3M", atm_iv: surfaceData[atmIndex]["3M"] },
      ]
      setTermStructure(termData)

      // Generate alerts
      const newAlerts = []
      if (Math.abs(ivSkew) > 2) {
        newAlerts.push(`High IV skew detected: ${ivSkew.toFixed(1)}%`)
      }
      if (baseIV > 0.22) {
        newAlerts.push("Elevated volatility regime")
      }
      if (termData[0].atm_iv > termData[2].atm_iv) {
        newAlerts.push("Inverted term structure")
      }
      setAlerts(newAlerts)
    }

    generateIVSurface()
    const interval = setInterval(generateIVSurface, 5000)

    return () => clearInterval(interval)
  }, [selectedExpiry])

  const getSkewColor = (skew: number) => {
    if (Math.abs(skew) > 2) return "bg-red-600"
    if (Math.abs(skew) > 1) return "bg-yellow-600"
    return "bg-green-600"
  }

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
      {/* IV Surface Chart */}
      <Card className="bg-slate-800 border-slate-700 lg:col-span-2">
        <CardHeader>
          <CardTitle className="text-white flex items-center gap-2">
            <TrendingUp className="w-5 h-5" />
            Implied Volatility Surface
          </CardTitle>
          <CardDescription className="text-slate-400">
            Real-time option volatility across strikes and expiries
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="mb-4 flex gap-2">
            {["1W", "2W", "1M", "2M", "3M"].map((expiry) => (
              <Button
                key={expiry}
                variant={selectedExpiry === expiry ? "default" : "outline"}
                size="sm"
                onClick={() => setSelectedExpiry(expiry)}
              >
                {expiry}
              </Button>
            ))}
          </div>

          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={ivData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="strike" stroke="#9CA3AF" tick={{ fontSize: 12 }} tickFormatter={(value) => `${value}`} />
              <YAxis
                stroke="#9CA3AF"
                tick={{ fontSize: 12 }}
                tickFormatter={(value) => `${(value * 100).toFixed(0)}%`}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: "#1F2937",
                  border: "1px solid #374151",
                  borderRadius: "8px",
                }}
                formatter={(value: number) => [`${(value * 100).toFixed(1)}%`, "IV"]}
              />
              <Line
                type="monotone"
                dataKey={selectedExpiry}
                stroke="#3B82F6"
                strokeWidth={2}
                dot={{ fill: "#3B82F6", strokeWidth: 2, r: 3 }}
              />
            </LineChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      {/* IV Metrics */}
      <div className="space-y-6">
        <Card className="bg-slate-800 border-slate-700">
          <CardHeader>
            <CardTitle className="text-white">IV Metrics</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <div className="flex justify-between items-center mb-2">
                <span className="text-sm text-slate-400">IV Skew</span>
                <Badge className={getSkewColor(ivSkew)}>{ivSkew.toFixed(1)}%</Badge>
              </div>
              <p className="text-xs text-slate-500">25Δ Put - Call IV differential</p>
            </div>

            <div>
              <div className="flex justify-between items-center mb-2">
                <span className="text-sm text-slate-400">ATM IV</span>
                <span className="text-sm text-white">
                  {((termStructure.find((t) => t.expiry === selectedExpiry)?.atm_iv || 0) * 100).toFixed(1)}%
                </span>
              </div>
            </div>

            <div>
              <div className="flex justify-between items-center mb-2">
                <span className="text-sm text-slate-400">Vol of Vol</span>
                <span className="text-sm text-white">{(Math.random() * 0.5 + 0.3).toFixed(2)}</span>
              </div>
            </div>

            <div>
              <div className="flex justify-between items-center mb-2">
                <span className="text-sm text-slate-400">Put/Call Ratio</span>
                <span className="text-sm text-white">{(Math.random() * 0.5 + 0.8).toFixed(2)}</span>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-slate-800 border-slate-700">
          <CardHeader>
            <CardTitle className="text-white flex items-center gap-2">
              <AlertTriangle className="w-4 h-4" />
              IV Alerts
            </CardTitle>
          </CardHeader>
          <CardContent>
            {alerts.length > 0 ? (
              <div className="space-y-2">
                {alerts.map((alert, index) => (
                  <div
                    key={index}
                    className="p-2 bg-yellow-900/20 border border-yellow-600/30 rounded text-xs text-yellow-400"
                  >
                    {alert}
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-sm text-slate-400">No alerts</p>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Term Structure */}
      <Card className="bg-slate-800 border-slate-700 lg:col-span-3">
        <CardHeader>
          <CardTitle className="text-white">Volatility Term Structure</CardTitle>
          <CardDescription className="text-slate-400">ATM implied volatility across expiration dates</CardDescription>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={termStructure}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="expiry" stroke="#9CA3AF" tick={{ fontSize: 12 }} />
              <YAxis
                stroke="#9CA3AF"
                tick={{ fontSize: 12 }}
                tickFormatter={(value) => `${(value * 100).toFixed(0)}%`}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: "#1F2937",
                  border: "1px solid #374151",
                  borderRadius: "8px",
                }}
                formatter={(value: number) => [`${(value * 100).toFixed(1)}%`, "ATM IV"]}
              />
              <Line
                type="monotone"
                dataKey="atm_iv"
                stroke="#8B5CF6"
                strokeWidth={3}
                dot={{ fill: "#8B5CF6", strokeWidth: 2, r: 4 }}
              />
            </LineChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>
    </div>
  )
}
