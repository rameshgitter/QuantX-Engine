"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"

interface OrderBookLevel {
  price: number
  quantity: number
  orders: number
}

interface OrderBook {
  bids: OrderBookLevel[]
  asks: OrderBookLevel[]
  spread: number
  midPrice: number
}

export function OrderBookView() {
  const [orderBook, setOrderBook] = useState<OrderBook>({
    bids: [],
    asks: [],
    spread: 0,
    midPrice: 0,
  })

  const [imbalance, setImbalance] = useState(0)
  const [microPrice, setMicroPrice] = useState(0)

  useEffect(() => {
    const generateOrderBook = () => {
      const midPrice = 18450 + Math.sin(Date.now() * 0.001) * 10
      const spread = 0.25 + Math.random() * 0.5

      const bids: OrderBookLevel[] = []
      const asks: OrderBookLevel[] = []

      // Generate bid levels
      for (let i = 0; i < 10; i++) {
        const price = midPrice - spread / 2 - i * 0.25
        const quantity = Math.floor(Math.random() * 500) + 100
        const orders = Math.floor(Math.random() * 10) + 1
        bids.push({ price, quantity, orders })
      }

      // Generate ask levels
      for (let i = 0; i < 10; i++) {
        const price = midPrice + spread / 2 + i * 0.25
        const quantity = Math.floor(Math.random() * 500) + 100
        const orders = Math.floor(Math.random() * 10) + 1
        asks.push({ price, quantity, orders })
      }

      // Calculate imbalance
      const bidVolume = bids.slice(0, 5).reduce((sum, level) => sum + level.quantity, 0)
      const askVolume = asks.slice(0, 5).reduce((sum, level) => sum + level.quantity, 0)
      const totalVolume = bidVolume + askVolume
      const imbalanceValue = totalVolume > 0 ? (bidVolume - askVolume) / totalVolume : 0

      // Calculate micro price
      const bestBid = bids[0]
      const bestAsk = asks[0]
      const microPriceValue =
        bestBid && bestAsk
          ? (bestBid.price * bestAsk.quantity + bestAsk.price * bestBid.quantity) /
            (bestBid.quantity + bestAsk.quantity)
          : midPrice

      setOrderBook({
        bids,
        asks,
        spread,
        midPrice,
      })
      setImbalance(imbalanceValue)
      setMicroPrice(microPriceValue)
    }

    generateOrderBook()
    const interval = setInterval(generateOrderBook, 500)

    return () => clearInterval(interval)
  }, [])

  const maxQuantity = Math.max(...orderBook.bids.map((b) => b.quantity), ...orderBook.asks.map((a) => a.quantity))

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
      {/* Order Book */}
      <Card className="bg-slate-800 border-slate-700 lg:col-span-2">
        <CardHeader>
          <CardTitle className="text-white">Level 2 Order Book</CardTitle>
          <CardDescription className="text-slate-400">
            Real-time bid/ask levels with depth visualization
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 gap-4">
            {/* Bids */}
            <div>
              <h3 className="text-sm font-medium text-green-400 mb-3">BIDS</h3>
              <div className="space-y-1">
                {orderBook.bids.slice(0, 10).map((bid, index) => (
                  <div key={index} className="relative">
                    <div
                      className="absolute inset-0 bg-green-500 opacity-20 rounded"
                      style={{ width: `${(bid.quantity / maxQuantity) * 100}%` }}
                    />
                    <div className="relative flex justify-between items-center p-2 text-xs">
                      <span className="text-green-400 font-mono">{bid.price.toFixed(2)}</span>
                      <span className="text-slate-300">{bid.quantity}</span>
                      <span className="text-slate-500">{bid.orders}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Asks */}
            <div>
              <h3 className="text-sm font-medium text-red-400 mb-3">ASKS</h3>
              <div className="space-y-1">
                {orderBook.asks.slice(0, 10).map((ask, index) => (
                  <div key={index} className="relative">
                    <div
                      className="absolute inset-0 bg-red-500 opacity-20 rounded"
                      style={{ width: `${(ask.quantity / maxQuantity) * 100}%` }}
                    />
                    <div className="relative flex justify-between items-center p-2 text-xs">
                      <span className="text-red-400 font-mono">{ask.price.toFixed(2)}</span>
                      <span className="text-slate-300">{ask.quantity}</span>
                      <span className="text-slate-500">{ask.orders}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Market Microstructure */}
      <div className="space-y-6">
        <Card className="bg-slate-800 border-slate-700">
          <CardHeader>
            <CardTitle className="text-white">Market Metrics</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <div className="flex justify-between items-center mb-2">
                <span className="text-sm text-slate-400">Spread</span>
                <span className="text-sm font-mono text-white">₹{orderBook.spread.toFixed(2)}</span>
              </div>
            </div>

            <div>
              <div className="flex justify-between items-center mb-2">
                <span className="text-sm text-slate-400">Mid Price</span>
                <span className="text-sm font-mono text-white">₹{orderBook.midPrice.toFixed(2)}</span>
              </div>
            </div>

            <div>
              <div className="flex justify-between items-center mb-2">
                <span className="text-sm text-slate-400">Micro Price</span>
                <span className="text-sm font-mono text-white">₹{microPrice.toFixed(2)}</span>
              </div>
            </div>

            <div>
              <div className="flex justify-between items-center mb-2">
                <span className="text-sm text-slate-400">Order Imbalance</span>
                <Badge variant={imbalance > 0 ? "default" : "destructive"}>{(imbalance * 100).toFixed(1)}%</Badge>
              </div>
              <Progress value={Math.abs(imbalance) * 100} className="h-2" />
            </div>
          </CardContent>
        </Card>

        <Card className="bg-slate-800 border-slate-700">
          <CardHeader>
            <CardTitle className="text-white">LOB Features</CardTitle>
            <CardDescription className="text-slate-400">Real-time feature extraction</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="flex justify-between">
              <span className="text-sm text-slate-400">Bid-Ask Ratio</span>
              <span className="text-sm text-white">1.23</span>
            </div>
            <div className="flex justify-between">
              <span className="text-sm text-slate-400">Volume Weighted Price</span>
              <span className="text-sm text-white">₹18,451.25</span>
            </div>
            <div className="flex justify-between">
              <span className="text-sm text-slate-400">Price Impact</span>
              <span className="text-sm text-white">0.045%</span>
            </div>
            <div className="flex justify-between">
              <span className="text-sm text-slate-400">Liquidity Score</span>
              <span className="text-sm text-white">8.7/10</span>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
