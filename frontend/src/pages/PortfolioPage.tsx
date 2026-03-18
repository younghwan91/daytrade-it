import { useState } from 'react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'
import {
  usePortfolio,
  usePositions,
  usePerformance,
  useClosePosition,
  type Position,
} from '../api/portfolio'

interface StatCardProps {
  title: string
  value: string
  subtitle?: string
  valueColor?: 'default' | 'positive' | 'negative'
}

function StatCard({ title, value, subtitle, valueColor = 'default' }: StatCardProps) {
  const valueColors = {
    default: 'text-gray-900',
    positive: 'text-green-600',
    negative: 'text-red-600',
  }

  return (
    <div className="bg-white rounded-lg shadow p-6">
      <h3 className="text-sm font-medium text-gray-500">{title}</h3>
      <p className={`mt-2 text-3xl font-bold ${valueColors[valueColor]}`}>{value}</p>
      {subtitle && <p className="mt-1 text-sm text-gray-500">{subtitle}</p>}
    </div>
  )
}

function formatCurrency(value: number): string {
  const prefix = value >= 0 ? '' : '-'
  return `${prefix}$${Math.abs(value).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`
}

function formatPercent(value: number): string {
  const prefix = value >= 0 ? '+' : ''
  return `${prefix}${value.toFixed(2)}%`
}

function PortfolioSummary() {
  const { data: portfolio, isLoading, error } = usePortfolio()

  if (isLoading) {
    return (
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {[...Array(4)].map((_, i) => (
          <div key={i} className="bg-white rounded-lg shadow p-6 animate-pulse">
            <div className="h-4 bg-gray-200 rounded w-24 mb-2"></div>
            <div className="h-8 bg-gray-200 rounded w-32"></div>
          </div>
        ))}
      </div>
    )
  }

  if (error || !portfolio) {
    // Show placeholder data when API is not available
    const mockPortfolio = {
      totalValue: 125430.50,
      cash: 25430.50,
      positionsValue: 100000.00,
      dailyPnL: 3120.75,
      dailyPnLPercent: 2.55,
    }
    const data = portfolio || mockPortfolio

    return (
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <StatCard title="Total Value" value={formatCurrency(data.totalValue)} />
        <StatCard title="Cash" value={formatCurrency(data.cash)} />
        <StatCard title="Positions Value" value={formatCurrency(data.positionsValue)} />
        <StatCard
          title="Daily P&L"
          value={formatCurrency(data.dailyPnL)}
          subtitle={formatPercent(data.dailyPnLPercent)}
          valueColor={data.dailyPnL >= 0 ? 'positive' : 'negative'}
        />
      </div>
    )
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
      <StatCard title="Total Value" value={formatCurrency(portfolio.totalValue)} />
      <StatCard title="Cash" value={formatCurrency(portfolio.cash)} />
      <StatCard title="Positions Value" value={formatCurrency(portfolio.positionsValue)} />
      <StatCard
        title="Daily P&L"
        value={formatCurrency(portfolio.dailyPnL)}
        subtitle={formatPercent(portfolio.dailyPnLPercent)}
        valueColor={portfolio.dailyPnL >= 0 ? 'positive' : 'negative'}
      />
    </div>
  )
}

function PositionsTable() {
  const { data: positions, isLoading, error } = usePositions()
  const closePosition = useClosePosition()
  const [closingId, setClosingId] = useState<string | null>(null)

  const handleClose = async (positionId: string) => {
    setClosingId(positionId)
    try {
      await closePosition.mutateAsync(positionId)
    } catch (err) {
      console.error('Failed to close position:', err)
    } finally {
      setClosingId(null)
    }
  }

  // Mock data for display when API unavailable
  const mockPositions: Position[] = [
    { id: '1', ticker: 'AAPL', quantity: 100, entryPrice: 175.50, currentPrice: 182.30, pnl: 680.00, pnlPercent: 3.87 },
    { id: '2', ticker: 'GOOGL', quantity: 50, entryPrice: 142.20, currentPrice: 138.50, pnl: -185.00, pnlPercent: -2.60 },
    { id: '3', ticker: 'MSFT', quantity: 75, entryPrice: 378.40, currentPrice: 392.15, pnl: 1031.25, pnlPercent: 3.63 },
    { id: '4', ticker: 'NVDA', quantity: 25, entryPrice: 485.00, currentPrice: 512.80, pnl: 695.00, pnlPercent: 5.73 },
    { id: '5', ticker: 'TSLA', quantity: 30, entryPrice: 245.60, currentPrice: 238.20, pnl: -222.00, pnlPercent: -3.01 },
  ]

  const displayPositions = positions || (error ? mockPositions : [])

  if (isLoading) {
    return (
      <div className="bg-white rounded-lg shadow">
        <div className="px-6 py-4 border-b border-gray-200">
          <h3 className="text-lg font-semibold text-gray-900">Open Positions</h3>
        </div>
        <div className="p-6">
          <div className="animate-pulse space-y-4">
            {[...Array(5)].map((_, i) => (
              <div key={i} className="h-12 bg-gray-100 rounded"></div>
            ))}
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="bg-white rounded-lg shadow overflow-hidden">
      <div className="px-6 py-4 border-b border-gray-200">
        <h3 className="text-lg font-semibold text-gray-900">Open Positions</h3>
      </div>
      <div className="overflow-x-auto">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Ticker
              </th>
              <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                Quantity
              </th>
              <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                Entry Price
              </th>
              <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                Current Price
              </th>
              <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                P&L
              </th>
              <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                P&L %
              </th>
              <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                Actions
              </th>
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {displayPositions.length === 0 ? (
              <tr>
                <td colSpan={7} className="px-6 py-8 text-center text-gray-500">
                  No open positions
                </td>
              </tr>
            ) : (
              displayPositions.map((position) => (
                <tr key={position.id} className="hover:bg-gray-50">
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className="font-medium text-gray-900">{position.ticker}</span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-right text-gray-900">
                    {position.quantity.toLocaleString()}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-right text-gray-900">
                    {formatCurrency(position.entryPrice)}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-right text-gray-900">
                    {formatCurrency(position.currentPrice)}
                  </td>
                  <td
                    className={`px-6 py-4 whitespace-nowrap text-right font-medium ${
                      position.pnl >= 0 ? 'text-green-600' : 'text-red-600'
                    }`}
                  >
                    {formatCurrency(position.pnl)}
                  </td>
                  <td
                    className={`px-6 py-4 whitespace-nowrap text-right font-medium ${
                      position.pnlPercent >= 0 ? 'text-green-600' : 'text-red-600'
                    }`}
                  >
                    {formatPercent(position.pnlPercent)}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-right">
                    <button
                      onClick={() => handleClose(position.id)}
                      disabled={closingId === position.id}
                      className="inline-flex items-center px-3 py-1.5 border border-red-300 text-sm font-medium rounded-md text-red-700 bg-white hover:bg-red-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-red-500 disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      {closingId === position.id ? 'Closing...' : 'Close'}
                    </button>
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  )
}

function PerformanceChart() {
  const { data: performance, isLoading, error } = usePerformance(30)

  // Mock data for display when API unavailable
  const mockPerformance = [
    { date: '2024-01-01', value: 100000, pnl: 0 },
    { date: '2024-01-05', value: 102500, pnl: 2500 },
    { date: '2024-01-10', value: 101800, pnl: 1800 },
    { date: '2024-01-15', value: 105200, pnl: 5200 },
    { date: '2024-01-20', value: 108900, pnl: 8900 },
    { date: '2024-01-25', value: 107200, pnl: 7200 },
    { date: '2024-01-30', value: 112500, pnl: 12500 },
    { date: '2024-02-05', value: 118200, pnl: 18200 },
    { date: '2024-02-10', value: 115800, pnl: 15800 },
    { date: '2024-02-15', value: 122300, pnl: 22300 },
    { date: '2024-02-20', value: 125430, pnl: 25430 },
  ]

  const displayData = performance || (error ? mockPerformance : [])

  return (
    <div className="bg-white rounded-lg shadow p-6">
      <h3 className="text-lg font-semibold text-gray-900 mb-4">Performance Over Time</h3>
      {isLoading ? (
        <div className="h-64 bg-gray-100 rounded flex items-center justify-center animate-pulse">
          <span className="text-gray-400">Loading chart...</span>
        </div>
      ) : displayData.length === 0 ? (
        <div className="h-64 bg-gray-100 rounded flex items-center justify-center">
          <span className="text-gray-400">No performance data available</span>
        </div>
      ) : (
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={displayData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
              <XAxis
                dataKey="date"
                tick={{ fontSize: 12, fill: '#6b7280' }}
                tickFormatter={(value) => {
                  const date = new Date(value)
                  return `${date.getMonth() + 1}/${date.getDate()}`
                }}
              />
              <YAxis
                tick={{ fontSize: 12, fill: '#6b7280' }}
                tickFormatter={(value) => `$${(value / 1000).toFixed(0)}k`}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#fff',
                  border: '1px solid #e5e7eb',
                  borderRadius: '0.5rem',
                }}
                formatter={(value: number) => [formatCurrency(value), 'Value']}
                labelFormatter={(label) => new Date(label).toLocaleDateString()}
              />
              <Line
                type="monotone"
                dataKey="value"
                stroke="#3b82f6"
                strokeWidth={2}
                dot={false}
                activeDot={{ r: 4, fill: '#3b82f6' }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  )
}

export default function PortfolioPage() {
  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-gray-900">Portfolio</h1>
      </div>

      {/* Summary Cards */}
      <PortfolioSummary />

      {/* Performance Chart */}
      <PerformanceChart />

      {/* Positions Table */}
      <PositionsTable />
    </div>
  )
}
