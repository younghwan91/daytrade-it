import { useState } from 'react'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts'
import { useRunBacktest, BacktestResult, Trade, EquityPoint } from '../api/backtest'

const STRATEGIES = [
  { value: 'momentum', label: 'Momentum' },
  { value: 'mean_reversion', label: 'Mean Reversion' },
  { value: 'trend_following', label: 'Trend Following' },
  { value: 'ml_ensemble', label: 'ML Ensemble' },
]

interface MetricCardProps {
  title: string
  value: string
  subtitle?: string
  type?: 'positive' | 'negative' | 'neutral'
}

function MetricCard({ title, value, subtitle, type = 'neutral' }: MetricCardProps) {
  const colors = {
    positive: 'text-green-600',
    negative: 'text-red-600',
    neutral: 'text-gray-900',
  }

  return (
    <div className="bg-white rounded-lg shadow p-6">
      <h3 className="text-sm font-medium text-gray-500">{title}</h3>
      <p className={`mt-2 text-3xl font-bold ${colors[type]}`}>{value}</p>
      {subtitle && <p className="mt-1 text-sm text-gray-500">{subtitle}</p>}
    </div>
  )
}

interface EquityCurveChartProps {
  data: EquityPoint[]
}

function EquityCurveChart({ data }: EquityCurveChartProps) {
  return (
    <div className="bg-white rounded-lg shadow p-6">
      <h3 className="text-lg font-semibold text-gray-900 mb-4">Equity Curve</h3>
      <div className="h-80">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
            <XAxis
              dataKey="date"
              tick={{ fontSize: 12 }}
              tickFormatter={(value) => new Date(value).toLocaleDateString()}
            />
            <YAxis
              tick={{ fontSize: 12 }}
              tickFormatter={(value) => `$${value.toLocaleString()}`}
            />
            <Tooltip
              formatter={(value: number) => [`$${value.toLocaleString()}`, 'Equity']}
              labelFormatter={(label) => new Date(label).toLocaleDateString()}
            />
            <Line
              type="monotone"
              dataKey="equity"
              stroke="#3b82f6"
              strokeWidth={2}
              dot={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}

interface TradesTableProps {
  trades: Trade[]
}

function TradesTable({ trades }: TradesTableProps) {
  return (
    <div className="bg-white rounded-lg shadow overflow-hidden">
      <div className="px-6 py-4 border-b border-gray-200">
        <h3 className="text-lg font-semibold text-gray-900">Trades</h3>
      </div>
      <div className="overflow-x-auto">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Date
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Ticker
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Direction
              </th>
              <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                Entry
              </th>
              <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                Exit
              </th>
              <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                P&L
              </th>
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {trades.map((trade) => (
              <tr key={trade.id} className="hover:bg-gray-50">
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                  {new Date(trade.date).toLocaleDateString()}
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                  {trade.ticker}
                </td>
                <td className="px-6 py-4 whitespace-nowrap">
                  <span
                    className={`px-2 py-1 text-xs font-medium rounded ${
                      trade.direction === 'BUY'
                        ? 'bg-green-100 text-green-800'
                        : 'bg-red-100 text-red-800'
                    }`}
                  >
                    {trade.direction}
                  </span>
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 text-right">
                  ${trade.entryPrice.toFixed(2)}
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 text-right">
                  ${trade.exitPrice.toFixed(2)}
                </td>
                <td
                  className={`px-6 py-4 whitespace-nowrap text-sm font-medium text-right ${
                    trade.pnl >= 0 ? 'text-green-600' : 'text-red-600'
                  }`}
                >
                  {trade.pnl >= 0 ? '+' : ''}${trade.pnl.toFixed(2)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

interface BacktestResultsProps {
  result: BacktestResult
}

function BacktestResults({ result }: BacktestResultsProps) {
  const { metrics, equityCurve, trades } = result

  const formatPercent = (value: number) => `${(value * 100).toFixed(2)}%`
  const formatNumber = (value: number) => value.toFixed(2)

  return (
    <div className="space-y-6">
      {/* Metrics Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <MetricCard
          title="Total Return"
          value={formatPercent(metrics.totalReturn)}
          type={metrics.totalReturn >= 0 ? 'positive' : 'negative'}
        />
        <MetricCard
          title="Sharpe Ratio"
          value={formatNumber(metrics.sharpeRatio)}
          type={metrics.sharpeRatio >= 1 ? 'positive' : metrics.sharpeRatio >= 0 ? 'neutral' : 'negative'}
        />
        <MetricCard
          title="Max Drawdown"
          value={formatPercent(metrics.maxDrawdown)}
          type="negative"
        />
        <MetricCard
          title="Win Rate"
          value={formatPercent(metrics.winRate)}
          type={metrics.winRate >= 0.5 ? 'positive' : 'negative'}
        />
      </div>

      {/* Equity Curve Chart */}
      <EquityCurveChart data={equityCurve} />

      {/* Trades Table */}
      <TradesTable trades={trades} />
    </div>
  )
}

export default function BacktestPage() {
  const [tickers, setTickers] = useState('AAPL, GOOGL, MSFT')
  const [startDate, setStartDate] = useState('2023-01-01')
  const [endDate, setEndDate] = useState('2023-12-31')
  const [initialCapital, setInitialCapital] = useState('100000')
  const [strategy, setStrategy] = useState('momentum')
  const [result, setResult] = useState<BacktestResult | null>(null)

  const runBacktest = useRunBacktest()

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    
    const config = {
      tickers: tickers.split(',').map((t) => t.trim()).filter(Boolean),
      startDate,
      endDate,
      initialCapital: parseFloat(initialCapital),
      strategy,
    }

    runBacktest.mutate(config, {
      onSuccess: (data) => setResult(data),
    })
  }

  return (
    <div className="p-6 space-y-6">
      {/* Configuration Form */}
      <div className="bg-white rounded-lg shadow p-6">
        <h2 className="text-xl font-semibold text-gray-900 mb-6">Backtest Configuration</h2>
        <form onSubmit={handleSubmit} className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {/* Tickers Input */}
            <div className="lg:col-span-2">
              <label htmlFor="tickers" className="block text-sm font-medium text-gray-700 mb-1">
                Tickers (comma-separated)
              </label>
              <input
                type="text"
                id="tickers"
                value={tickers}
                onChange={(e) => setTickers(e.target.value)}
                placeholder="AAPL, GOOGL, MSFT"
                className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none"
              />
            </div>

            {/* Strategy Dropdown */}
            <div>
              <label htmlFor="strategy" className="block text-sm font-medium text-gray-700 mb-1">
                Strategy
              </label>
              <select
                id="strategy"
                value={strategy}
                onChange={(e) => setStrategy(e.target.value)}
                className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none bg-white"
              >
                {STRATEGIES.map((s) => (
                  <option key={s.value} value={s.value}>
                    {s.label}
                  </option>
                ))}
              </select>
            </div>

            {/* Start Date */}
            <div>
              <label htmlFor="startDate" className="block text-sm font-medium text-gray-700 mb-1">
                Start Date
              </label>
              <input
                type="date"
                id="startDate"
                value={startDate}
                onChange={(e) => setStartDate(e.target.value)}
                className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none"
              />
            </div>

            {/* End Date */}
            <div>
              <label htmlFor="endDate" className="block text-sm font-medium text-gray-700 mb-1">
                End Date
              </label>
              <input
                type="date"
                id="endDate"
                value={endDate}
                onChange={(e) => setEndDate(e.target.value)}
                className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none"
              />
            </div>

            {/* Initial Capital */}
            <div>
              <label htmlFor="initialCapital" className="block text-sm font-medium text-gray-700 mb-1">
                Initial Capital ($)
              </label>
              <input
                type="number"
                id="initialCapital"
                value={initialCapital}
                onChange={(e) => setInitialCapital(e.target.value)}
                min="1000"
                step="1000"
                className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none"
              />
            </div>
          </div>

          {/* Submit Button */}
          <div className="flex justify-end">
            <button
              type="submit"
              disabled={runBacktest.isPending}
              className="px-6 py-2.5 bg-blue-600 text-white font-medium rounded-lg hover:bg-blue-700 focus:ring-4 focus:ring-blue-300 disabled:bg-blue-400 disabled:cursor-not-allowed transition-colors"
            >
              {runBacktest.isPending ? (
                <span className="flex items-center gap-2">
                  <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
                    <circle
                      className="opacity-25"
                      cx="12"
                      cy="12"
                      r="10"
                      stroke="currentColor"
                      strokeWidth="4"
                      fill="none"
                    />
                    <path
                      className="opacity-75"
                      fill="currentColor"
                      d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                    />
                  </svg>
                  Running...
                </span>
              ) : (
                'Run Backtest'
              )}
            </button>
          </div>

          {/* Error Message */}
          {runBacktest.isError && (
            <div className="p-4 bg-red-50 border border-red-200 rounded-lg">
              <p className="text-sm text-red-600">
                Failed to run backtest. Please try again.
              </p>
            </div>
          )}
        </form>
      </div>

      {/* Loading State */}
      {runBacktest.isPending && (
        <div className="bg-white rounded-lg shadow p-12 flex flex-col items-center justify-center">
          <svg className="animate-spin h-12 w-12 text-blue-600 mb-4" viewBox="0 0 24 24">
            <circle
              className="opacity-25"
              cx="12"
              cy="12"
              r="10"
              stroke="currentColor"
              strokeWidth="4"
              fill="none"
            />
            <path
              className="opacity-75"
              fill="currentColor"
              d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
            />
          </svg>
          <p className="text-lg font-medium text-gray-900">Running Backtest...</p>
          <p className="text-sm text-gray-500 mt-1">This may take a few moments</p>
        </div>
      )}

      {/* Results Section */}
      {result && !runBacktest.isPending && <BacktestResults result={result} />}
    </div>
  )
}
