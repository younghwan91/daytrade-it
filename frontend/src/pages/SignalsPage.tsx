import { useState, useMemo } from 'react'
import { useSignals, useDeleteSignal, Signal } from '../api/signals'

interface FilterBarProps {
  ticker: string
  onTickerChange: (value: string) => void
  direction: 'ALL' | 'BUY' | 'SELL'
  onDirectionChange: (value: 'ALL' | 'BUY' | 'SELL') => void
  minConfidence: number
  onMinConfidenceChange: (value: number) => void
}

function FilterBar({
  ticker,
  onTickerChange,
  direction,
  onDirectionChange,
  minConfidence,
  onMinConfidenceChange,
}: FilterBarProps) {
  return (
    <div className="bg-white rounded-lg shadow p-4">
      <div className="flex flex-wrap items-center gap-4">
        <div className="flex-1 min-w-[200px]">
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Ticker
          </label>
          <input
            type="text"
            value={ticker}
            onChange={(e) => onTickerChange(e.target.value)}
            placeholder="Search by ticker..."
            className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500"
          />
        </div>

        <div className="w-40">
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Direction
          </label>
          <select
            value={direction}
            onChange={(e) => onDirectionChange(e.target.value as 'ALL' | 'BUY' | 'SELL')}
            className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500"
          >
            <option value="ALL">All</option>
            <option value="BUY">Buy</option>
            <option value="SELL">Sell</option>
          </select>
        </div>

        <div className="w-48">
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Min Confidence: {minConfidence}%
          </label>
          <input
            type="range"
            min="0"
            max="100"
            value={minConfidence}
            onChange={(e) => onMinConfidenceChange(Number(e.target.value))}
            className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
          />
        </div>
      </div>
    </div>
  )
}

function DirectionBadge({ direction }: { direction: 'BUY' | 'SELL' }) {
  const styles = {
    BUY: 'bg-green-100 text-green-800',
    SELL: 'bg-red-100 text-red-800',
  }

  return (
    <span className={`px-2 py-1 text-xs font-medium rounded ${styles[direction]}`}>
      {direction}
    </span>
  )
}

function ConfidenceBar({ confidence }: { confidence: number }) {
  const percentage = Math.round(confidence * 100)
  const barColor =
    percentage >= 80
      ? 'bg-green-500'
      : percentage >= 60
      ? 'bg-yellow-500'
      : 'bg-red-500'

  return (
    <div className="flex items-center gap-2">
      <div className="w-24 h-2 bg-gray-200 rounded-full overflow-hidden">
        <div
          className={`h-full ${barColor} transition-all`}
          style={{ width: `${percentage}%` }}
        />
      </div>
      <span className="text-sm text-gray-600 w-10">{percentage}%</span>
    </div>
  )
}

function formatDate(dateString: string): string {
  const date = new Date(dateString)
  return date.toLocaleString('en-US', {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  })
}

function SignalsTableSkeleton() {
  return (
    <div className="bg-white rounded-lg shadow overflow-hidden">
      <div className="px-6 py-4 border-b border-gray-200">
        <div className="h-6 w-32 bg-gray-200 rounded animate-pulse" />
      </div>
      <div className="divide-y divide-gray-200">
        {[...Array(5)].map((_, i) => (
          <div key={i} className="px-6 py-4">
            <div className="flex items-center gap-4">
              <div className="h-5 w-20 bg-gray-200 rounded animate-pulse" />
              <div className="h-5 w-16 bg-gray-200 rounded animate-pulse" />
              <div className="h-5 w-32 bg-gray-200 rounded animate-pulse" />
              <div className="h-5 w-24 bg-gray-200 rounded animate-pulse" />
              <div className="h-5 w-24 bg-gray-200 rounded animate-pulse" />
              <div className="h-5 w-24 bg-gray-200 rounded animate-pulse" />
              <div className="ml-auto flex gap-2">
                <div className="h-8 w-24 bg-gray-200 rounded animate-pulse" />
                <div className="h-8 w-20 bg-gray-200 rounded animate-pulse" />
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

function EmptyState() {
  return (
    <div className="bg-white rounded-lg shadow">
      <div className="px-6 py-12 text-center">
        <svg
          className="mx-auto h-12 w-12 text-gray-400"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"
          />
        </svg>
        <h3 className="mt-4 text-lg font-medium text-gray-900">No signals found</h3>
        <p className="mt-2 text-sm text-gray-500">
          No trading signals match your current filters. Try adjusting your criteria.
        </p>
      </div>
    </div>
  )
}

interface SignalsTableProps {
  signals: Signal[]
  onExecuteTrade: (signal: Signal) => void
  onDismiss: (signal: Signal) => void
}

function SignalsTable({ signals, onExecuteTrade, onDismiss }: SignalsTableProps) {
  if (signals.length === 0) {
    return <EmptyState />
  }

  return (
    <div className="bg-white rounded-lg shadow overflow-hidden">
      <div className="overflow-x-auto">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Ticker
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Direction
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Confidence
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Source
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Created
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Expires
              </th>
              <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                Actions
              </th>
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {signals.map((signal) => (
              <tr key={signal.id} className="hover:bg-gray-50">
                <td className="px-6 py-4 whitespace-nowrap">
                  <span className="font-medium text-gray-900">{signal.ticker}</span>
                </td>
                <td className="px-6 py-4 whitespace-nowrap">
                  <DirectionBadge direction={signal.direction} />
                </td>
                <td className="px-6 py-4 whitespace-nowrap">
                  <ConfidenceBar confidence={signal.confidence} />
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                  {signal.source}
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                  {formatDate(signal.createdAt)}
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                  {formatDate(signal.expiresAt)}
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-right">
                  <div className="flex justify-end gap-2">
                    <button
                      onClick={() => onExecuteTrade(signal)}
                      className="px-3 py-1.5 text-sm font-medium text-white bg-blue-600 rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
                    >
                      Execute Trade
                    </button>
                    <button
                      onClick={() => onDismiss(signal)}
                      className="px-3 py-1.5 text-sm font-medium text-gray-700 bg-gray-100 rounded-md hover:bg-gray-200 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-gray-500"
                    >
                      Dismiss
                    </button>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

export default function SignalsPage() {
  const [ticker, setTicker] = useState('')
  const [direction, setDirection] = useState<'ALL' | 'BUY' | 'SELL'>('ALL')
  const [minConfidence, setMinConfidence] = useState(0)

  const filters = useMemo(
    () => ({
      ticker: ticker || undefined,
      direction,
      minConfidence: minConfidence / 100,
    }),
    [ticker, direction, minConfidence]
  )

  const { data: signals, isLoading, error } = useSignals(filters)
  const deleteSignal = useDeleteSignal()

  const handleExecuteTrade = (signal: Signal) => {
    // TODO: Implement trade execution - navigate to trade form or open modal
    console.log('Execute trade for signal:', signal)
  }

  const handleDismiss = (signal: Signal) => {
    deleteSignal.mutate(signal.id)
  }

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-gray-900">Trading Signals</h1>
      </div>

      <FilterBar
        ticker={ticker}
        onTickerChange={setTicker}
        direction={direction}
        onDirectionChange={setDirection}
        minConfidence={minConfidence}
        onMinConfidenceChange={setMinConfidence}
      />

      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <p className="text-red-700">Failed to load signals. Please try again.</p>
        </div>
      )}

      {isLoading ? (
        <SignalsTableSkeleton />
      ) : (
        <SignalsTable
          signals={signals || []}
          onExecuteTrade={handleExecuteTrade}
          onDismiss={handleDismiss}
        />
      )}
    </div>
  )
}
