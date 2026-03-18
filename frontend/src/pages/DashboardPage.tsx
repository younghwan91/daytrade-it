interface StatCardProps {
  title: string
  value: string
  change?: string
  changeType?: 'positive' | 'negative' | 'neutral'
}

function StatCard({ title, value, change, changeType = 'neutral' }: StatCardProps) {
  const changeColors = {
    positive: 'text-green-600',
    negative: 'text-red-600',
    neutral: 'text-gray-500',
  }

  return (
    <div className="bg-white rounded-lg shadow p-6">
      <h3 className="text-sm font-medium text-gray-500">{title}</h3>
      <p className="mt-2 text-3xl font-bold text-gray-900">{value}</p>
      {change && (
        <p className={`mt-1 text-sm ${changeColors[changeType]}`}>{change}</p>
      )}
    </div>
  )
}

function PlaceholderChart({ title }: { title: string }) {
  return (
    <div className="bg-white rounded-lg shadow p-6">
      <h3 className="text-lg font-semibold text-gray-900 mb-4">{title}</h3>
      <div className="h-64 bg-gray-100 rounded flex items-center justify-center">
        <span className="text-gray-400">Chart placeholder</span>
      </div>
    </div>
  )
}

function RecentSignals() {
  const signals = [
    { id: 1, symbol: 'BTC/USDT', type: 'BUY', confidence: 0.85, time: '5m ago' },
    { id: 2, symbol: 'ETH/USDT', type: 'SELL', confidence: 0.72, time: '12m ago' },
    { id: 3, symbol: 'SOL/USDT', type: 'BUY', confidence: 0.91, time: '23m ago' },
    { id: 4, symbol: 'BNB/USDT', type: 'HOLD', confidence: 0.65, time: '35m ago' },
  ]

  return (
    <div className="bg-white rounded-lg shadow">
      <div className="px-6 py-4 border-b border-gray-200">
        <h3 className="text-lg font-semibold text-gray-900">Recent Signals</h3>
      </div>
      <ul className="divide-y divide-gray-200">
        {signals.map((signal) => (
          <li key={signal.id} className="px-6 py-4 flex items-center justify-between">
            <div>
              <p className="font-medium text-gray-900">{signal.symbol}</p>
              <p className="text-sm text-gray-500">{signal.time}</p>
            </div>
            <div className="flex items-center gap-4">
              <span
                className={`px-2 py-1 text-xs font-medium rounded ${
                  signal.type === 'BUY'
                    ? 'bg-green-100 text-green-800'
                    : signal.type === 'SELL'
                    ? 'bg-red-100 text-red-800'
                    : 'bg-gray-100 text-gray-800'
                }`}
              >
                {signal.type}
              </span>
              <span className="text-sm text-gray-600">
                {(signal.confidence * 100).toFixed(0)}%
              </span>
            </div>
          </li>
        ))}
      </ul>
    </div>
  )
}

function RecentTrades() {
  const trades = [
    { id: 1, symbol: 'BTC/USDT', side: 'BUY', amount: '0.05', price: '43,250', time: '2m ago' },
    { id: 2, symbol: 'ETH/USDT', side: 'SELL', amount: '1.2', price: '2,280', time: '15m ago' },
    { id: 3, symbol: 'SOL/USDT', side: 'BUY', amount: '10', price: '98.50', time: '1h ago' },
  ]

  return (
    <div className="bg-white rounded-lg shadow">
      <div className="px-6 py-4 border-b border-gray-200">
        <h3 className="text-lg font-semibold text-gray-900">Recent Trades</h3>
      </div>
      <ul className="divide-y divide-gray-200">
        {trades.map((trade) => (
          <li key={trade.id} className="px-6 py-4 flex items-center justify-between">
            <div>
              <p className="font-medium text-gray-900">{trade.symbol}</p>
              <p className="text-sm text-gray-500">{trade.time}</p>
            </div>
            <div className="text-right">
              <p className="font-medium text-gray-900">
                {trade.amount} @ ${trade.price}
              </p>
              <span
                className={`text-sm ${
                  trade.side === 'BUY' ? 'text-green-600' : 'text-red-600'
                }`}
              >
                {trade.side}
              </span>
            </div>
          </li>
        ))}
      </ul>
    </div>
  )
}

export default function DashboardPage() {
  return (
    <div className="p-6 space-y-6">
      {/* Overview Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <StatCard
          title="Portfolio Value"
          value="$125,430"
          change="+2.5% today"
          changeType="positive"
        />
        <StatCard
          title="Daily P&L"
          value="+$3,120"
          change="+2.5%"
          changeType="positive"
        />
        <StatCard
          title="Open Positions"
          value="12"
          change="3 in profit"
          changeType="neutral"
        />
        <StatCard
          title="Active Signals"
          value="8"
          change="2 high confidence"
          changeType="neutral"
        />
      </div>

      {/* Charts Area */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <PlaceholderChart title="Portfolio Performance" />
        <PlaceholderChart title="P&L Distribution" />
      </div>

      {/* Recent Activity */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <RecentSignals />
        <RecentTrades />
      </div>
    </div>
  )
}
