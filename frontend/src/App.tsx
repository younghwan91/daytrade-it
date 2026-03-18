import { Routes, Route } from 'react-router-dom'
import Layout from './components/Layout'
import DashboardPage from './pages/DashboardPage'
import SignalsPage from './pages/SignalsPage'
import PortfolioPage from './pages/PortfolioPage'
import BacktestPage from './pages/BacktestPage'

function PlaceholderPage({ title }: { title: string }) {
  return (
    <div className="p-6">
      <p className="text-gray-600">
        {title} page content coming soon...
      </p>
    </div>
  )
}

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<Layout />}>
        <Route index element={<DashboardPage />} />
        <Route path="signals" element={<SignalsPage />} />
        <Route path="trades" element={<PlaceholderPage title="Trades" />} />
        <Route path="portfolio" element={<PortfolioPage />} />
        <Route path="backtest" element={<BacktestPage />} />
      </Route>
    </Routes>
  )
}
