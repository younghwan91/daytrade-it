import { useMutation, useQuery } from '@tanstack/react-query'
import { api } from './client'

export interface BacktestConfig {
  tickers: string[]
  startDate: string
  endDate: string
  initialCapital: number
  strategy: string
}

export interface Trade {
  id: string
  date: string
  ticker: string
  direction: 'BUY' | 'SELL'
  entryPrice: number
  exitPrice: number
  pnl: number
}

export interface EquityPoint {
  date: string
  equity: number
}

export interface BacktestResult {
  id: string
  config: BacktestConfig
  metrics: {
    totalReturn: number
    sharpeRatio: number
    maxDrawdown: number
    winRate: number
  }
  equityCurve: EquityPoint[]
  trades: Trade[]
}

export function useRunBacktest() {
  return useMutation({
    mutationFn: (config: BacktestConfig) =>
      api.post<BacktestResult>('/backtest/run', config),
  })
}

export function useBacktestResults(id: string | null) {
  return useQuery({
    queryKey: ['backtest', id],
    queryFn: () => api.get<BacktestResult>(`/backtest/results/${id}`),
    enabled: !!id,
  })
}
