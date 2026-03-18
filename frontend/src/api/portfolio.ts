import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { api } from './client'

// Types
export interface PortfolioOverview {
  totalValue: number
  cash: number
  positionsValue: number
  dailyPnL: number
  dailyPnLPercent: number
}

export interface Position {
  id: string
  ticker: string
  quantity: number
  entryPrice: number
  currentPrice: number
  pnl: number
  pnlPercent: number
}

export interface PerformanceDataPoint {
  date: string
  value: number
  pnl: number
}

// API hooks
export function usePortfolio() {
  return useQuery({
    queryKey: ['portfolio'],
    queryFn: () => api.get<PortfolioOverview>('/portfolio'),
  })
}

export function usePositions() {
  return useQuery({
    queryKey: ['positions'],
    queryFn: () => api.get<Position[]>('/portfolio/positions'),
  })
}

export function usePerformance(days: number = 30) {
  return useQuery({
    queryKey: ['performance', days],
    queryFn: () => api.get<PerformanceDataPoint[]>(`/portfolio/performance?days=${days}`),
  })
}

export function useClosePosition() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (positionId: string) =>
      api.post<{ success: boolean }>(`/portfolio/positions/${positionId}/close`, {}),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['portfolio'] })
      queryClient.invalidateQueries({ queryKey: ['positions'] })
    },
  })
}
