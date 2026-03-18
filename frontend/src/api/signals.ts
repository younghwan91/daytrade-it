import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { api } from './client'

export interface Signal {
  id: string
  ticker: string
  direction: 'BUY' | 'SELL'
  confidence: number
  source: string
  createdAt: string
  expiresAt: string
}

export interface SignalFilters {
  ticker?: string
  direction?: 'ALL' | 'BUY' | 'SELL'
  minConfidence?: number
}

export interface CreateSignalInput {
  ticker: string
  direction: 'BUY' | 'SELL'
  confidence: number
  source: string
  expiresAt: string
}

async function fetchSignals(filters: SignalFilters): Promise<Signal[]> {
  const params = new URLSearchParams()
  if (filters.ticker) params.set('ticker', filters.ticker)
  if (filters.direction && filters.direction !== 'ALL') {
    params.set('direction', filters.direction)
  }
  if (filters.minConfidence !== undefined) {
    params.set('minConfidence', filters.minConfidence.toString())
  }
  const query = params.toString()
  return api.get<Signal[]>(`/signals${query ? `?${query}` : ''}`)
}

async function createSignal(input: CreateSignalInput): Promise<Signal> {
  return api.post<Signal>('/signals', input)
}

async function deleteSignal(id: string): Promise<void> {
  return api.delete(`/signals/${id}`)
}

export function useSignals(filters: SignalFilters = {}) {
  return useQuery({
    queryKey: ['signals', filters],
    queryFn: () => fetchSignals(filters),
  })
}

export function useCreateSignal() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: createSignal,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['signals'] })
    },
  })
}

export function useDeleteSignal() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: deleteSignal,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['signals'] })
    },
  })
}
