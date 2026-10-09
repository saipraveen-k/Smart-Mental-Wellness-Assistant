import type { Envelope } from '../types'

const TOKEN_KEY = 'smwa_token'

export function getToken(): string | null {
  return localStorage.getItem(TOKEN_KEY)
}

export function setToken(token: string | null) {
  if (token) localStorage.setItem(TOKEN_KEY, token)
  else localStorage.removeItem(TOKEN_KEY)
}

export class ApiError extends Error {
  code: string
  status: number
  suggestion?: string
  constructor(code: string, message: string, status: number, suggestion?: string) {
    super(message)
    this.code = code
    this.status = status
    this.suggestion = suggestion
  }
}

interface Options {
  method?: string
  body?: unknown
  formData?: FormData
  auth?: boolean
  headers?: Record<string, string>
}

export async function api<T>(path: string, options: Options = {}): Promise<T> {
  const headers: Record<string, string> = { ...(options.headers || {}) }
  const token = getToken()
  if (token && options.auth !== false) headers['Authorization'] = `Bearer ${token}`

  let body: BodyInit | undefined
  if (options.formData) {
    body = options.formData
  } else if (options.body !== undefined) {
    headers['Content-Type'] = 'application/json'
    body = JSON.stringify(options.body)
  }

  let res: Response
  try {
    res = await fetch(path, { method: options.method || 'GET', headers, body })
  } catch {
    throw new ApiError('NETWORK_ERROR', 'Could not reach the server.', 0, 'Check that the backend is running.')
  }

  let payload: Envelope<T>
  try {
    payload = await res.json()
  } catch {
    throw new ApiError('BAD_RESPONSE', `Unexpected response (${res.status}).`, res.status)
  }

  if (!res.ok || !payload.success) {
    if (res.status === 401 && token) {
      setToken(null)
      window.dispatchEvent(new Event('smwa:unauthorized'))
    }
    const err = payload.error
    throw new ApiError(err?.code || 'ERROR', err?.message || 'Request failed.', res.status, err?.suggestion)
  }
  return payload.data
}

export const apiUrl = (path: string) => path
