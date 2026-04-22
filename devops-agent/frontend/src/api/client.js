/**
 * API 客户端 —— 统一封装后端所有接口
 * 后端地址通过 Vite proxy 转发，前端直接用相对路径
 */

const API_BASE = '/api/v1'

async function request(url, options = {}) {
  const res = await fetch(url, {
    headers: {
      'Content-Type': 'application/json',
      ...options.headers,
    },
    ...options,
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || err.message || `HTTP ${res.status}`)
  }
  return res.json()
}

// ============================================================
//  健康检查
// ============================================================
export const healthCheck = () => fetch('/health').then(r => r.json())
export const appInfo = () => request(`${API_BASE}/info`)

// ============================================================
//  会话管理
// ============================================================
export const listSessions = (page = 1, pageSize = 20) =>
  request(`${API_BASE}/sessions?page=${page}&page_size=${pageSize}`)

export const getSession = (id) => request(`${API_BASE}/sessions/${id}`)

export const deleteSession = (id) =>
  request(`${API_BASE}/sessions/${id}`, { method: 'DELETE' })

export const createSession = (title = '新对话') =>
  request(`${API_BASE}/sessions`, {
    method: 'POST',
    body: JSON.stringify({ title }),
  })

// ============================================================
//  对话
// ============================================================
export const sendChat = (message, sessionId = null) =>
  request(`${API_BASE}/chat`, {
    method: 'POST',
    body: JSON.stringify({ message, session_id: sessionId }),
  })

export function streamChat(message, sessionId = null, onEvent) {
  return new Promise((resolve, reject) => {
    const evtSource = new EventSource(
      `${API_BASE}/chat/stream`,
      { withCredentials: false }
    )
    // SSE 不支持 POST，后端实际用 POST，这里需要改用 fetch + ReadableStream
    // 但 EventSource 只支持 GET，所以后端需要支持 GET 或我们用 fetch
    reject(new Error('请使用 streamChatFetch'))
  })
}

export async function streamChatFetch(message, sessionId, onEvent) {
  const response = await fetch(`${API_BASE}/chat/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message, session_id: sessionId }),
  })

  if (!response.ok) {
    const err = await response.json().catch(() => ({}))
    throw new Error(err.detail || `HTTP ${response.status}`)
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  function _flushBuffer() {
    const lines = buffer.split('\n')
    buffer = ''
    let currentEvent = null
    let currentData = null
    for (const line of lines) {
      const trimmed = line.trim()
      if (trimmed.startsWith('event:')) {
        currentEvent = trimmed.slice(6).trim()
      } else if (trimmed.startsWith('data:')) {
        currentData = trimmed.slice(5).trim()
      } else if (trimmed === '' && currentEvent && currentData) {
        try {
          const payload = JSON.parse(currentData)
          onEvent(currentEvent, payload)
        } catch (e) {
          onEvent(currentEvent, { raw: currentData })
        }
        currentEvent = null
        currentData = null
      }
    }
  }

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })

    const lines = buffer.split('\n')
    buffer = lines.pop() || ''

    let currentEvent = null
    let currentData = null

    for (const line of lines) {
      const trimmed = line.trim()
      if (trimmed.startsWith('event:')) {
        currentEvent = trimmed.slice(6).trim()
      } else if (trimmed.startsWith('data:')) {
        currentData = trimmed.slice(5).trim()
      } else if (trimmed === '' && currentEvent && currentData) {
        try {
          const payload = JSON.parse(currentData)
          onEvent(currentEvent, payload)
        } catch (e) {
          onEvent(currentEvent, { raw: currentData })
        }
        currentEvent = null
        currentData = null
      }
    }
  }

  // 流结束后 flush 剩余 buffer（防止最后一个事件因缺少尾换行而丢失）
  _flushBuffer()
}

export const getChatHistory = (sessionId, page = 1, pageSize = 50) =>
  request(`${API_BASE}/chat/history?session_id=${sessionId}&page=${page}&page_size=${pageSize}`)

// ============================================================
//  OS 探针
// ============================================================
export const probeDisk = (path = '/var/log') =>
  request(`${API_BASE}/probe/disk?path=${encodeURIComponent(path)}`)

export const probeProcesses = (params = {}) => {
  const qs = new URLSearchParams(params).toString()
  return request(`${API_BASE}/probe/processes${qs ? '?' + qs : ''}`)
}

export const probeNetwork = (action = 'connections', hostname = null) => {
  let url = `${API_BASE}/probe/network?action=${action}`
  if (hostname) url += `&hostname=${encodeURIComponent(hostname)}`
  return request(url)
}

export const probeLogs = (params = {}) => {
  const qs = new URLSearchParams(params).toString()
  return request(`${API_BASE}/probe/logs${qs ? '?' + qs : ''}`)
}

// ============================================================
//  命令执行
// ============================================================
export const executeCommand = (command, timeout = 30, dryRun = false, sessionId = null) =>
  request(`${API_BASE}/execute`, {
    method: 'POST',
    body: JSON.stringify({ command, timeout, dry_run: dryRun, session_id: sessionId }),
  })

// ============================================================
//  审计日志
// ============================================================
export const queryAudit = (params = {}) => {
  const qs = new URLSearchParams(params).toString()
  return request(`${API_BASE}/audit${qs ? '?' + qs : ''}`)
}

export const auditStats = () => request(`${API_BASE}/audit/stats`)

// ============================================================
//  推理链路
// ============================================================
export const getReasoningChain = (sessionId, roundNumber = null) => {
  let url = `${API_BASE}/reasoning/${sessionId}`
  if (roundNumber !== null) url += `?round_number=${roundNumber}`
  return request(url)
}

export const getReasoningSummary = (sessionId) =>
  request(`${API_BASE}/reasoning/${sessionId}/summary`)

// ============================================================
//  安全层
// ============================================================
export const safetyStatus = () => request(`${API_BASE}/safety/status`)

export const validateCommand = (command, context = null) =>
  request(`${API_BASE}/safety/validate`, {
    method: 'POST',
    body: JSON.stringify({ command, context }),
  })

export const validateBatch = (commands) =>
  request(`${API_BASE}/safety/validate/batch`, {
    method: 'POST',
    body: JSON.stringify({ commands }),
  })

export const safetyExecute = (command, user = 'devops-runner', timeout = 30) =>
  request(`${API_BASE}/safety/execute`, {
    method: 'POST',
    body: JSON.stringify({ command, user, timeout }),
  })

export const configBaseline = (paths = null) =>
  request(`${API_BASE}/safety/config/baseline`, {
    method: 'POST',
    body: JSON.stringify(paths ? { paths } : {}),
  })

export const configScan = (quick = true) =>
  request(`${API_BASE}/safety/config/scan`, {
    method: 'POST',
    body: JSON.stringify({ quick }),
  })

export const configPaths = () => request(`${API_BASE}/safety/config/paths`)

export const injectionScan = (text) =>
  request(`${API_BASE}/safety/injection/scan`, {
    method: 'POST',
    body: JSON.stringify({ text }),
  })

export const injectionStats = () => request(`${API_BASE}/safety/injection/stats`)
