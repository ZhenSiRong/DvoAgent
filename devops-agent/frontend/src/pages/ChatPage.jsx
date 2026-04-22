import { useState, useRef, useEffect, useCallback } from 'react'
import { Send, Loader2, Bot, User, Sparkles, Terminal, AlertTriangle, MessageSquare } from 'lucide-react'
import { streamChatFetch, sendChat, listSessions, getSession, deleteSession } from '../api/client'

const EVENT_LABELS = {
  start: '开始处理',
  sense: '环境感知',
  analyze: '分析推理',
  plan: '制定方案',
  execute: '执行工具',
  execute_done: '工具完成',
  output: '生成回复',
  done: '完成',
  error: '错误',
}

const EVENT_COLORS = {
  start: 'text-slate-400',
  sense: 'text-cyan-400',
  analyze: 'text-amber-400',
  plan: 'text-violet-400',
  execute: 'text-orange-400',
  execute_done: 'text-emerald-400',
  output: 'text-primary-400',
  done: 'text-emerald-400',
  error: 'text-red-400',
}

export default function ChatPage() {
  const [sessions, setSessions] = useState([])
  const [currentSessionId, setCurrentSessionId] = useState(null)
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [isStreaming, setIsStreaming] = useState(false)
  const [streamEvents, setStreamEvents] = useState([])
  const [activeEvent, setActiveEvent] = useState(null)
  const messagesEndRef = useRef(null)
  const inputRef = useRef(null)

  // 加载会话列表
  useEffect(() => {
    loadSessions()
  }, [])

  // 自动滚动到底部
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, streamEvents])

  const loadSessions = async () => {
    try {
      const res = await listSessions(1, 50)
      if (res.code === 0) {
        setSessions(res.data.items || [])
      }
    } catch (e) {
      console.error('加载会话失败:', e)
    }
  }

  const loadSessionMessages = async (sessionId) => {
    try {
      const res = await getSession(sessionId)
      if (res.code === 0) {
        const msgs = (res.data.messages || []).map(m => ({
          role: m.role,
          content: m.content,
          id: Math.random().toString(36).slice(2),
        }))
        setMessages(msgs)
      }
    } catch (e) {
      console.error('加载消息失败:', e)
    }
  }

  const handleNewChat = () => {
    setCurrentSessionId(null)
    setMessages([])
    setInput('')
    inputRef.current?.focus()
  }

  const handleSelectSession = (sessionId) => {
    setCurrentSessionId(sessionId)
    loadSessionMessages(sessionId)
  }

  const handleDeleteSession = async (e, sessionId) => {
    e.stopPropagation()
    if (!confirm('确定删除此会话？')) return
    try {
      await deleteSession(sessionId)
      setSessions(prev => prev.filter(s => s.session_id !== sessionId))
      if (currentSessionId === sessionId) {
        handleNewChat()
      }
    } catch (e) {
      alert('删除失败: ' + e.message)
    }
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!input.trim() || isStreaming) return

    const userMessage = input.trim()
    setInput('')
    setIsStreaming(true)
    setStreamEvents([])
    setActiveEvent(null)

    // 添加用户消息到界面
    const userMsgId = Date.now().toString()
    setMessages(prev => [...prev, { role: 'user', content: userMessage, id: userMsgId }])

    let assistantReply = ''
    let finalSessionId = currentSessionId
    let streamError = null

    try {
      await streamChatFetch(userMessage, currentSessionId, (eventType, payload) => {
        setActiveEvent(eventType)
        setStreamEvents(prev => [...prev, { type: eventType, payload, time: Date.now() }])

        if (eventType === 'output') {
          assistantReply = payload.reply || ''
          finalSessionId = payload.session_id || currentSessionId
        }
        if (eventType === 'error') {
          streamError = payload.detail || payload.message || '未知错误'
        }
      })

      // 添加助手回复（或错误提示）
      if (assistantReply) {
        setMessages(prev => [...prev, {
          role: 'assistant',
          content: assistantReply,
          id: 'assistant-' + Date.now(),
        }])
      } else if (streamError) {
        setMessages(prev => [...prev, {
          role: 'assistant',
          content: `❌ Agent 推理异常: ${streamError}`,
          id: 'error-' + Date.now(),
          isError: true,
        }])
      }

      // 更新当前会话ID
      if (finalSessionId && finalSessionId !== currentSessionId) {
        setCurrentSessionId(finalSessionId)
        await loadSessions()
      }
    } catch (err) {
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: `❌ 请求失败: ${err.message}`,
        id: 'error-' + Date.now(),
        isError: true,
      }])
    } finally {
      setIsStreaming(false)
      setActiveEvent(null)
    }
  }

  const formatTime = (iso) => {
    if (!iso) return ''
    const d = new Date(iso)
    return d.toLocaleString('zh-CN', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })
  }

  return (
    <div className="flex h-[calc(100vh-3.5rem)] lg:h-screen">
      {/* 会话侧边栏 */}
      <div className="hidden md:flex w-64 flex-col border-r border-slate-800 bg-slate-900/50">
        <div className="p-3 border-b border-slate-800">
          <button
            onClick={handleNewChat}
            className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-primary-600 hover:bg-primary-500 rounded-lg text-sm font-medium transition-colors"
          >
            <Sparkles className="w-4 h-4" />
            新对话
          </button>
        </div>
        <div className="flex-1 overflow-y-auto p-2 space-y-1 scrollbar-thin">
          {sessions.map(session => (
            <div
              key={session.session_id}
              onClick={() => handleSelectSession(session.session_id)}
              className={`
                group flex items-center gap-2 px-3 py-2 rounded-lg cursor-pointer text-sm transition-colors
                ${currentSessionId === session.session_id
                  ? 'bg-slate-800 text-white'
                  : 'text-slate-400 hover:bg-slate-800/50 hover:text-slate-200'
                }
              `}
            >
              <MessageSquare className="w-4 h-4 shrink-0" />
              <div className="flex-1 min-w-0">
                <div className="truncate">{session.title || '新对话'}</div>
                <div className="text-xs text-slate-500">{formatTime(session.updated_at)}</div>
              </div>
              <button
                onClick={(e) => handleDeleteSession(e, session.session_id)}
                className="opacity-0 group-hover:opacity-100 p-1 hover:text-red-400 transition-opacity"
              >
                ×
              </button>
            </div>
          ))}
          {sessions.length === 0 && (
            <div className="text-center text-slate-600 text-sm py-8">暂无会话</div>
          )}
        </div>
      </div>

      {/* 聊天主区域 */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* 消息列表 */}
        <div className="flex-1 overflow-y-auto p-4 space-y-4 scrollbar-thin">
          {messages.length === 0 && !isStreaming && (
            <div className="flex flex-col items-center justify-center h-full text-slate-500">
              <Bot className="w-12 h-12 mb-4 text-slate-600" />
              <h2 className="text-xl font-semibold text-slate-300 mb-2">DevOps Agent</h2>
              <p className="text-sm max-w-md text-center">
                面向国产化环境的运维智能体。您可以询问系统状态、执行诊断命令或请求运维操作。
              </p>
              <div className="mt-6 grid grid-cols-1 sm:grid-cols-2 gap-3 max-w-lg w-full">
                {['查看磁盘使用情况', '分析系统日志', '检查网络连接', '列出高 CPU 进程'].map((demo) => (
                  <button
                    key={demo}
                    onClick={() => { setInput(demo); inputRef.current?.focus() }}
                    className="px-4 py-2.5 bg-slate-800/50 hover:bg-slate-800 rounded-lg text-sm text-left text-slate-300 transition-colors border border-slate-800"
                  >
                    {demo}
                  </button>
                ))}
              </div>
            </div>
          )}

          {messages.map((msg) => (
            <div
              key={msg.id}
              className={`flex gap-3 ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              {msg.role === 'assistant' && (
                <div className="w-8 h-8 rounded-lg bg-primary-900/50 border border-primary-800/50 flex items-center justify-center shrink-0">
                  <Bot className="w-4 h-4 text-primary-400" />
                </div>
              )}
              <div
                className={`
                  max-w-[85%] sm:max-w-[75%] rounded-2xl px-4 py-3 text-sm leading-relaxed whitespace-pre-wrap
                  ${msg.role === 'user'
                    ? 'bg-primary-600 text-white'
                    : msg.isError
                      ? 'bg-red-900/20 border border-red-800/30 text-red-200'
                      : 'bg-slate-800 text-slate-200'
                  }
                `}
              >
                {msg.content}
              </div>
              {msg.role === 'user' && (
                <div className="w-8 h-8 rounded-lg bg-slate-700 flex items-center justify-center shrink-0">
                  <User className="w-4 h-4 text-slate-300" />
                </div>
              )}
            </div>
          ))}

          {/* 流式推理进度 */}
          {isStreaming && (
            <div className="flex gap-3">
              <div className="w-8 h-8 rounded-lg bg-primary-900/50 border border-primary-800/50 flex items-center justify-center shrink-0">
                <Bot className="w-4 h-4 text-primary-400" />
              </div>
              <div className="bg-slate-800 rounded-2xl px-4 py-3 max-w-[85%] sm:max-w-[75%]">
                {/* 推理阶段指示器 */}
                <div className="flex items-center gap-2 mb-3">
                  <Loader2 className="w-4 h-4 animate-spin text-primary-400" />
                  <span className="text-sm text-primary-300 font-medium">
                    {EVENT_LABELS[activeEvent] || '处理中...'}
                  </span>
                </div>

                {/* 推理事件时间线 */}
                <div className="space-y-1.5">
                  {streamEvents.slice(-6).map((evt, idx) => (
                    <div key={evt.time + idx} className="flex items-center gap-2 text-xs">
                      <span className={`font-medium ${EVENT_COLORS[evt.type] || 'text-slate-400'}`}>
                        {EVENT_LABELS[evt.type] || evt.type}
                      </span>
                      {evt.payload?.tool_name && (
                        <span className="text-slate-500 font-mono">{evt.payload.tool_name}</span>
                      )}
                      {evt.payload?.reply_preview && (
                        <span className="text-slate-500 truncate max-w-[200px]">
                          {evt.payload.reply_preview}
                        </span>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* 输入区域 */}
        <div className="border-t border-slate-800 p-4 bg-slate-900/30">
          <form onSubmit={handleSubmit} className="flex gap-3 max-w-4xl mx-auto">
            <input
              ref={inputRef}
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder={isStreaming ? 'Agent 正在思考...' : '输入运维问题或指令...'}
              disabled={isStreaming}
              className="flex-1 bg-slate-800 border border-slate-700 rounded-xl px-4 py-3 text-sm
                         text-slate-100 placeholder-slate-500
                         focus:outline-none focus:ring-2 focus:ring-primary-500/50 focus:border-primary-500/50
                         disabled:opacity-50 disabled:cursor-not-allowed transition-all"
            />
            <button
              type="submit"
              disabled={isStreaming || !input.trim()}
              className="px-5 py-3 bg-primary-600 hover:bg-primary-500 disabled:bg-slate-700 disabled:text-slate-500
                         rounded-xl text-white transition-colors flex items-center gap-2"
            >
              {isStreaming ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <Send className="w-4 h-4" />
              )}
              <span className="hidden sm:inline">发送</span>
            </button>
          </form>
          <div className="max-w-4xl mx-auto mt-2 flex items-center gap-4 text-xs text-slate-600">
            <span className="flex items-center gap-1">
              <Terminal className="w-3 h-3" />
              支持自然语言运维查询
            </span>
            <span className="flex items-center gap-1">
              <AlertTriangle className="w-3 h-3" />
              危险操作需经安全校验
            </span>
          </div>
        </div>
      </div>
    </div>
  )
}
