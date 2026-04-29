"""
Agent 核心引擎 — DevOps Agent 的大脑

从 nanoclaw-py (ApeCodeAI) 的 agent.py 改造而来：
- 删除：send_message（Telegram 特有）、schedule_task/list_tasks/pause_task/resume_task/cancel_task
- 新增：内置工具定义（探针 + 执行器）→ 替代 MCP 外部工具
- 新增：安全拦截器插入推理链路
- 新增：双协议 LLM 调用（OpenAI / Anthropic）
- 保留：会话管理、消息历史、session_id 持久化

核心流程（Tool-Use Loop）：
1. 用户输入 → 构建 messages 列表
2. 调用 LLM（含系统 Prompt + 工具描述）
3. LLM 返回文本回复 或 tool_calls
4. 如果有 tool_calls：
   a. 安全拦截器检查每个调用
   b. 执行对应工具（探针/执行器）
   c. 将结果回传给 LLM
   d. 回到步骤 3，直到 LLM 返回纯文本
5. 返回最终回复
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from ..config import get_settings, get_llm_runtime_config, LLMRuntimeConfig
from ..db import (
    create_session,
    get_session,
    append_message,
    get_session_messages_all,
    touch_session,
)
from ..db import (
    append_reasoning_entry,
)
from ..tools import get_tool_definitions as _registry_get_tool_definitions
from ..tools import dispatch_tool as _registry_dispatch_tool
from ..memory import get_memory_manager
from .llm_client import (
    LLMMessage,
    ToolDefinition,
    LLMResponse,
    call_llm,
    LLMProtocol,
)

logger = logging.getLogger(__name__)

# ============================================================
#  配置常量
# ============================================================

# 最大工具调用轮次（防止无限循环）
MAX_TOOL_ROUNDS = 10


@dataclass
class AgentContext:
    """单次对话的运行时上下文"""
    session_id: str = ""
    user_id: str = "default"
    tool_round: int = 0              # 当前工具调用轮次
    execution_count: int = 0         # 命令执行次数
    probe_call_count: int = 0        # 探针调用次数
    total_llm_tokens: int = 0        # 累计 token 用量
    start_time: float = field(default_factory=time.monotonic)
    reasoning_chain: list[dict] = field(default_factory=list)  # 推理链路日志


# ============================================================
#  工具定义 — LLM 可调用的能力清单
# ============================================================

def get_tool_definitions() -> list[ToolDefinition]:
    """
    返回所有可用工具的定义列表。

    从 ToolRegistry 动态获取，支持插件化扩展。
    新增工具只需注册到 registry，无需修改此函数。
    """
    return _registry_get_tool_definitions()


# ============================================================
#  工具执行调度
# ============================================================

async def dispatch_tool_call(
    tool_name: str,
    arguments: dict[str, Any],
    ctx: AgentContext,
) -> dict[str, Any]:
    """
    分发工具调用到对应的实现函数。

    通过 ToolRegistry 动态调度，所有工具调用统一入口：
    - 记录调用日志
    - 统计调用轮次
    - 统一错误处理格式

    实际执行委托给 ToolRegistry，支持插件化扩展。
    """
    ctx.tool_round += 1
    logger.info("工具调用 #%d: %s(%s)", ctx.tool_round, tool_name, arguments)

    try:
        return await _registry_dispatch_tool(tool_name, arguments, ctx)
    except Exception as e:
        logger.error("工具 %s 执行异常: %s", tool_name, e, exc_info=True)
        return {
            "error": f"工具执行异常: {type(e).__name__}: {e}",
            "is_error": True,
        }





# ============================================================
#  系统提示词构建
# ============================================================

def build_system_prompt() -> str:
    """构建 Agent 的系统 Prompt——角色、能力、规则、约束"""
    tools_info = ""
    for t in get_tool_definitions():
        params_desc = ", ".join(
            f"{k}: {v.get('description', v.get('type', ''))}"
            for k, v in t.parameters.get("properties", {}).items()
        )
        tools_info += f"- **{t.name}**({params_desc}): {t.description}\n"

    return f"""你是 DevOps Agent，一个面向国产化 Linux 环境（龙芯 loongarch64 + 麒麟高级服务器版 V11）的运维智能体。

## 你的身份
你是一个专业的 Linux 运维助手，能够通过自然语言理解用户的运维需求，
然后自动调用系统工具收集信息、分析问题、执行操作。

## 你可以使用的工具

### 📊 只读探针（随时可用）
{tools_info}

### 🔧 执行命令（受安全约束）
- **execute_command(command, timeout)**: 执行运维命令
  - 受安全校验器和白名单双重保护
  - 以 devops-runner 最小权限用户运行
  - 危险命令（rm -rf /, > /etc/passwd 等）会被拦截
  - 超时 30 秒自动终止

## 工作方式
1. **先观察再行动**：用户提出需求后，先用探针了解当前系统状态
2. **分析后建议**：根据收集到的信息给出分析和建议
3. **确认后执行**：需要执行操作时，告知用户将执行什么命令
4. **报告结果**：执行完成后报告结果

## ⚠️ 安全铁律
1. 绝不删除或修改系统关键配置文件（/etc/* 关键文件）
2. 所有命令必须通过 execute_command 工具执行（禁止直接模拟 shell）
3. 不确定的操作应主动告知用户风险并请求确认
4. 不要尝试提权到 root 用户
5. 敏感信息（密码、密钥、token）在回复中脱敏处理

## 输出要求
- 使用中文回答
- 技术信息保持准确（精确数值、完整命令、确切错误信息）
- 结构化输出（关键信息用列表或表格呈现）
- 操作前给出预期影响评估"""


# ============================================================
#  核心 Agent 循环
# ============================================================

async def run_agent(
    user_input: str,
    session_id: str | None = None,
    history: list[dict] | None = None,
    stream: bool = False,
) -> tuple[str, AgentContext]:
    """
    Agent 主入口 — Tool-Use 推理循环。

    这是整个系统的核心。接收用户自然语言输入，经过多轮 LLM 推理
    和工具调用后，返回最终回复。

    流程：
    1. 构建上下文（system prompt + 历史消息 + 当前输入）
    2. 调用 LLM
    3. 如果 LLM 返回 tool_calls → 执行工具 → 将结果回传 → 回到 2
    4. 如果 LLM 返回纯文本 → 结束循环，返回回复

    Args:
        user_input: 用户的消息
        session_id: 会话 ID（None 则新建）
        history: 历史消息（从 DB 加载，用于续接对话）
        stream: 是否流式输出（预留，Day5 完善）

    Returns:
        (reply_text, agent_context): 最终回复文本和运行时上下文
    """
    llm_cfg = await get_llm_runtime_config()

    # ---- 初始化上下文 ----
    sid = session_id or f"sess_{int(time.time() * 1000)}"
    ctx = AgentContext(session_id=sid)

    # ---- 构建 LLM 消息列表 ----
    system_prompt = build_system_prompt()

    # 注入相关记忆（跨会话长期记忆）
    try:
        mm = get_memory_manager()
        memory_text = await mm.get_memory_text_for_prompt(query=user_input)
        if memory_text:
            system_prompt += f"\n\n## 已知信息（来自历史会话）\n{memory_text}\n"
    except Exception as e:
        logger.warning("记忆注入失败（非阻塞）: %s", e)

    messages = [LLMMessage(role="system", content=system_prompt)]

    # 注入历史消息
    if history:
        for msg in history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "tool":
                # 工具结果消息
                messages.append(LLMMessage(
                    role="tool",
                    content=json.dumps(content, ensure_ascii=False),
                    name=msg.get("name", ""),
                    tool_call_id=msg.get("tool_call_id", ""),
                ))
            else:
                messages.append(LLMMessage(role=role, content=content))

    # ---- 提示词注入检测（Phase 2 安全层） ----
    from ..safety.prompt_injection import scan_input
    injection_result = scan_input(user_input)
    if injection_result.is_blocked:
        await append_reasoning_entry(
            session_id=sid,
            round_number=ctx.tool_round + 1,
            stage="SENSE",
            content="【提示词注入拦截】" + json.dumps(injection_result.to_dict(), ensure_ascii=False),
        )
        block_msg = (
            "⚠️ 检测到安全风险，输入已被安全层拦截。\n"
            f"最高风险等级：{injection_result.highest_severity.value}\n"
            f"匹配攻击模式数：{injection_result.match_count}\n"
            f"建议：{injection_result.recommendations[0] if injection_result.recommendations else '请使用安全的运维查询语言'}"
        )
        return block_msg, ctx

    # 添加当前用户输入
    messages.append(LLMMessage(role="user", content=user_input))

    # ---- 五段式日志：SENSE 阶段 ----
    await append_reasoning_entry(
        session_id=sid,
        round_number=ctx.tool_round + 1,
        stage="SENSE",
        content=json.dumps({"user_input": user_input, "input_length": len(user_input)}, ensure_ascii=False),
        metadata={"timestamp": time.time()},
    )

    # ---- Tool-Use Loop ----
    tools_defs = get_tool_definitions()

    while ctx.tool_round < MAX_TOOL_ROUNDS:
        # 根据 protocol 切换主/备参数
        if llm_cfg.protocol == "anthropic":
            base_url = llm_cfg.anthropic_base_url
            api_key = llm_cfg.anthropic_api_key
            model = llm_cfg.anthropic_model
            fallback_base_url = llm_cfg.base_url
            fallback_api_key = llm_cfg.api_key
            fallback_model = llm_cfg.model
        else:
            base_url = llm_cfg.base_url
            api_key = llm_cfg.api_key
            model = llm_cfg.model
            fallback_base_url = llm_cfg.anthropic_base_url
            fallback_api_key = llm_cfg.anthropic_api_key
            fallback_model = llm_cfg.anthropic_model

        # 调用 LLM
        response: LLMResponse = await call_llm(
            messages=messages,
            protocol=LLMProtocol(llm_cfg.protocol),
            tools=tools_defs,
            base_url=base_url,
            api_key=api_key,
            model=model,
            temperature=llm_cfg.temperature,
            max_tokens=llm_cfg.max_tokens,
            fallback_base_url=fallback_base_url,
            fallback_api_key=fallback_api_key,
            fallback_model=fallback_model,
        )

        # 记录 token 用量
        ctx.total_llm_tokens += sum(response.usage.values())

        # ---- 五段式日志：ANALYZE 阶段（LLM 推理过程）----
        await append_reasoning_entry(
            session_id=sid,
            round_number=ctx.tool_round + 1,
            stage="ANALYZE",
            content=json.dumps({
                "has_tool_calls": bool(response.tool_calls),
                "finish_reason": response.finish_reason,
                "protocol": response.protocol_used,
                "reply_preview": (response.reply_text or "")[:200],
                "usage": response.usage,
            }, ensure_ascii=False, default=str),
        )

        # ---- 五段式日志：PLAN 阶段（如果有工具调用）----
        if response.tool_calls:
            await append_reasoning_entry(
                session_id=sid,
                round_number=ctx.tool_round + 1,
                stage="PLAN",
                content=json.dumps({
                    "tool_count": len(response.tool_calls),
                    "tool_calls": [
                        {"name": tc.get("name", ""), "args": tc.get("arguments", tc.get("args", {}))}
                        for tc in response.tool_calls
                    ],
                }, ensure_ascii=False, default=str),
            )

        # 记录推理链路（旧版兼容，保留）
        ctx.reasoning_chain.append({
            "round": ctx.tool_round,
            "has_tool_calls": bool(response.tool_calls),
            "finish_reason": response.finish_reason,
            "protocol": response.protocol_used,
            "tokens_used": response.usage,
        })

        # ---- 无工具调用 → 直接返回回复 ----
        if not response.tool_calls:
            elapsed = time.monotonic() - ctx.start_time
            reply_text = response.reply_text or "（无回复）"

            # ---- 五段式日志：OUTPUT 阶段 ----
            await append_reasoning_entry(
                session_id=sid,
                round_number=ctx.tool_round + 1,
                stage="OUTPUT",
                content=json.dumps({
                    "reply_preview": reply_text[:300],
                    "total_tool_rounds": ctx.tool_round,
                    "total_executions": ctx.execution_count,
                    "total_probe_calls": ctx.probe_call_count,
                    "total_tokens": ctx.total_llm_tokens,
                    "elapsed_seconds": round(elapsed, 2),
                }, ensure_ascii=False),
            )

            logger.info(
                "会话 %s 完成: %d 轮工具调用, %d 次执行, %.1fs",
                sid, ctx.tool_round, ctx.execution_count, elapsed,
            )

            # ---- 自动提取并保存记忆（非阻塞） ----
            try:
                mm = get_memory_manager()
                await mm.extract_and_save_from_session(
                    session_id=sid,
                    user_input=user_input,
                    reply=reply_text,
                    ctx_info={
                        "tool_rounds": ctx.tool_round,
                        "executions": ctx.execution_count,
                        "probe_calls": ctx.probe_call_count,
                    },
                )
            except Exception as mem_err:
                logger.warning("记忆提取失败（非阻塞）: %s", mem_err)

            return reply_text, ctx

        # ---- 有工具调用 → 逐个执行 ----
        logger.info(
            "LLM 返回 %d 个工具调用", len(response.tool_calls),
        )

        # 将 assistant 的回复（含 tool_calls）加入消息历史
        assistant_msg = LLMMessage(
            role="assistant",
            content=response.reply_text,
            tool_calls=response.tool_calls,
        )
        messages.append(assistant_msg)

        for tc in response.tool_calls:
            tool_name = tc.get("name", "")
            args = tc.get("arguments", tc.get("args", {}))
            tool_id = tc.get("id", "")

            logger.info("执行工具: %s(%s)", tool_name, args)

            # 执行工具
            tool_result = await dispatch_tool_call(tool_name, args, ctx)

            # ---- 五段式日志：EXECUTE 阶段 ----
            exec_content = json.dumps({
                "tool_name": tool_name,
                "arguments": args,
                "result_preview": str(tool_result)[:500],
                "is_error": tool_result.get("is_error", False) if isinstance(tool_result, dict) else False,
            }, ensure_ascii=False, default=str)
            await append_reasoning_entry(
                session_id=sid,
                round_number=ctx.tool_round + 1,
                stage="EXECUTE",
                content=exec_content,
            )

            # 将工具结果加入消息历史
            messages.append(LLMMessage(
                role="tool",
                content=json.dumps(tool_result, ensure_ascii=False, default=str),
                name=tool_name,
                tool_call_id=tool_id,
            ))

    # 超过最大轮次
    logger.warning(
        "会话 %s 达到最大工具调用轮次(%d)，强制结束",
        sid, MAX_TOOL_ROUNDS,
    )
    return (
        "抱歉，本次请求涉及的操作步骤过多，已自动中止。"
        "请尝试简化您的需求或将复杂任务拆分为多个步骤。",
        ctx,
    )


# ============================================================
#  SSE 流式推理 —— 推理链路事件流
# ============================================================

async def run_agent_stream(
    user_input: str,
    session_id: str | None = None,
    history: list[dict] | None = None,
) -> Any:
    """
    Agent 流式推理入口 —— 产生推理链路事件（SSE 格式）。

    与 run_agent() 逻辑一致，但改为异步生成器，
    在 tool-use loop 的每个关键阶段 yield 事件，
    让前端实时看到 Agent 的工作进度。

    事件格式：
        {"event": "start", "session_id": "..."}
        {"event": "sense", "status": "ok"}
        {"event": "analyze", "has_tool_calls": true, ...}
        {"event": "plan", "tools": [...]}
        {"event": "execute", "tool_name": "..."}
        {"event": "execute_done", "tool_name": "...", "result": ...}
        {"event": "output", "reply": "...", "metrics": {...}}
        {"event": "done", "session_id": "..."}
    """
    llm_cfg = await get_llm_runtime_config()
    sid = session_id or f"sess_{int(time.time() * 1000)}"
    ctx = AgentContext(session_id=sid)

    # ---- start 事件 ----
    yield {
        "event": "start",
        "session_id": sid,
        "timestamp": time.time(),
    }

    # ---- 构建消息列表 ----
    system_prompt = build_system_prompt()
    try:
        mm = get_memory_manager()
        memory_text = await mm.get_memory_text_for_prompt(query=user_input)
        if memory_text:
            system_prompt += f"\n\n## 已知信息（来自历史会话）\n{memory_text}\n"
    except Exception as e:
        logger.warning("记忆注入失败（非阻塞）: %s", e)

    messages = [LLMMessage(role="system", content=system_prompt)]

    if history:
        for msg in history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "tool":
                messages.append(LLMMessage(
                    role="tool",
                    content=json.dumps(content, ensure_ascii=False),
                    name=msg.get("name", ""),
                    tool_call_id=msg.get("tool_call_id", ""),
                ))
            else:
                messages.append(LLMMessage(role=role, content=content))

    # ---- 提示词注入检测 ----
    from ..safety.prompt_injection import scan_input
    injection_result = scan_input(user_input)
    if injection_result.is_blocked:
        yield {
            "event": "sense",
            "status": "blocked",
            "injection_checked": True,
            "highest_severity": injection_result.highest_severity.value,
            "match_count": injection_result.match_count,
        }
        block_msg = (
            "检测到安全风险，输入已被安全层拦截。"
            f"最高风险等级：{injection_result.highest_severity.value}"
        )
        yield {"event": "output", "reply": block_msg, "metrics": {}}
        yield {"event": "done", "session_id": sid}
        return

    yield {
        "event": "sense",
        "status": "ok",
        "injection_checked": True,
    }

    # 添加当前用户输入
    messages.append(LLMMessage(role="user", content=user_input))

    # ---- Tool-Use Loop ----
    tools_defs = get_tool_definitions()

    while ctx.tool_round < MAX_TOOL_ROUNDS:
        # 根据 protocol 切换主/备参数
        if llm_cfg.protocol == "anthropic":
            base_url = llm_cfg.anthropic_base_url
            api_key = llm_cfg.anthropic_api_key
            model = llm_cfg.anthropic_model
            fallback_base_url = llm_cfg.base_url
            fallback_api_key = llm_cfg.api_key
            fallback_model = llm_cfg.model
        else:
            base_url = llm_cfg.base_url
            api_key = llm_cfg.api_key
            model = llm_cfg.model
            fallback_base_url = llm_cfg.anthropic_base_url
            fallback_api_key = llm_cfg.anthropic_api_key
            fallback_model = llm_cfg.anthropic_model

        # 调用 LLM
        response: LLMResponse = await call_llm(
            messages=messages,
            protocol=LLMProtocol(llm_cfg.protocol),
            tools=tools_defs,
            base_url=base_url,
            api_key=api_key,
            model=model,
            temperature=llm_cfg.temperature,
            max_tokens=llm_cfg.max_tokens,
            fallback_base_url=fallback_base_url,
            fallback_api_key=fallback_api_key,
            fallback_model=fallback_model,
        )

        ctx.total_llm_tokens += sum(response.usage.values())

        # ---- analyze 事件 ----
        yield {
            "event": "analyze",
            "has_tool_calls": bool(response.tool_calls),
            "finish_reason": response.finish_reason,
            "protocol": response.protocol_used,
            "reply_preview": (response.reply_text or "")[:200],
            "usage": response.usage,
        }

        # ---- plan 事件（如果有工具调用） ----
        if response.tool_calls:
            yield {
                "event": "plan",
                "tool_count": len(response.tool_calls),
                "tools": [
                    {"name": tc.get("name", ""), "args": tc.get("arguments", tc.get("args", {}))}
                    for tc in response.tool_calls
                ],
            }

        # 无工具调用 → 直接返回
        if not response.tool_calls:
            elapsed = time.monotonic() - ctx.start_time
            reply_text = response.reply_text or "（无回复）"

            yield {
                "event": "output",
                "reply": reply_text,
                "metrics": {
                    "total_tool_rounds": ctx.tool_round,
                    "total_executions": ctx.execution_count,
                    "total_probe_calls": ctx.probe_call_count,
                    "total_tokens": ctx.total_llm_tokens,
                    "elapsed_seconds": round(elapsed, 2),
                },
            }

            # 自动提取记忆
            try:
                mm = get_memory_manager()
                await mm.extract_and_save_from_session(
                    session_id=sid,
                    user_input=user_input,
                    reply=reply_text,
                    ctx_info={
                        "tool_rounds": ctx.tool_round,
                        "executions": ctx.execution_count,
                        "probe_calls": ctx.probe_call_count,
                    },
                )
            except Exception as mem_err:
                logger.warning("记忆提取失败（非阻塞）: %s", mem_err)

            yield {"event": "done", "session_id": sid}
            return

        # 有工具调用 → 逐个执行
        assistant_msg = LLMMessage(
            role="assistant",
            content=response.reply_text,
            tool_calls=response.tool_calls,
        )
        messages.append(assistant_msg)

        for tc in response.tool_calls:
            tool_name = tc.get("name", "")
            args = tc.get("arguments", tc.get("args", {}))
            tool_id = tc.get("id", "")

            # execute 事件（开始）
            yield {
                "event": "execute",
                "tool_name": tool_name,
                "arguments": args,
            }

            tool_result = await dispatch_tool_call(tool_name, args, ctx)

            # execute_done 事件（完成）
            yield {
                "event": "execute_done",
                "tool_name": tool_name,
                "result_preview": str(tool_result)[:300],
                "is_error": tool_result.get("is_error", False) if isinstance(tool_result, dict) else False,
            }

            messages.append(LLMMessage(
                role="tool",
                content=json.dumps(tool_result, ensure_ascii=False, default=str),
                name=tool_name,
                tool_call_id=tool_id,
            ))

    # 超过最大轮次
    yield {
        "event": "output",
        "reply": "抱歉，本次请求涉及的操作步骤过多，已自动中止。请尝试简化您的需求。",
        "metrics": {},
    }
    yield {"event": "done", "session_id": sid}


# ============================================================
#  会话持久化（DB 版）
# ============================================================


async def save_session_history(session_id: str, messages: list[dict]) -> None:
    """
    保存会话消息历史到数据库。

    对每条消息执行 upsert 语义：通过 session_id + role + content 前 100 字符
    做幂等判断，避免重复写入。如果会话不存在则自动创建。
    """
    # 确保会话存在
    existing = await get_session(session_id)
    if not existing:
        await create_session(title="新对话", session_id=session_id)

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        tool_calls = msg.get("tool_calls")
        await append_message(
            session_id=session_id,
            role=role,
            content=content,
            tool_calls=tool_calls if isinstance(tool_calls, list) else None,
        )

    # 更新会话时间戳
    await touch_session(session_id)


async def load_session_history(session_id: str) -> list[dict] | None:
    """从数据库加载会话消息历史，返回 LLM 格式的 dict 列表。"""
    messages = await get_session_messages_all(session_id)
    if not messages:
        return None

    return [
        {
            "role": m.role,
            "content": m.content,
            "tool_calls": m.tool_calls or None,
        }
        for m in messages
    ]


def clear_session(session_id: str) -> None:
    """清除内存缓存中的会话（兼容接口，DB 中用 delete_session）"""


__all__ = [
    "AgentContext",
    "get_tool_definitions",
    "build_system_prompt",
    "dispatch_tool_call",
    "run_agent",
    "save_session_history",
    "load_session_history",
    "clear_session",
]
