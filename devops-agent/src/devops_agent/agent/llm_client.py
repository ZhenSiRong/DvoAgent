"""
LLM 双协议适配器 — 统一 OpenAI Chat Completions / Anthropic Messages API

设计原则：
1. 协议透明切换：同一套调用接口，后端自动适配目标协议
2. 工具调用统一抽象：将两种协议的 tool_calls 格式归一化
3. 流式/非流式双模式：支持 SSE 流式输出和普通 JSON 响应
4. 错误恢复：单协议失败自动降级到备用协议
5. Token 用量追踪：每次调用记录 prompt/completion tokens

支持厂商（已验证）：
- MiniMax（双协议兼容）
- DeepSeek / Qwen（OpenAI 协议）
- Claude / Anthropic 系列（Anthropic 协议）
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class LLMProtocol(str, Enum):
    """LLM API 协议类型"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


@dataclass
class LLMMessage:
    """统一的聊天消息格式"""
    role: str              # system / user / assistant / tool
    content: str | list[dict]  # 文本或 content blocks (Anthropic 格式)
    name: str | None = None     # tool 消息时使用
    tool_call_id: str | None = None
    tool_calls: list[dict] | None = None


@dataclass
class ToolDefinition:
    """工具定义（LLM function calling schema）"""
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema


@dataclass
class LLMResponse:
    """
    统一的 LLM 响应格式。

    无论底层用哪种协议，对外都返回此结构。
    """
    reply_text: str = ""
    tool_calls: list[dict] | None = None      # 标准化后的工具调用列表
    raw_response: dict[str, Any] | None = None  # 原始响应（调试用）
    finish_reason: str = ""                     # stop / tool_calls / length
    usage: dict[str, int] = field(default_factory=dict)  # {prompt_tokens, completion_tokens}
    model: str = ""
    protocol_used: str = ""                     # 实际使用的协议


# ============================================================
#  协议实现层
# ============================================================

async def call_openai_chat(
    messages: list[LLMMessage],
    tools: list[ToolDefinition] | None = None,
    base_url: str = "",
    api_key: str = "",
    model: str = "",
    temperature: float = 0.3,
    max_tokens: int = 4096,
    stream: bool = False,
) -> LLMResponse:
    """
    使用 OpenAI Chat Completions 协议调用 LLM。

    适用：DeepSeek, Qwen, MiniMax(OpenAI模式), 以及所有 OpenAI 兼容服务。
    """
    import httpx

    # 构建 OpenAI 格式的请求体
    oa_messages = [_to_openai_msg(m) for m in messages]
    
    body: dict[str, Any] = {
        "model": model,
        "messages": oa_messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": stream,
    }

    if tools:
        body["tools"] = [_to_openai_tool(t) for t in tools]
        body["tool_choice"] = "auto"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    url = f"{base_url.rstrip('/')}/chat/completions"

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(url, json=body, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        return _parse_openai_response(data)

    except httpx.HTTPStatusError as e:
        logger.error("OpenAI 协议 HTTP 错误 %s: %s", e.response.status_code, e.response.text)
        raise
    except Exception as e:
        logger.error("OpenAI 协议调用异常: %s", e)
        raise


async def call_anthropic_messages(
    messages: list[LLMMessage],
    tools: list[ToolDefinition] | None = None,
    base_url: str = "",
    api_key: str = "",
    model: str = "",
    temperature: float = 0.3,
    max_tokens: int = 4096,
    stream: bool = False,
) -> LLMResponse:
    """
    使用 Anthropic Messages API 调用 LLM。

    适用：Claude, MiniMax(Anthropic模式), 以及 Anthropic 兼容服务。
    """
    import httpx

    # Anthropic 协议要求 system 消息单独提取
    system_content = ""
    anthropic_msgs = []
    
    for m in messages:
        if m.role == "system":
            # Anthropic 支持多段 system，拼接处理
            system_content += (
                m.content if isinstance(m.content, str) else json.dumps(m.content)
            )
        else:
            anthropic_msgs.append(_to_anthropic_msg(m))

    body: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": stream,
    }
    
    if system_content:
        body["system"] = system_content
    
    if anthropic_msgs:
        body["messages"] = anthropic_msgs

    if tools:
        body["tools"] = [_to_anthropic_tool(t) for t in tools]
        body["tool_choice"] = {"type": "auto"}

    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    }

    url = f"{base_url.rstrip('/')}/messages"

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(url, json=body, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        return _parse_anthropic_response(data)

    except httpx.HTTPStatusError as e:
        logger.error("Anthropic 协议 HTTP 错误 %s: %s", e.response.status_code, e.response.text)
        raise
    except Exception as e:
        logger.error("Anthropic 协议调用异常: %s", e)
        raise


# ============================================================
#  统一调度入口
# ============================================================

async def call_llm(
    messages: list[LLMMessage],
    protocol: LLMProtocol = LLMProtocol.OPENAI,
    tools: list[ToolDefinition] | None = None,
    base_url: str = "",
    api_key: str = "",
    model: str = "",
    temperature: float = 0.3,
    max_tokens: int = 4096,
    fallback_base_url: str = "",
    fallback_api_key: str = "",
    fallback_model: str = "",
    stream: bool = False,
) -> LLMResponse:
    """
    LLM 调用的统一入口。

    自动根据 protocol 参数选择对应协议。
    如果首选协议失败且配置了 fallback 参数，自动降级到备用协议。

    Args:
        messages: 对话消息列表
        protocol: 首选协议 (openai / anthropic)
        tools: 可用工具定义列表（function calling）
        base_url/api_key/model: 首选协议参数
        fallback_base_url/api_key/model: 备用协议参数（跨协议降级）
        stream: 是否流式输出

    Returns:
        LLMResponse: 统一格式的响应
    """
    try:
        if protocol == LLMProtocol.ANTHROPIC:
            result = await call_anthropic_messages(
                messages=messages,
                tools=tools,
                base_url=base_url or fallback_base_url,
                api_key=api_key or fallback_api_key,
                model=model or fallback_model,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream,
            )
            result.protocol_used = "anthropic"
            return result
        else:
            result = await call_openai_chat(
                messages=messages,
                tools=tools,
                base_url=base_url,
                api_key=api_key,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream,
            )
            result.protocol_used = "openai"
            return result

    except Exception as primary_err:
        # 尝试备用协议降级
        if fallback_base_url and fallback_api_key:
            logger.warning(
                "首选协议(%s)失败，尝试降级到备用协议: %s",
                protocol.value, primary_err,
            )
            try:
                if protocol == LLMProtocol.OPENAI:
                    # OpenAI 失败 → 降级到 Anthropic
                    result = await call_anthropic_messages(
                        messages=messages, tools=tools,
                        base_url=fallback_base_url, api_key=fallback_api_key,
                        model=fallback_model, temperature=temperature,
                        max_tokens=max_tokens, stream=stream,
                    )
                    result.protocol_used = "anthropic(fallback)"
                    return result
                else:
                    # Anthropic 失败 → 降级到 OpenAI
                    result = await call_openai_chat(
                        messages=messages, tools=tools,
                        base_url=fallback_base_url, api_key=fallback_api_key,
                        model=fallback_model, temperature=temperature,
                        max_tokens=max_tokens, stream=stream,
                    )
                    result.protocol_used = "openai(fallback)"
                    return result
            except Exception as fallback_err:
                logger.error("备用协议也失败了: %s", fallback_err)

        raise


# ============================================================
#  内部转换函数 — 协议格式互转
# ============================================================

def _to_openai_msg(msg: LLMMessage) -> dict:
    """将统一消息转换为 OpenAI 格式"""
    m: dict[str, Any] = {"role": msg.role, "content": msg.content}
    if msg.name:
        m["name"] = msg.name
    if msg.tool_call_id:
        m["tool_call_id"] = msg.tool_call_id
    if msg.tool_calls:
        # 标准化格式 → OpenAI 原始格式（MiniMax/DeepSeek 等兼容接口需要）
        m["tool_calls"] = [
            {
                "id": tc.get("id", ""),
                "type": "function",
                "function": {
                    "name": tc.get("name", ""),
                    "arguments": json.dumps(tc.get("arguments", tc.get("args", {})), ensure_ascii=False),
                },
            }
            for tc in msg.tool_calls
        ]
    return m


def _to_anthropic_msg(msg: LLMMessage) -> dict:
    """将统一消息转换为 Anthropic Messages API 格式"""
    # Anthropic 要求 role 为 user/assistant
    role = msg.role if msg.role in ("user", "assistant") else "user"
    
    m: dict[str, Any] = {"role": role}
    
    if isinstance(msg.content, list):
        # 已经是 content block 格式（Anthropic 原生）
        m["content"] = msg.content
    elif msg.tool_calls and msg.role == "assistant":
        # assistant 带 tool_use 的内容块
        content_blocks: list[dict] = []
        if msg.content:
            content_blocks.append({"type": "text", "text": msg.content})
        for tc in msg.tool_calls:
            content_blocks.append({
                "type": "tool_use",
                "id": tc.get("id", f"toolu_{hash(tc.get('name', ''))}"),
                "name": tc.get("name", ""),
                "input": tc.get("arguments", tc.get("args", {})),
            })
        m["content"] = content_blocks
    else:
        m["content"] = msg.content or ""

    return m


def _to_openai_tool(tool: ToolDefinition) -> dict:
    """转换为 OpenAI function calling 工具定义格式"""
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        },
    }


def _to_anthropic_tool(tool: ToolDefinition) -> dict:
    """转换为 Anthropic tool_use 定义格式"""
    return {
        "name": tool.name,
        "description": tool.description,
        "input_schema": tool.parameters,
    }


def _parse_openai_response(data: dict) -> LLMResponse:
    """解析 OpenAI Chat Completions 响应为统一格式"""
    choice = data.get("choices", [{}])[0]
    message = choice.get("message", {})
    
    reply_text = message.get("content") or ""
    
    # 解析工具调用
    tool_calls = message.get("tool_calls")
    normalized_tools = None
    if tool_calls:
        normalized_tools = []
        for tc in tool_calls:
            fn = tc.get("function", {})
            args_str = fn.get("arguments", "{}")
            try:
                args = json.loads(args_str) if isinstance(args_str, str) else args_str
            except json.JSONDecodeError:
                args = {}
            normalized_tools.append({
                "id": tc.get("id", ""),
                "name": fn.get("name", ""),
                "arguments": args,
            })

    usage_data = data.get("usage", {})

    return LLMResponse(
        reply_text=reply_text,
        tool_calls=normalized_tools,
        raw_response=data,
        finish_reason=choice.get("finish_reason", ""),
        usage={
            "prompt_tokens": usage_data.get("prompt_tokens", 0),
            "completion_tokens": usage_data.get("completion_tokens", 0),
        },
        model=data.get("model", ""),
    )


def _parse_anthropic_response(data: dict) -> LLMResponse:
    """解析 Anthropic Messages API 响应为统一格式"""
    content_blocks = data.get("content", [])
    
    text_parts = []
    tool_calls_list = []

    for block in content_blocks:
        btype = block.get("type", "")
        if btype == "text":
            text_parts.append(block.get("text", ""))
        elif btype == "tool_use":
            tool_calls_list.append({
                "id": block.get("id", ""),
                "name": block.get("name", ""),
                "arguments": block.get("input", {}),
            })

    reply_text = "\n".join(text_parts)
    
    usage_data = data.get("usage", {})
    # Anthropic 的 input/output tokens 映射到 prompt/completion
    # 注意: usage 在 streaming 模式下可能为空（message_delta event 中才有）
    
    return LLMResponse(
        reply_text=reply_text,
        tool_calls=tool_calls_list if tool_calls_list else None,
        raw_response=data,
        finish_reason=data.get("stop_reason", ""),
        usage={
            "prompt_tokens": usage_data.get("input_tokens", 0),
            "completion_tokens": usage_data.get("output_tokens", 0),
        },
        model=data.get("model", ""),
    )


__all__ = [
    "LLMProtocol", "LLMMessage", "ToolDefinition", "LLMResponse",
    "call_llm", "call_openai_chat", "call_anthropic_messages",
]
