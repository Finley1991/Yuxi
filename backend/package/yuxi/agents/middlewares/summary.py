"""Yuxi 对 DeepAgents 会话摘要中间件的适配。

整体作用
========

这是 Yuxi 的「上下文自动压缩」核心。当一个对话线程跑了很久、累计的 messages 加上
system prompt 和 tools 超过设定的压力阈值（默认 100K tokens），中间件会：

1. 先做「确定性压缩」：把超长的工具结果完整写入 Workdir 的 outputs/large_tool_results/，
   请求里只保留路径 + 哈希 + 预览；把 write_file/edit_file 的过长 content 参数截断。不调模型。
2. 重新算 token，如果仍超阈值，才做「摘要压缩」：把较早的对话历史写入
   outputs/conversation_history/，调一次摘要模型，用 summary message + 最近 N 条
   原文组成新视图。

关键不变量
---------
- PostgreSQL 的 messages 表不删，压缩只改 LangGraph checkpoint 的视图。
- 完整工具结果写不进 Workdir 时拒绝用裁剪内容替换原 ToolMessage（fail-closed）。
- system prompt 和 tool schemas 每次按当前 Agent 配置重新装配，不存进摘要 event。

入口
----
- 自动压缩：`wrap_model_call` / `awrap_model_call`（每次模型请求前判断）
- 主动压缩：`aforce_summarize`（用户点「压缩上下文」按钮时）
- 工厂：`create_summary_middleware_from_context`（按 Agent context 配置组装）
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import warnings
from collections.abc import Awaitable, Callable, Iterable
from contextvars import ContextVar
from typing import Any

from deepagents.middleware.summarization import (
    Command,
    ContextOverflowError,
    SummarizationMiddleware,
    _aclip_overflow_tail,
    _clip_overflow_tail,
)
from langchain.agents.middleware.summarization import ContextSize
from langchain.agents.middleware.types import ExtendedModelResponse, ModelRequest, ModelResponse
from langchain.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, ToolMessage, get_buffer_string
from langchain_core.messages.utils import count_tokens_approximately
from langgraph.config import get_stream_writer
from langgraph.constants import TAG_NOSTREAM

from yuxi.agents.context import (
    DEFAULT_SUMMARY_KEEP_MESSAGES,
    DEFAULT_SUMMARY_THRESHOLD_K,
    DEFAULT_SUMMARY_TOOL_RESULT_TOKEN_LIMIT,
    DEFAULT_YUXI_SUMMARY_PROMPT,
)
from yuxi.models.chat import load_chat_model, resolve_chat_model_spec
from yuxi.utils.logging_config import logger

# 近似 token 估算的字符比——4 个字符约 1 token。仅用于压力判断和预览长度，不是计费口径。
_APPROX_CHARS_PER_TOKEN = 4

# 默认工具结果落盘阈值：超过 300 tokens 的工具结果完整写入 Workdir，请求只保留预览。
_DEFAULT_SUMMARY_TOOL_RESULT_LIMIT_TOKENS = 300

# write_file / edit_file 的 content 参数在请求视图里截断到 2000 字符。
# 例子：模型写了个 8000 字符的 HTML，history 里的 AIMessage.tool_calls[0].args.content
# 在送给下一轮模型时只看到前 20 个字符 + "...(argument truncated for context view)"。
_DEFAULT_TOOL_ARG_MAX_LENGTH = 2000

_TRUNCATED_TOOL_ARG_TEXT = "...(argument truncated for context view)"

# 标记某条 ToolMessage 已经被落盘替换过，避免下次压缩时重复写文件。
_TOOL_RESULT_SAVED_MARKER = "yuxi_tool_result_saved"

# 这两个工具的返回值是结构化 JSON（检索结果），用专门的预览器保留元数据。
_STRUCTURED_SEARCH_TOOL_NAMES = {"query_kb", "web_search"}
_SEARCH_CONTENT_KEYS = ("content", "text", "snippet", "summary")

# ContextVar 用于在一次模型请求内「只发一次 compression started 事件」。
# 例子：wrap_model_call 入口 set 一个 dict，_offload_to_backend 第一次调用时
# 把 started 置 True，后续再调用就不重复发 started 事件。
_SUMMARY_COMPRESSION_STATE: ContextVar[dict[str, bool] | None] = ContextVar(
    "yuxi_summary_compression_state",
    default=None,
)


class YuxiSummarizationMiddleware(SummarizationMiddleware):
    """先确定性压缩工具结果，再按同一压力阈值决定是否生成摘要。

    为什么要分两阶段：
    - 直接调摘要模型代价高（一次额外 LLM 调用 + 写历史文件 + 更新 checkpoint）。
    - 确定性压缩只裁剪/落盘，不调模型，对「工具结果太大但对话轮数不多」的场景就够了。
    - 所以流程是：达到阈值 → 先确定性压缩 → 重新算 token → 仍超阈值才调摘要模型。
    """

    # 摘要模型调用的配置：lc_source 标记来源让审计区分，TAG_NOSTREAM 不让摘要调用走流式。
    _SUMMARY_INVOKE_CONFIG = {"metadata": {"lc_source": "summarization"}, "tags": [TAG_NOSTREAM]}

    def __init__(
        self,
        *args,
        tool_result_offload_token_limit: int | None = _DEFAULT_SUMMARY_TOOL_RESULT_LIMIT_TOKENS,
        tool_arg_max_length: int = _DEFAULT_TOOL_ARG_MAX_LENGTH,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.tool_result_offload_token_limit = tool_result_offload_token_limit
        self.tool_arg_max_length = tool_arg_max_length

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """同步执行单阈值压缩流程。

        例子（一次模型请求的完整生命周期）：
        - 进来：request.messages 共 120K tokens（超过 100K 阈值）
        - 流程：
          1. _get_effective_messages + _count_tokens → total_tokens = 120K
          2. _should_summarize → True（超阈值）
          3. _compact_messages → 把 web_search 的 50K 结果落盘，请求视图降到 80K
          4. compacted_tokens = 80K < 100K，should_summarize 变 False
          5. handler(80K 视图) → 正常调用主模型，返回
        - 失败路径：如果步骤 5 抛 ContextOverflowError，overflow_triggered=True，继续做摘要
        """
        compression_state: dict[str, bool] = {"started": False}
        compression_token = _SUMMARY_COMPRESSION_STATE.set(compression_state)
        try:
            try:
                result = self._wrap_model_call_with_compaction(request, handler)
            except Exception as exc:
                # 如果压缩已经开始才失败，发 failed 事件让前端知道。
                if compression_state["started"]:
                    _emit_compression("failed", error=repr(exc))
                raise
            self._emit_completed(result)
            return result
        finally:
            _SUMMARY_COMPRESSION_STATE.reset(compression_token)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """异步执行单阈值压缩流程。逻辑和同步版完全一致，只是用 await。"""
        compression_state: dict[str, bool] = {"started": False}
        compression_token = _SUMMARY_COMPRESSION_STATE.set(compression_state)
        try:
            try:
                result = await self._awrap_model_call_with_compaction(request, handler)
            except Exception as exc:
                if compression_state["started"]:
                    _emit_compression("failed", error=repr(exc))
                raise
            self._emit_completed(result)
            return result
        finally:
            _SUMMARY_COMPRESSION_STATE.reset(compression_token)

    async def aforce_summarize(self, state: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        """主动压缩已有 checkpoint，并返回待持久化更新与结果指标。

        用途：用户点「压缩上下文」按钮时调用（不创建 AgentRun，只更新 checkpoint）。
        与 awrap_model_call 的区别：不经过模型请求边界，直接对当前 state 做一次完整压缩。

        例子：
        - 操作前 state：
          messages = [user1, ai1(tool_calls=[web_search]), tool1(50K), ai2, user2, ai3, user3, ai4]
          _summarization_event = None
        - 操作后返回的 update：
          {
            "_summarization_event": {
              "cutoff_index": 3,  # 前 3 条被压缩
              "summary_message": <AIMessage "对话摘要：用户问了X，ai做了Y...">,
              "file_path": "/workdir/outputs/conversation_history/...-abc123.md"
            },
            "_summarization_session_id": "session-xxx"
          }
        - 调用方（context_compression_service）拿 update 去 state_repository.update()，
          checkpoint 就变成 [summary_message, user3, ai4] 了。
        """
        messages = list(state.get("messages") or [])
        previous_event = state.get("_summarization_event")
        # 如果之前压缩过，_summarization_event 里有 cutoff_index，这里要跳过已摘要的区间，
        # 只对 cutoff 之后的原文做新一轮压缩。
        effective_messages = self._apply_event_to_messages(messages, previous_event)
        before_tokens = self._count_tokens(effective_messages, None, [])
        compacted_messages = self._compact_messages(effective_messages)
        cutoff_index = self._determine_cutoff_index(compacted_messages)
        if cutoff_index <= 0:
            # 历史不够长，没什么可压缩的。返回 no_op 让调用方知道不需要更新 checkpoint。
            return {}, {
                "status": "no_op",
                "before_tokens": before_tokens,
                "after_tokens": self._count_tokens(compacted_messages, None, []),
                "reason": "insufficient_history",
            }

        messages_to_summarize, preserved_messages = self._partition_messages(
            compacted_messages,
            cutoff_index,
        )
        offloaded_messages, failed_media = await self._aoffload_inline_media(
            self._backend,
            messages_to_summarize,
        )
        session_id = self._get_session_id(state)
        file_path = await self._aoffload_to_backend(self._backend, offloaded_messages, session_id)
        if file_path is None:
            # 主动压缩必须能恢复历史，落盘失败直接报错（不像自动压缩可以降级）。
            raise RuntimeError("主动压缩无法保存可恢复的对话历史")
        summary = await self._acreate_summary_or_raise(offloaded_messages)
        if failed_media:
            logger.warning(
                "Conversation history offloaded to %s, but %d media block(s) could not be offloaded.",
                file_path,
                failed_media,
            )

        summary_messages = self._build_new_messages_with_path(summary, file_path)
        # 把本次局部 cutoff 换算成完整 state 的位置（叠加之前的 cutoff）。
        state_cutoff_index = self._compute_state_cutoff(previous_event, cutoff_index)
        update = {
            "_summarization_event": {
                "cutoff_index": state_cutoff_index,
                "summary_message": summary_messages[0],
                "file_path": file_path,
            },
            "_summarization_session_id": session_id,
        }
        # 模拟应用 update 后的 messages，算压缩后的 token 数（用于指标上报）。
        persisted_messages = self._apply_event_to_messages(messages, update["_summarization_event"])
        after_tokens = self._count_tokens(persisted_messages, None, [])
        return update, {
            "status": "completed",
            "before_tokens": before_tokens,
            "after_tokens": after_tokens,
            "compressed_messages": cutoff_index,
            "file_path": file_path,
        }

    def _wrap_model_call_with_compaction(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse | ExtendedModelResponse:
        """同步版核心压缩流程。

        完整步骤：
        1. 算 token + 截断过长参数（_truncate_args）
        2. 判断是否达到压力阈值（_should_summarize）
        3. 没达到 → 直接调主模型；达到 → 继续
        4. 确定性压缩（_compact_messages）→ 重新算 token
        5. 压缩后低于阈值 → 调主模型返回；仍超 → 继续
        6. 选 cutoff → 落盘历史 → 调摘要模型 → 构造新视图 → 调主模型 → 返回 Command 更新 state
        """
        effective_messages = self._get_effective_messages(request)
        total_tokens = self._count_tokens(effective_messages, request.system_message, request.tools)
        truncated_messages, _ = self._truncate_args(effective_messages, total_tokens)
        should_compact = self._should_summarize(truncated_messages, total_tokens)

        overflow_triggered = False
        if not should_compact:
            # 没到阈值，直接调主模型。如果主模型自己抛 ContextOverflowError
            # （provider 层面说上下文太长），视为强制摘要信号，继续走压缩流程。
            try:
                return handler(request.override(messages=truncated_messages))
            except ContextOverflowError:
                overflow_triggered = True

        compacted_messages = self._compact_messages(truncated_messages)
        if should_compact:
            _emit_compression_started_once()

        compacted_tokens = self._count_tokens(compacted_messages, request.system_message, request.tools)
        pressure_threshold = self._entry_trigger_tokens()
        # 三个条件任一触发摘要：
        # - overflow_triggered：主模型明确说上下文溢出
        # - pressure_threshold is None：没配置阈值，总是摘要（兜底）
        # - compacted_tokens >= pressure_threshold：确定性压缩后仍超阈值
        should_summarize = overflow_triggered or pressure_threshold is None or compacted_tokens >= pressure_threshold
        if not should_summarize:
            # 确定性压缩就够了，不用调摘要模型。
            try:
                response = handler(request.override(messages=compacted_messages)) # override 是 ModelRequest 的一个方法，返回一个新的 ModelRequest
            except ContextOverflowError:
                overflow_triggered = True
            else:
                _emit_compression("completed")
                return response

        cutoff_index = self._determine_cutoff_index(compacted_messages)
        if cutoff_index <= 0:
            # 没有可摘要的历史（比如 messages 太少），直接调主模型。
            response = handler(request.override(messages=compacted_messages))
            if should_compact:
                _emit_compression("completed")
            return response

        # 切分：前 cutoff_index 条做摘要，后面的保留原文。
        messages_to_summarize, preserved_messages = self._partition_messages(compacted_messages, cutoff_index)
        new_state_tail: list[AnyMessage] = []
        if overflow_triggered:
            # 溢出场景：连 preserved_messages 都可能超模型上限，再裁一次尾部。
            # 例子：max_input=128K，preserved 还有 150K，clip_overflow_tail 会把超大的
            # 工具结果再落盘，只留 head + tail。
            preserved_messages, new_state_tail = _clip_overflow_tail(
                preserved_messages,
                self._backend,
                keep=self._lc_helper.keep,
                max_input_tokens=self._get_profile_limits(),
                token_counter=self.token_counter,
                large_tool_results_prefix=self._large_tool_results_prefix,
            )

        offloaded_messages, failed_media = self._offload_inline_media(self._backend, messages_to_summarize)
        session_id = self._get_session_id(request.state)
        file_path = self._offload_to_backend(self._backend, offloaded_messages, session_id)
        self._report_offload_result(file_path, failed_media)

        summary = self._create_summary(offloaded_messages)
        new_messages = self._build_new_messages_with_path(summary, file_path)
        new_event = self._build_summary_event(request.state, cutoff_index, new_messages[0], file_path)
        # 新视图 = [summary_message, *preserved_messages]，调主模型。
        response = handler(request.override(messages=[*new_messages, *preserved_messages]))
        # 返回 ExtendedModelResponse：主模型响应 + 一个 Command(update=...) 让 LangGraph
        # 把 _summarization_event 写进 checkpoint state，后续请求会跳过已摘要区间。
        return ExtendedModelResponse(
            model_response=response,
            command=Command(update=self._build_state_update(new_event, session_id, new_state_tail)),
        )

    async def _awrap_model_call_with_compaction(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse | ExtendedModelResponse:
        """异步版核心压缩流程。逻辑和同步版完全一致，唯一差异：落盘和摘要用 asyncio.gather 并行。

        为什么异步版能并行：_aoffload_to_backend（写历史文件）和 _acreate_summary（调摘要模型）
        之间没有依赖——摘要模型读的是 messages_to_summarize（内存里的），不需要等文件写完。
        所以可以并行。同步版没法并行，只能串行。
        """
        effective_messages = self._get_effective_messages(request)
        total_tokens = self._count_tokens(effective_messages, request.system_message, request.tools)
        truncated_messages, _ = self._truncate_args(effective_messages, total_tokens)
        should_compact = self._should_summarize(truncated_messages, total_tokens)

        overflow_triggered = False
        if not should_compact:
            try:
                return await handler(request.override(messages=truncated_messages))
            except ContextOverflowError:
                overflow_triggered = True

        compacted_messages = self._compact_messages(truncated_messages)
        if should_compact:
            _emit_compression_started_once()

        compacted_tokens = self._count_tokens(compacted_messages, request.system_message, request.tools)
        pressure_threshold = self._entry_trigger_tokens()
        should_summarize = overflow_triggered or pressure_threshold is None or compacted_tokens >= pressure_threshold
        if not should_summarize:
            try:
                response = await handler(request.override(messages=compacted_messages))
            except ContextOverflowError:
                overflow_triggered = True
            else:
                _emit_compression("completed")
                return response

        cutoff_index = self._determine_cutoff_index(compacted_messages)
        if cutoff_index <= 0:
            response = await handler(request.override(messages=compacted_messages))
            if should_compact:
                _emit_compression("completed")
            return response

        messages_to_summarize, preserved_messages = self._partition_messages(compacted_messages, cutoff_index)
        new_state_tail: list[AnyMessage] = []
        if overflow_triggered:
            preserved_messages, new_state_tail = await _aclip_overflow_tail(
                preserved_messages,
                self._backend,
                keep=self._lc_helper.keep,
                max_input_tokens=self._get_profile_limits(),
                token_counter=self.token_counter,
                large_tool_results_prefix=self._large_tool_results_prefix,
            )

        offloaded_messages, failed_media = await self._aoffload_inline_media(
            self._backend,
            messages_to_summarize,
        )
        session_id = self._get_session_id(request.state)
        # 并行：一边写历史文件，一边调摘要模型。
        file_path, summary = await asyncio.gather(
            self._aoffload_to_backend(self._backend, offloaded_messages, session_id),
            self._acreate_summary(offloaded_messages),
        )
        self._report_offload_result(file_path, failed_media)

        new_messages = self._build_new_messages_with_path(summary, file_path)
        new_event = self._build_summary_event(request.state, cutoff_index, new_messages[0], file_path)
        response = await handler(request.override(messages=[*new_messages, *preserved_messages]))
        return ExtendedModelResponse(
            model_response=response,
            command=Command(update=self._build_state_update(new_event, session_id, new_state_tail)),
        )

    def _compact_messages(self, messages: list[AnyMessage]) -> list[AnyMessage]:
        """压缩过大的工具结果及文件写入参数，保持消息顺序和标识不变。

        只做两件事，不调模型：
        1. AIMessage 的 write_file/edit_file 参数过长 → 截断到 tool_arg_max_length
        2. ToolMessage 超过 tool_result_offload_token_limit → 完整写入 Workdir，请求只留预览

        例子（操作前/后对比）：

        操作前 messages：
          [
            HumanMessage("帮我写个 8000 字的报告"),
            AIMessage(tool_calls=[{name:"write_file", args:{path:"/report.md", content:"<8000字>"}}]),
            ToolMessage(content="File written to /report.md"),
            HumanMessage("再搜一下竞品"),
            AIMessage(tool_calls=[{name:"web_search", args:{query:"竞品"}}]),
            ToolMessage(content="<50000字的搜索结果JSON>"),  # 这条超大
          ]

        操作后 messages（送给主模型的视图）：
          [
            HumanMessage("帮我写个 8000 字的报告"),
            AIMessage(tool_calls=[{name:"write_file", args:{path:"/report.md",
                content:"<前20字>...(argument truncated for context view)"}}]),  # 参数截断
            ToolMessage(content="File written to /report.md"),
            HumanMessage("再搜一下竞品"),
            AIMessage(tool_calls=[{name:"web_search", args:{query:"竞品"}}]),
            ToolMessage(content="[Tool result saved]\nTool: web_search\nApprox tokens: 12500\n
                SHA-256: abc...\nFull output path: /workdir/outputs/large_tool_results/web_search-abc123.txt\n\n
                Output preview:\n{\"kind\":\"web_search\",\"result_count\":8,...}"),  # 落盘+预览
          ]

        为什么这样做：
        - 模型不需要看到自己写过的完整文件内容（文件在 Workdir，模型可以 read_file 取）。
        - 工具的完整结果保存在文件里，模型按需读取，避免每轮都把 50K 结果塞进上下文。
        - 消息 id 和顺序不变，保证 LangGraph checkpoint 的 reducer 能正确合并。
        """
        compacted: list[AnyMessage] = []
        modified = False
        for message in messages:
            updated = message
            if isinstance(message, AIMessage):
                updated = _truncate_ai_tool_call_args(message, max_length=self.tool_arg_max_length)
            elif (
                isinstance(message, ToolMessage)
                and getattr(message, "additional_kwargs", {}).get(_TOOL_RESULT_SAVED_MARKER) is not True
                and _should_offload_tool_message(message, self.tool_result_offload_token_limit)
            ):
                # 已标记 _TOOL_RESULT_SAVED_MARKER 的不重复处理（避免一次请求内多次压缩重复写文件）。
                updated = _replace_tool_message_content(
                    message,
                    backend=self._backend,
                    tool_result_token_limit=self.tool_result_offload_token_limit,
                    large_tool_results_prefix=self._large_tool_results_prefix,
                )
            compacted.append(updated)
            modified = modified or updated is not message
        # 没改动就返回原 list（避免无谓的复制，让上游 reference 比较生效）。
        return compacted if modified else messages

    def _should_summarize(self, messages: list[AnyMessage], total_tokens: int) -> bool:
        """判断是否达到压力阈值——任一 trigger clause 满足就触发。

        clause 有三种 kind：
        - "messages": 消息数达到 N 条
        - "tokens": token 数达到 N
        - "fraction": token 数达到 max_input_tokens 的某比例

        例子：trigger=("tokens", 102400) → clause = {"tokens": 102400}
          total_tokens = 120000 >= 102400 → True
        """
        if not self._lc_helper._trigger_clauses:
            return False
        for clause in self._lc_helper._trigger_clauses:
            if self._trigger_clause_met(clause, messages, total_tokens):
                return True
        return False

    def _trigger_clause_met(self, clause: dict[str, Any], messages: list[AnyMessage], total_tokens: int) -> bool:
        """单个 clause 是否满足——所有 kind 都满足才算满足（AND 关系）。"""
        for kind, value in clause.items():
            if kind == "messages" and len(messages) < value:
                return False
            if kind == "tokens" and total_tokens < value:
                return False
            if kind == "fraction":
                max_input_tokens = self._get_profile_limits()
                if max_input_tokens is None or total_tokens < max(int(max_input_tokens * value), 1):
                    return False
        return True

    def _entry_trigger_tokens(self) -> int | None:
        """从所有 clause 里算出最小的 token 阈值。

        用于「确定性压缩后是否还要摘要」的二次判断。
        例子：clause 里有 tokens=102400 和 fraction=0.8（max_input=200000 → 160000），
        取 min = 102400，压缩后 compacted_tokens >= 102400 才摘要。
        """
        thresholds: list[int] = []
        for clause in self._lc_helper._trigger_clauses or []:
            token_threshold = clause.get("tokens")
            if isinstance(token_threshold, int) and token_threshold > 0:
                thresholds.append(token_threshold)
            fraction = clause.get("fraction")
            if isinstance(fraction, int | float) and (max_input_tokens := self._get_profile_limits()) is not None:
                thresholds.append(max(int(max_input_tokens * fraction), 1))
        return min(thresholds) if thresholds else None

    def _build_summary_prompt(self, messages: list[AnyMessage]) -> str | None:
        """构造摘要模型的 prompt——用 summary_prompt 模板 + 裁剪后的历史。"""
        trimmed = self._lc_helper._trim_messages_for_summary(messages)
        if not trimmed:
            return None
        return self._lc_helper.summary_prompt.format(messages=get_buffer_string(trimmed, format="xml")).rstrip()

    def _create_summary(self, messages: list[AnyMessage]) -> str:
        """同步调摘要模型。失败不抛异常，返回错误文本（避免摘要失败影响主流程）。"""
        if not messages:
            return "No previous conversation history."
        prompt = self._build_summary_prompt(messages)
        if prompt is None:
            return "Previous conversation was too long to summarize."
        try:
            return self.model.invoke(prompt, config=self._SUMMARY_INVOKE_CONFIG).text.strip()
        except Exception as exc:
            return f"Error generating summary: {exc!s}"

    async def _acreate_summary(self, messages: list[AnyMessage]) -> str:
        """异步调摘要模型。失败不抛异常，返回错误文本。"""
        if not messages:
            return "No previous conversation history."
        prompt = self._build_summary_prompt(messages)
        if prompt is None:
            return "Previous conversation was too long to summarize."
        try:
            response = await self.model.ainvoke(prompt, config=self._SUMMARY_INVOKE_CONFIG)
            return response.text.strip()
        except Exception as exc:
            return f"Error generating summary: {exc!s}"

    async def _acreate_summary_or_raise(self, messages: list[AnyMessage]) -> str:
        """主动压缩专用的摘要调用——失败必须抛异常。

        区别于 _acreate_summary：自动压缩失败可以降级（返回错误文本，主模型还能继续），
        主动压缩失败必须让用户知道（按钮点了但没成功）。
        """
        prompt = self._build_summary_prompt(messages) if messages else None
        if prompt is None:
            raise RuntimeError("没有可供主动压缩的对话历史")
        response = await self.model.ainvoke(prompt, config=self._SUMMARY_INVOKE_CONFIG)
        summary = response.text.strip()
        if not summary:
            raise RuntimeError("摘要模型返回空内容")
        return summary

    def _offload_to_backend(self, backend, messages: list[AnyMessage], session_id: str) -> str | None:
        """同步版：把待摘要的历史消息写入 Workdir 的 conversation_history/。"""
        _emit_compression_started_once()
        return super()._offload_to_backend(backend, messages, session_id)

    async def _aoffload_to_backend(self, backend, messages: list[AnyMessage], session_id: str) -> str | None:
        """异步版：把待摘要的历史消息写入 Workdir 的 conversation_history/。"""
        _emit_compression_started_once()
        return await super()._aoffload_to_backend(backend, messages, session_id)

    def _build_summary_event(
        self,
        state: dict[str, Any],
        cutoff_index: int,
        summary_message: AnyMessage,
        file_path: str | None,
    ) -> dict[str, Any]:
        """构造写入 checkpoint 的 _summarization_event。

        例子：
          state["_summarization_event"] = None  # 第一次压缩
          cutoff_index = 3  # 本次局部 cutoff
          返回：
            {
              "cutoff_index": 3,  # 叠加后还是 3
              "summary_message": <AIMessage "对话摘要...">,
              "file_path": "/workdir/outputs/conversation_history/conv-abc.md"
            }

        第二次压缩时 previous_event 已存在，cutoff_index 会叠加：
          previous cutoff = 3, 本次局部 cutoff = 5
          返回 cutoff_index = 8（前 8 条都被压缩过）
        """
        return {
            "cutoff_index": self._compute_state_cutoff(state.get("_summarization_event"), cutoff_index),
            "summary_message": summary_message,
            "file_path": file_path,
        }

    @staticmethod
    def _build_state_update(
        event: dict[str, Any],
        session_id: str,
        new_state_tail: list[AnyMessage],
    ) -> dict[str, Any]:
        """构造 Command(update=...) 的 payload，让 LangGraph 写进 checkpoint。

        例子：
          event = {"cutoff_index": 3, "summary_message": ..., "file_path": ...}
          new_state_tail = []  # 没溢出时为空
          返回：
            {
              "_summarization_event": event,
              "_summarization_session_id": "session-xxx"
            }
          如果 new_state_tail 非空（溢出场景），还会加 "messages": [...] 让 reducer 合并。
        """
        update: dict[str, Any] = {
            "_summarization_event": event,
            "_summarization_session_id": session_id,
        }
        if new_state_tail:
            update["messages"] = list(new_state_tail)
        return update

    @staticmethod
    def _report_offload_result(file_path: str | None, failed_media: int) -> None:
        """落盘结果的日志/警告：完全失败 → error；部分媒体失败 → warning。"""
        if file_path is None:
            message = (
                "Offloading conversation history to backend failed during summarization. "
                "Older messages will not be recoverable."
            )
            logger.error(message)
            warnings.warn(message, stacklevel=3)
        elif failed_media:
            logger.warning(
                "Conversation history offloaded to %s, but %d media block(s) could not be offloaded.",
                file_path,
                failed_media,
            )

    @staticmethod
    def _summarization_event_from_result(result: Any) -> dict[str, Any] | None:
        """从 ExtendedModelResponse 里提取 _summarization_event（如果有）。"""
        if not isinstance(result, ExtendedModelResponse):
            return None
        update = getattr(getattr(result, "command", None), "update", None)
        event = update.get("_summarization_event") if isinstance(update, dict) else None
        return event if isinstance(event, dict) else None

    def _emit_completed(self, result: Any) -> None:
        """压缩成功后发 completed 事件给前端（带 cutoff_index 和 file_path）。"""
        event = self._summarization_event_from_result(result)
        if event is not None:
            _emit_compression(
                "completed",
                cutoff_index=event.get("cutoff_index"),
                file_path=event.get("file_path"),
            )


def create_summary_middleware(
    model: str | BaseChatModel,
    *,
    backend,
    trigger: ContextSize | list[ContextSize] | None,
    keep: ContextSize | list[ContextSize] | None,
    summary_prompt: str | None = None,
    trim_tokens_to_summarize: int | None = None,
    tool_result_offload_token_limit: int | None = _DEFAULT_SUMMARY_TOOL_RESULT_LIMIT_TOKENS,
) -> YuxiSummarizationMiddleware:
    """创建绑定单次运行 backend 的摘要中间件。

    例子：
      create_summary_middleware(
          model="doubao:glm-5-3-flash",
          backend=sandbox_backend,
          trigger=("tokens", 102400),  # 100K tokens 触发
          keep=("messages", 10),       # 保留最近 10 条原文
          summary_prompt="你是上下文压缩助手...",
      )
    """
    middleware_kwargs = {
        "model": model,
        "backend": backend,
        "trigger": trigger,
        "keep": keep,
        "token_counter": _count_tokens_for_summary_trigger,
        "trim_tokens_to_summarize": trim_tokens_to_summarize,
        "tool_result_offload_token_limit": tool_result_offload_token_limit,
    }
    if summary_prompt and summary_prompt.strip():
        middleware_kwargs["summary_prompt"] = summary_prompt
    return YuxiSummarizationMiddleware(**middleware_kwargs)


def create_summary_middleware_from_context(context, *, backend) -> YuxiSummarizationMiddleware:
    """按 Agent 运行时配置创建自动与主动压缩共用的摘要器。

    从 context 读：
    - summary_threshold（默认 100K）→ trigger tokens = 100 * 1024
    - model → 摘要模型
    - summary_keep_messages（默认 10）→ keep
    - summary_prompt（默认 DEFAULT_YUXI_SUMMARY_PROMPT）
    - summary_tool_result_token_limit（默认 300）→ 工具结果落盘阈值

    自动压缩和主动压缩复用同一个摘要器实例，配置一致。
    """
    trigger_tokens = getattr(context, "summary_threshold", DEFAULT_SUMMARY_THRESHOLD_K) * 1024
    model_spec = resolve_chat_model_spec(context.model)
    return create_summary_middleware(
        model=load_chat_model(fully_specified_name=model_spec, session_id=context.thread_id),
        backend=backend,
        trigger=("tokens", trigger_tokens),
        keep=("messages", getattr(context, "summary_keep_messages", DEFAULT_SUMMARY_KEEP_MESSAGES)),
        summary_prompt=getattr(context, "summary_prompt", None) or DEFAULT_YUXI_SUMMARY_PROMPT,
        trim_tokens_to_summarize=trigger_tokens,
        tool_result_offload_token_limit=getattr(
            context,
            "summary_tool_result_token_limit",
            DEFAULT_SUMMARY_TOOL_RESULT_TOKEN_LIMIT,
        ),
    )


def _emit_compression(status: str, **extra: Any) -> None:
    """通过 LangGraph stream writer 发 custom event 给前端。

    事件类型 yuxi.context_compression，status 取值：
    - started：压缩开始
    - completed：压缩完成（带 cutoff_index, file_path）
    - failed：压缩失败（带 error）

    前端监听这个事件来更新状态面板的「压缩中...」提示。
    如果不在 LangGraph 执行上下文里（比如主动压缩在 canonical graph 外调用），
    get_stream_writer 会抛 RuntimeError，这里 swallow 掉。
    """
    try:
        writer = get_stream_writer()
    except RuntimeError:
        return
    writer({"type": "yuxi.context_compression", "status": status, **extra})


def _emit_compression_started_once() -> None:
    """一次模型请求内只发一次 started 事件。

    场景：_offload_to_backend 可能被调用多次（比如同步版串行调用），但 started 事件
    只应该在第一次发出。用 _SUMMARY_COMPRESSION_STATE ContextVar 跟踪。
    """
    state = _SUMMARY_COMPRESSION_STATE.get()
    if state is not None and state["started"]:
        return
    if state is not None:
        state["started"] = True
    _emit_compression("started")


def _count_tokens_for_summary_trigger(messages: Iterable[Any], **kwargs: Any) -> None:
    """近似 token 计数器，用于压力判断。

    为什么用近似（count_tokens_approximately）而不是精确 tokenize：
    - 压力阈值只需要粗略判断（超没超 100K），不需要精确到个位。
    - 精确 tokenize 对长上下文很慢（几十毫秒），近似算法（4字符≈1token）几乎零开销。
    - 计费口径走主模型返回的 usage_metadata，不依赖这里。
    """
    kwargs.pop("use_usage_metadata_scaling", None)
    return count_tokens_approximately(messages, use_usage_metadata_scaling=False, **kwargs)


def _build_tool_result_preview(
    content: str,
    token_limit: int | None,
    *,
    tool_name: str | None = None,
    raw_content: Any = None,
) -> tuple[str, int]:
    """在单条工具预算内生成结构化检索或通用首中尾预览。

    返回 (preview_text, omitted_chars)：
    - omitted_chars > 0 表示有内容被省略（会附加提示「Read the full output from saved file」）
    - omitted_chars = 0 表示没有省略（content 本身就在预算内）

    例子（web_search 工具结果）：
      content = '{"results": [{title:"A", content:"..."}, ...共 50 条]}'  # 50000 字符
      token_limit = 300 → max_chars = 1200

      走 _structured_search_preview：
      - 解析 JSON，取前 8 条结果的 title/url/score（不要 content 全文）
      - 每条结果分配少量 content_preview
      - 返回 JSON 字符串，约 1100 字符
      - omitted_chars = 50000 - 1100 = 48900

    例子（普通 execute 工具结果）：
      content = "ls 命令输出 5000 行..."  # 20000 字符
      token_limit = 300 → max_chars = 1200

      走 _generic_tool_result_preview：
      - 头部 480 字符 + [HEAD]
      - 中间 240 字符 + [MIDDLE]
      - 尾部 480 字符 + [TAIL]
      - omitted_chars = 20000 - 1200 = 18800
    """
    text = content.strip()
    if token_limit is None:
        return text, 0
    if token_limit <= 0:
        return "", len(text)

    max_chars = token_limit * _APPROX_CHARS_PER_TOKEN
    if len(text) <= max_chars:
        return text, 0

    # 优先用结构化预览器（针对 query_kb / web_search）
    structured_preview = _structured_search_preview(tool_name, raw_content, max_chars)
    if structured_preview is not None:
        return structured_preview, max(len(text) - len(structured_preview), 0)

    # 通用 head + middle + tail 预览
    preview = _generic_tool_result_preview(text, max_chars)
    return preview, len(text) - len(preview)


def _extract_text_content(content: Any) -> str:
    """从 ToolMessage.content 里提取纯文本。

    content 可能是 str、list[dict]（多模态）、None 等。
    例子：[{"type":"text", "text":"hello"}, {"type":"image", ...}] → "hello"
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            item if isinstance(item, str) else item["text"]
            for item in content
            if isinstance(item, str) or isinstance(item, dict) and isinstance(item.get("text"), str)
        ]
        return "\n".join(part for part in parts if part)
    return "" if content is None else str(content)


def _tool_result_path(tool_name: str | None, content: str, prefix: str) -> str:
    """生成工具结果落盘的文件路径：{prefix}/{tool_name}-{sha256前16位}.txt

    例子：
      tool_name = "web_search"
      content = '{"results": [...]}'
      prefix = "/workdir/outputs/large_tool_results"
      → "/workdir/outputs/large_tool_results/web_search-abc123def4567890.txt"

    为什么用内容哈希：同一内容只写一次文件，重复工具结果不会产生多个文件。
    """
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "-", (tool_name or "").strip()).strip(".-") or "tool-result"
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}/{safe_name}-{digest}.txt"


def _write_tool_result(backend, path: str, content: str) -> str:
    """把工具结果写入 backend（Workdir）。失败抛异常（fail-closed）。

    为什么 fail-closed：如果写不进文件但用裁剪内容替换原 ToolMessage，模型会丢失
    完整工具结果且无法恢复。宁可让整个压缩流程失败，也不要丢数据。
    """
    if backend is None:
        raise RuntimeError(f"Cannot save tool result to {path}: backend is unavailable")
    result = backend.write(path, content)
    error = getattr(result, "error", None)
    if not error or "already exists" in str(error).lower():
        return path
    raise RuntimeError(f"Failed to write tool result to {path}: {error}")


def _should_offload_tool_message(message: ToolMessage, token_limit: int | None) -> bool:
    """判断某条 ToolMessage 是否需要落盘。

    例子：token_limit=300，content 50000 字符
      estimated_tokens = ceil(50000 / 4) = 12500 > 300 → True（需要落盘）
    """
    if token_limit is None or token_limit <= 0:
        return True
    content = _extract_text_content(message.content)
    estimated_tokens = max((len(content) + _APPROX_CHARS_PER_TOKEN - 1) // _APPROX_CHARS_PER_TOKEN, 1)
    return estimated_tokens > token_limit


def _replace_tool_message_content(
    message: ToolMessage,
    *,
    backend,
    tool_result_token_limit: int | None,
    large_tool_results_prefix: str,
) -> ToolMessage:
    """把一条超大的 ToolMessage 替换为「落盘路径 + 预览」格式。

    操作前 ToolMessage.content：
      '{"results": [{title:"A", content:"...很长的正文..."}, ...共 50 条]}'  # 50000 字符

    操作后 ToolMessage.content：
      [Tool result saved]
      Tool: web_search
      Approx tokens: 12500
      SHA-256: abc123def4567890...
      Full output path: /workdir/outputs/large_tool_results/web_search-abc123def4567890.txt

      Output preview:
      {"kind":"web_search","result_count":50,"results":[{"title":"A","url":"...","score":0.9,"content_preview":"..."}, ...]}

      [Truncated 48900 chars. Read the full output from the saved file.]

    同时给 additional_kwargs 加 _TOOL_RESULT_SAVED_MARKER=True，防止下次压缩重复处理。

    为什么这样设计：
    - 模型还能看到工具结果的元数据（有几条、标题是什么），不会完全失忆。
    - 需要完整内容时模型可以 read_file(full_output_path) 主动取。
    - 同一内容哈希相同，多次调用不会产生重复文件。
    """
    content = _extract_text_content(message.content)
    tool_name = message.name if isinstance(message.name, str) and message.name else None
    path = _write_tool_result(
        backend,
        _tool_result_path(tool_name, content, large_tool_results_prefix),
        content,
    )
    preview, omitted_chars = _build_tool_result_preview(
        content,
        tool_result_token_limit,
        tool_name=tool_name,
        raw_content=message.content,
    )
    approx_tokens = max((len(content) + _APPROX_CHARS_PER_TOKEN - 1) // _APPROX_CHARS_PER_TOKEN, 1)
    lines = [
        "[Tool result saved]",
        f"Tool: {tool_name or 'unknown'}",
        f"Approx tokens: {approx_tokens}",
        f"SHA-256: {hashlib.sha256(content.encode('utf-8')).hexdigest()}",
        f"Full output path: {path}",
    ]
    if preview:
        lines.extend(["", "Output preview:", preview])
    if omitted_chars:
        lines.append(f"[Truncated {omitted_chars} chars. Read the full output from the saved file.]")

    additional_kwargs = dict(getattr(message, "additional_kwargs", {}) or {})
    additional_kwargs[_TOOL_RESULT_SAVED_MARKER] = True
    return message.model_copy(update={"content": "\n".join(lines), "additional_kwargs": additional_kwargs})


def _structured_search_preview(tool_name: str | None, raw_content: Any, max_chars: int) -> str | None:
    """针对 query_kb / web_search 的结构化预览器。

    与通用 head/middle/tail 不同，检索结果保留元数据（title/url/score）比保留正文片段更有用。
    例子：50 条搜索结果，预算 1200 字符
      - 取前 8 条
      - 每条只保留 title/url/score（不要 content 全文）
      - 如果还超预算，逐步减少 content_preview 长度，最后甚至去掉 content_preview
      - 如果连 metadata 都装不下，只保留 {"kind":"web_search","result_count":50}
    """
    if tool_name not in _STRUCTURED_SEARCH_TOOL_NAMES or max_chars <= 0:
        return None
    parsed = _parse_structured_tool_result(raw_content)
    if parsed is None:
        return None

    results = parsed["results"]
    preview = _search_header(tool_name, parsed, len(results))
    selected = [_search_result_record(tool_name, result) for result in results[:8]]
    # 逐步删除结果条目直到能装下预算
    while selected and len(_encode_search_preview(preview, selected, len(results))) > max_chars:
        selected.pop()

    if not selected:
        # 连一条结果都装不下，退化到只保留 count
        compact = _encode_search_preview(preview, [], len(results))
        minimal = _encode_search_preview(
            {
                "kind": "knowledge_base" if tool_name == "query_kb" else "web_search",
                "result_count": len(results),
            },
            [],
            len(results),
        )
        if len(compact) <= max_chars:
            return compact
        return minimal if len(minimal) <= max_chars else None

    base_text = _encode_search_preview(preview, selected, len(results))
    # 剩余预算分给每条结果的 content_preview
    per_result_chars = max(max_chars - len(base_text) - 24 * len(selected), 0) // len(selected)
    for record, body in selected:
        if body and per_result_chars >= 24:
            record["content_preview"] = _clip_search_content(body, per_result_chars)

    encoded = _encode_search_preview(preview, selected, len(results))
    # 如果还超预算，逐步缩短 content_preview
    while len(encoded) > max_chars and any("content_preview" in record for record, _ in selected):
        for record, _body in selected:
            content_preview = record.get("content_preview")
            if isinstance(content_preview, str):
                shortened = content_preview[: max(len(content_preview) - 16, 0)]
                if shortened:
                    record["content_preview"] = shortened
                else:
                    record.pop("content_preview")
        encoded = _encode_search_preview(preview, selected, len(results))
    return encoded


def _parse_structured_tool_result(content: Any) -> dict[str, Any] | None:
    """把工具结果解析成 {"results": [...]} 格式。解析失败返回 None（走通用预览器）。"""
    value = content
    if isinstance(content, list):
        text_parts = [item["text"] for item in content if isinstance(item, dict) and isinstance(item.get("text"), str)]
        if any(isinstance(item, dict) and ("type" in item or "text" in item) for item in content):
            if len(text_parts) != 1:
                return None
            value = text_parts[0]
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError):
            return None
    if isinstance(value, list) and all(isinstance(item, dict) for item in value):
        value = {"results": value}
    if not isinstance(value, dict) or not isinstance(value.get("results"), list):
        return None
    return value


def _search_header(tool_name: str, parsed: dict[str, Any], result_count: int) -> dict[str, Any]:
    """构造检索预览的头部元数据。"""
    preview = {
        "kind": "knowledge_base" if tool_name == "query_kb" else "web_search",
        "result_count": result_count,
    }
    for key in ("kb_id", "query", "response_time", "error"):
        if parsed.get(key) is not None:
            preview[key] = _bounded_search_scalar(parsed[key])
    return preview


def _search_result_record(tool_name: str, result: Any) -> tuple[dict[str, Any], str]:
    """从一条检索结果里提取要保留的字段（按工具类型选不同字段集）。

    query_kb 保留：id/kb_id/file_id/title/source/score/distance + metadata 子集
    web_search 保留：title/url/site_name/publish_time/score
    同时返回 content 正文（用于 content_preview 截取）。
    """
    if not isinstance(result, dict):
        return {"value": _bounded_search_scalar(str(result))}, ""

    keys = (
        ("id", "kb_id", "file_id", "title", "source", "score", "distance")
        if tool_name == "query_kb"
        else ("title", "url", "site_name", "publish_time", "score")
    )
    record = {key: _bounded_search_scalar(result[key]) for key in keys if result.get(key) is not None}
    if tool_name == "query_kb" and isinstance(result.get("metadata"), dict):
        metadata = result["metadata"]
        metadata_keys = (
            "source",
            "filename",
            "title",
            "chunk_index",
            "score",
            "rerank_score",
            "hybrid_score",
            "graph_score",
            "distance",
        )
        selected_metadata = {
            key: _bounded_search_scalar(metadata[key]) for key in metadata_keys if metadata.get(key) is not None
        }
        if selected_metadata:
            record["metadata"] = selected_metadata
    body = next((result[key] for key in _SEARCH_CONTENT_KEYS if isinstance(result.get(key), str)), "")
    return record, body


def _encode_search_preview(
    preview: dict[str, Any],
    selected: list[tuple[dict[str, Any], str]],
    result_count: int,
) -> str:
    """把预览 dict 编码成紧凑 JSON 字符串。"""
    encoded = {**preview, "results": [record for record, _body in selected]}
    omitted_results = result_count - len(selected)
    if omitted_results:
        encoded["omitted_results"] = omitted_results
    return json.dumps(encoded, ensure_ascii=False, separators=(",", ":"))


def _bounded_search_scalar(value: Any, max_chars: int = 240) -> Any:
    """标量值超过 max_chars 就裁剪（避免单条 title 过长挤爆预算）。"""
    return _clip_search_content(value, max_chars) if isinstance(value, str) else value


def _clip_search_content(value: str, max_chars: int) -> str:
    """裁剪字符串到 max_chars，长的用 head + … + tail 形式。

    例子：max_chars=100, value="12345...67890"（200 字符）
      head_length = (99 * 3) // 4 = 74
      tail_length = 99 - 74 = 25
      → value[:74] + "…" + value[-25:]
    """
    if max_chars <= 0:
        return ""
    if len(value) <= max_chars:
        return value
    if max_chars <= 24:
        return value[:max_chars]
    marker = "…"
    head_length = ((max_chars - len(marker)) * 3) // 4
    tail_length = max_chars - len(marker) - head_length
    return f"{value[:head_length]}{marker}{value[-tail_length:]}"


def _generic_tool_result_preview(text: str, max_chars: int) -> str:
    """通用工具结果预览：head + middle + tail 三段。

    例子：text 5000 字符，max_chars 1200
      labels 总长 = 10+12+9 = 31，content_budget = 1169
      head = 467, middle = 233, tail = 469
      输出：
        [HEAD]
        <前 467 字符>
        [MIDDLE]
        <中间 233 字符>
        [TAIL]
        <后 469 字符>

    为什么用 head/middle/tail 而不是只 head：
    - 很多工具输出的关键信息在头部（命令执行状态）和尾部（错误/结果摘要）。
    - 中间保留一小段是为了让模型知道「中间还有内容，不是被截断的连续文本」。
    """
    labels = ("[HEAD]\n", "\n\n[MIDDLE]\n", "\n\n[TAIL]\n")
    content_budget = max_chars - sum(len(label) for label in labels)
    if content_budget <= 0:
        return text[:max_chars]

    # 分配比例：head 40%, middle 20%, tail 40%
    head_length = (content_budget * 2) // 5
    middle_length = content_budget // 5
    tail_length = content_budget - head_length - middle_length
    middle_start = max((len(text) - middle_length) // 2, head_length)
    return "".join(
        (
            labels[0],
            text[:head_length],
            labels[1],
            text[middle_start : middle_start + middle_length],
            labels[2],
            text[-tail_length:],
        )
    )


def _truncate_ai_tool_call_args(message: AIMessage, *, max_length: int) -> AIMessage:
    """截断 AIMessage 里 write_file / edit_file 的过长 content 参数。

    操作前 AIMessage.tool_calls：
      [{"name": "write_file", "args": {"path": "/report.md", "content": "<8000字>"}}]

    操作后 AIMessage.tool_calls：
      [{"name": "write_file", "args": {"path": "/report.md",
        "content": "<前20字>...(argument truncated for context view)"}}]

    同时处理 additional_kwargs.tool_calls（provider 原始格式，用于审计/重放）。

    为什么只截断 write_file/edit_file：
    - 这两个工具的 content 参数本身就是文件内容，模型已经写进 Workdir 了，历史轮次不需要再看全文。
    - 其他工具的参数（如 web_search 的 query）本身就短，不需要截断。
    """
    if not message.tool_calls and not getattr(message, "additional_kwargs", None):
        return message

    updated_tool_calls = []
    tool_calls_modified = False
    for tool_call in message.tool_calls or []:
        if not isinstance(tool_call, dict):
            updated_tool_calls.append(tool_call)
            continue
        updated, modified = _truncate_tool_call_args(tool_call, max_length)
        updated_tool_calls.append(updated)
        tool_calls_modified = tool_calls_modified or modified

    additional_kwargs, provider_calls_modified = _truncate_provider_tool_calls(
        dict(getattr(message, "additional_kwargs", {}) or {}),
        max_length,
    )
    if not tool_calls_modified and not provider_calls_modified:
        return message

    updated_message = message.model_copy(update={"additional_kwargs": additional_kwargs})
    if tool_calls_modified:
        updated_message.tool_calls = updated_tool_calls
    return updated_message


def _truncate_tool_call_args(tool_call: dict[str, Any], max_length: int) -> tuple[dict[str, Any], bool]:
    """截断单个 tool_call 的 args（只对 write_file/edit_file 生效）。"""
    args = tool_call.get("args")
    if tool_call.get("name") not in {"write_file", "edit_file"} or not isinstance(args, dict):
        return tool_call, False

    truncated_args = {
        key: _truncate_string_arg(value, max_length) if isinstance(value, str) else value for key, value in args.items()
    }
    if truncated_args == args:
        return tool_call, False
    return {**tool_call, "args": truncated_args}, True


def _truncate_provider_tool_calls(
    additional_kwargs: dict[str, Any],
    max_length: int,
) -> tuple[dict[str, Any], bool]:
    """截断 provider 原始格式的 tool_calls（additional_kwargs.tool_calls[].function.arguments）。

    provider 原始格式是 JSON 字符串（OpenAI wire format），和上面的 dict 格式并存。
    两处都要截断，否则审计/重放会读到完整内容，压力判断不准。
    """
    raw_tool_calls = additional_kwargs.get("tool_calls")
    if not isinstance(raw_tool_calls, list):
        return additional_kwargs, False

    updated_tool_calls = []
    modified = False
    for raw_call in raw_tool_calls:
        function = raw_call.get("function") if isinstance(raw_call, dict) else None
        arguments = function.get("arguments") if isinstance(function, dict) else None
        if (
            not isinstance(function, dict)
            or function.get("name") not in {"write_file", "edit_file"}
            or not isinstance(arguments, str)
            or len(arguments) <= max_length
        ):
            updated_tool_calls.append(raw_call)
            continue
        updated_tool_calls.append(
            {**raw_call, "function": {**function, "arguments": _truncate_string_arg(arguments, max_length)}}
        )
        modified = True

    if not modified:
        return additional_kwargs, False
    return {**additional_kwargs, "tool_calls": updated_tool_calls}, True


def _truncate_string_arg(value: str, max_length: int) -> str:
    """截断字符串参数：保留前 20 字符 + 截断标记。

    例子：value = "<8000字>"，max_length = 2000
      → "<前20字>...(argument truncated for context view)"

    为什么只保留前 20 字符而不是 max_length：
    - 足够让模型识别「这是哪个文件/什么操作」。
    - 保留更多没意义——模型需要完整内容时会 read_file，不会从历史 tool_call 里拼。
    - 省更多 token。
    """
    if len(value) <= max_length:
        return value
    return f"{value[:20]}{_TRUNCATED_TOOL_ARG_TEXT}"
