import asyncio
from typing import List, Any

from llama_index.llms.openai_like import OpenAILike
from llama_index.core.llms import ChatMessage
from llama_index.core.llms.llm import ToolSelection
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.utils.workflow import Event, StartEvent, StopEvent
from llama_index.core.workflow import step, Context
from llama_index.core.tools import ToolOutput, BaseTool
from llama_index.utils.workflow import Workflow
from llama_index.core.memory import Memory
from agents.settings import LLM_NAMESPACE,LLM_URL,LLM_NAME,LLM_API_KEY


# definition InputEvent
class InputEvent(Event):
    chat_message_history: List[ChatMessage]

# definition StreamEvent
class StreamEvent(Event):
    # 增量
    delta: str

# definition ToolCallEvent
class ToolCallEvent(Event):
    tools_call: List[ToolSelection]

# definition FunctionOutputEvent
class FunctionOutputEvent(Event):
    output: ToolOutput


class FunctionAgentWorkflow(Workflow):

    # definition init
    def __init__(self,
                 *args: Any,
                 llm: FunctionCallingLLM | None = None,
                 tools: List[BaseTool] | None = None,
                 **kwargs) -> None:
        # extends
        super().__init__(*args, **kwargs)
        # initial tools
        self.tools = tools or []
        # initial llm
        self.llm = llm or OpenAILike()
        # 断言llm必须支持function call方式 or tool call
        assert self.llm.metadata.is_function_calling_model, [f"{self.llm} don't support this function call or tool call."]


    # definition step: prepare chat history
    @step
    async def prepare_chat_history(self, ctx: Context, ev:StartEvent) -> InputEvent:
        """ prepare chat history """

        # 清除工具调用的资源列表
        await ctx.store.set("tool_resources", [])
        # 获取内存，用来记录chat_message
        memory = await ctx.store.get("memory", default=None)
        # if memory is None
        if memory is None:
            memory = Memory.from_defaults(
                token_limit=self.llm.metadata.context_window,
            )
        # 获取用户查询的内容
        user_input = ev.user_input
        # 创建用户chat_message内容
        user_msg = ChatMessage(
            role="user",
            content=user_input,
        )
        memory.put(user_msg)
        # 获取chat_message_list
        chat_message_history = memory.get()
        # 保存对应的内容
        await ctx.store.set("memory", memory)
        # 返回对应事件
        return InputEvent(
            chat_message_history=chat_message_history
        )

    @step
    async def handle_llm_input(self, ctx: Context, ev: InputEvent) -> ToolCallEvent | StopEvent:
        """ chat message history and tool call llm """
        # 获取chat_message_history
        chat_message_history = ev.chat_message_history
        # llm stream to response provider side
        response_streams = await self.llm.astream_chat_with_tools(
            tools=self.tools,
            chat_history=chat_message_history,
        )
        # provider side stream event
        async for response_stream in response_streams:
            ctx.write_event_to_stream(StreamEvent(
                delta=response_stream.delta
            ))
        # 从llm响应中解析call_tools
        tools_call = await self.llm.get_tool_calls_from_response(
            response_stream=response_streams,
            # 好像默认会处理是否存在tool可用
            error_on_no_tool_call=False
        )
        # 如果llm决策不存在可调用llm，则直接返回结果
        if not tools_call:
            tool_resources = await ctx.store.set("tool_resources", [])
            return StopEvent(
                result={
                    "response": response_streams,
                    "tool_resources": [*tool_resources]
                }
            )
        else:
            return ToolCallEvent(
                tools_call=tools_call,
            )

    @step
    async def handle_tool_call(self, ctx:Context, ev:ToolCallEvent) -> InputEvent:
        """ tool call and observer tool call result to llm"""
        # 获取解析llm响应后的tools_call
        tools_call = ev.tools_call
        # 获取提供可用的tools名称dict
        available_tools: dict[str, ToolSelection] = { tool.metadata.get_name(): tool for tool in self.tools }
        # 后续需要提供给llm的chat_message 关于tools的结果
        tools_msg = []
        # 获取 tool_resources
        tool_resources = await ctx.store.get("tool_resources", [])
        # call tool
        for tool_call in tools_call:
            # 匹配可用的工具
            match_tool:ToolSelection = available_tools.get(tool_call.tool_name)
            # prepare tool chat message kwargs
            additional_kwargs = {
                "tool_call_id": match_tool.tool_id,
                "name": match_tool.tool_name
            }
            # check match is None
            if match_tool is None:
                tools_msg.append(
                    ChatMessage(
                        role="tool",
                        content=f"Tool {tool_call.tool_name} doesn't exist",
                        additional_kwargs=additional_kwargs
                    )
                )
                continue
            try:
                # 这里的tool_output.content哪里来的
                tool_output: ToolOutput = match_tool(**tool_call.tool_kwargs)
                tool_resources.append(tool_output)
                tools_msg.append(
                    ChatMessage(
                        role="tool",
                        content=tool_output.content,
                        additional_kwargs=additional_kwargs
                    )
                )
            except Exception as e:
                tools_msg.append(
                    ChatMessage(
                        role="tool",
                        content=f"Encountered error in tool call: {e}",
                        additional_kwargs=additional_kwargs
                    )
                )
            # update memory
            memory = await ctx.store.get("memory")
            # add tools call result
            for tool_msg in tools_msg:
                memory.put(tool_msg)
            await ctx.store.set("memory", memory)
            await ctx.store.set("tool_resources", tool_resources)
            # 获取对话历史数据
            chat_message_history = memory.get()

        return InputEvent(
            chat_message_history=chat_message_history
        )

async def function_agent_workflow():
    pass

if __name__ == "__main__":
    asyncio.run(function_agent_workflow())