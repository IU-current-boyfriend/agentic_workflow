import asyncio
from loguru import logger
from typing import Any, List
from llama_index.core.llms import ChatMessage
from llama_index.core.workflow import step, StartEvent, StopEvent, Context, Event
from llama_index.utils.workflow import Workflow, draw_all_possible_flows
from llama_index.core.llms import LLM
from llama_index.core.memory import Memory
from llama_index.core.tools import ToolSelection, ToolOutput, FunctionTool
from llama_index.core.tools.types import BaseTool
from llama_index.llms.openai_like import OpenAILike
from llama_index.core.agent.react import ReActChatFormatter, ReActOutputParser
from llama_index.core.agent.react.types import ActionReasoningStep, ObservationReasoningStep
from rag_rank_workflow.settings import LLM_NAMESPACE,LLM_URL,LLM_NAME,LLM_API_KEY
# definition event
class PrepEvent(Event):
    pass

class InputEvent(Event):
    input: List[ChatMessage]

class StreamEvent(Event):
    delta: str

class ToolCallEvent(Event):
    tool_calls: List[ToolSelection]

class FunctionOutputEvent(Event):
    output: ToolOutput

# definition tools
def add(a: int, b: int) -> int:
    """Useful function to add two numbers."""
    return a + b

def multiply(a: int, b: int) -> int:
    """Useful function to multiply two numbers."""
    return a * b

class ReactWorkflow(Workflow):

    # definition initialize params
    def __init__(self,
                 *args: Any,
                 llm: LLM | None = None,
                 tools: list[BaseTool] | None = None,
                 extra_context:str | None = None,
                 **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # 解释一下参数问题：
        # tools: 提供llm可调用的tools
        self.tools = tools or []
        # llm: 提供的llm client instance
        self.llm = llm or OpenAILike()
        # react 基础的架构,里面包含system prompt之类的内容，我现在重点关心workflow，后面再来看里面的内容
        self.formatter = ReActChatFormatter(
            context = extra_context or ""
        )
        # ReAct output parser instance object
        self.output_parser = ReActOutputParser()

    # workflow startEvent: one step
    @step
    async def new_user_msg(self, ctx: Context, ev: StartEvent) -> PrepEvent:
        """我感觉是把用户新的message填充到chatMessage中"""

        # clear sources
        await ctx.store.set("sources", [])

        # init memory if needed
        memory = await ctx.store.get("memory", default=None)

        # memory is None
        if memory is None:
            memory = Memory.from_defaults(
                token_limit = self.llm.metadata.context_window
            )
        # get user input
        user_input = ev.user_input
        # 构建一个chat message消息对象
        user_message = ChatMessage(
            role = "user",
            content = user_input
        )
        # 存储到memory中
        memory.put(user_message)
        # clear current reasoning
        await ctx.store.set("current_react_result", [])
        # 把包含历史消息对话的内存放到上下文中
        await ctx.store.set("memory", memory)
        # 返回事件
        return PrepEvent()

    # workflow prepEvent: second step
    @step
    async def prepare_chat_history(self, ctx: Context, ev: PrepEvent) -> InputEvent:
        # get chat history
        memory = await ctx.store.get("memory")
        chat_history = memory.get()
        # 当前llm rea act的结果，可能是Thought/action/observer => current_reasoning
        current_reasoning = await ctx.store.get("current_react_result", default = [])
        # format the prompt with react instruction
        llm_chat_message = self.formatter.format(
            tools = self.tools,
            chat_history=chat_history,
            current_reasoning=current_reasoning
        )

        return InputEvent(
            input = llm_chat_message
        )

    # workflow inputEvent: three step
    @step
    async def handle_llm_input(self, ctx: Context, ev: InputEvent) -> ToolCallEvent | StopEvent | PrepEvent:
        # 获取llm的消息对话
        llm_chat_message = ev.input
        # 获取此轮llm react的结果
        current_react_result = await ctx.store.get("current_react_result", default = [])
        # 获取历史对话内存
        memory = await ctx.store.get("memory")
        # 调用llm
        chat_response_async_gen = await self.llm.astream_chat(llm_chat_message)
        # 监听写入llm的response结果
        async for chat_response in chat_response_async_gen:
            # inside write event to stream
            # side receiver event
            ctx.write_event_to_stream(
                # 这里delta是流式的增量
                StreamEvent(delta=chat_response.delta or "")
            )

        # 后续的处理
        try:
            # chat_response.message.content是async for执行后的最后一次响应对象中完整正文
            current_reasoning_step = self.output_parser.parse(chat_response.message.content)
            current_react_result.append(current_reasoning_step)

            # 判断react流程是否完成
            if current_reasoning_step.is_done:
                # react process final
                memory.put(
                    ChatMessage(
                        role="assistant",
                        content=current_reasoning_step.response
                    )
                )
                await ctx.store.set("memory", memory)
                await ctx.store.set("current_react_result", current_react_result)
                # 获取资源
                sources = await ctx.store.get("sources", default=[])
                # 返回事件
                return StopEvent(
                    result={
                        "response": current_reasoning_step.response,
                        "sources": [sources],
                        "reasoning": current_react_result
                    }
                )
            elif isinstance(current_reasoning_step, ActionReasoningStep):
                # 这分支就是react中的 action环节
                # 获取action中涉及的tool name
                tool_name = current_reasoning_step.action
                tool_agrs = current_reasoning_step.action_input
                # 返回工具调用的结果事件
                return ToolCallEvent(
                    tool_calls=[
                        ToolSelection(
                            tool_id="fake",
                            tool_name=tool_name,
                            tool_kwargs=tool_agrs
                        )
                    ]
                )
        except Exception as e:
            current_react_result.append(
                ObservationReasoningStep(
                    observation=f"There was an error in parsing my reasoning: {e}"
                )
            )
            await ctx.store.set("current_react_result", current_react_result)

        # if no tool calls or final response, iterate again
        return PrepEvent()

    # workflow handle tool_call
    @step
    async def handle_tool_calls(self, ctx: Context, ev: ToolCallEvent) -> PrepEvent:
        # 获取调用需要调用的工具列表对象
        tools_call: List[ToolSelection] = ev.tool_calls
        # 获取能够调用的工具名称
        available_tool_names = { tool.metadata.get_name(): tool for tool in self.tools }
        # 获取当前react 结果
        current_react_result = await ctx.store.get("current_react_result", default=[])
        # 获取资源
        sources = await ctx.store.get("sources", default=[])

        # call tools -- safely
        for tool_call in tools_call:
            tool = available_tool_names.get(tool_call.tool_name)
            if not tool:
                # 需要调用的tool不存在
                current_react_result.append(
                    ObservationReasoningStep(
                        observation=f"The tool {tool_call.tool_name} doesn't exist"
                    )
                )
                # 跳过
                continue
            try:
                tool_output = tool(**tool_call.tool_kwargs)
                sources.append(tool_output)
                current_react_result.append(
                    ObservationReasoningStep(
                        observation=tool_output.content
                    )
                )
            except Exception as e:
                current_react_result.append(
                    ObservationReasoningStep(
                        observation=f"Error calling tool {tool.metadata.get_name()}: {e}"
                    )
                )
        # save new state in context
        await ctx.store.set("sources", sources)
        await ctx.store.set("current_react_result", current_react_result)

        # prep the next iteration
        return PrepEvent()



async def workflow_run():
    tools = [
        FunctionTool.from_defaults(add),
        FunctionTool.from_defaults(multiply),
    ]
    react_agent = ReactWorkflow(
        llm=OpenAILike(
            model=LLM_NAMESPACE[LLM_NAME],
            api_key=LLM_NAMESPACE[LLM_API_KEY],
            api_base=LLM_NAMESPACE[LLM_URL],
            is_function_calling_model=True,
            is_chat_model=True
        ),
        tools=tools,
        verbose=True,
        timeout=120
    )
    # ret = await agents.run(user_input="hello! please give me a joke.")

    ret = await react_agent.run(user_input="What is (2123 + 2321) * 312?")

    logger.success(f"ret['response']: {ret['response']}")


async def workflow_run_stream():
   tools = [
       FunctionTool.from_defaults(add),
       FunctionTool.from_defaults(multiply),
   ]
   react_workflow = ReactWorkflow(
       llm=OpenAILike(
           model=LLM_NAMESPACE[LLM_NAME],
           api_key=LLM_NAMESPACE[LLM_API_KEY],
           api_base=LLM_NAMESPACE[LLM_URL],
           is_chat_model=True,
           is_function_calling_model=True
       ),
       tools=tools,
       timeout=120, verbose=False)
   handle = react_workflow.run(user_input="What is (2123 + 2321) * 312?")
   async for event in handle.stream_events():
       if isinstance(event, StreamEvent):
           logger.success(event.delta)

   response = await handle
   logger.success(response)



   # draw_all_possible_flows(ReactWorkflow, filename="react_workflow.html")

if __name__ == "__main__":
    asyncio.run(workflow_run_stream())