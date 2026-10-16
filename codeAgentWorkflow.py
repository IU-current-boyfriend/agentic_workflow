import ast
import asyncio
import inspect
import io
import contextlib
import re
import traceback
from llama_index.llms.openai_like import OpenAILike
from typing import Tuple, Any, Dict, List, Callable
from llama_index.utils.workflow import Workflow
from llama_index.core.llms.llm import LLM
from llama_index.core.workflow import Event, StartEvent, StopEvent, Context, step
from llama_index.core.llms import ChatMessage
from llama_index.core.memory import Memory
from agents.settings import LLM_NAMESPACE,LLM_URL,LLM_NAME,LLM_API_KEY


# Define a few helper functions
def add(a: int, b: int) -> int:
    """Add two numbers together"""
    return a + b


def subtract(a: int, b: int) -> int:
    """Subtract two numbers"""
    return a - b


def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b


def divide(a: int, b: int) -> float:
    """Divide two numbers"""
    return a / b


class SimpleCodeExecutor:
    """
    A simple code executor that runs Python code with state persistence.

    This executor maintains a global and local state between executions,
    allowing for variables to persist across multiple code runs.

    NOTE: not safe for production use! Use with caution.
    """

    def __init__(self, locals: Dict[str, Any], globals: Dict[str, Any]) -> Tuple[bool, str, Any]:
        """
        Initialize the executor.
        Args:
          locals: Local variables to use in the execution context
          globals: Global variables to use in the execution context
        """
        # State that persists between executions
        self.locals = locals
        self.globals = globals

    def execute(self, code: str) -> Tuple[bool, str, Any]:
        """
        Execute Python code and capture output and return values.
        Args:
            code: Python code to execute
        Returns:
            Dict with keys `success`, `output`, and `return_value`
        """

        # capture stdout and stderr
        stdout = io.StringIO()
        stderr = io.StringIO()
        # definition output
        try:
            # Execute with captured output 改变输出的对象
            with (contextlib.redirect_stdout(stdout),
                  contextlib.redirect_stderr(stderr)):
            # Try to detect if there's a return value (last expression)
                try:
                    tree = ast.parse(code)
                    # 什么意思呢？把code转为ast tree, 然后ast.Expr表示表达式类型，
                    # 什么是表达式？expression，需要计算的那种，例如print()
                    # 本质上: 求值 → 产出结果
                    last_node = tree.body[-1] if tree.body else None
                    # if the last statement is an expression, capture its value
                    if isinstance(last_node, ast.Expr):
                        # split code add return value assignment
                        last_line = code.rstrip().split("\n")[-1]
                        # exec_code
                        exec_code = (
                            code[: -len(last_line) - 1]
                            + "\n__result__="
                            + last_line
                        )
                        # Execute modified code
                        exec(exec_code, self.globals, self.locals)
                        return_value = self.locals.get("__result__")
                    else:
                        # Normal execution
                        exec(code, self.globals, self.locals)
                    # continue...
                except Exception as e:
                    # If parsing fails, just execute the code as is
                    # exec result save system memory capture
                    exec(code, self.globals, self.locals)
            # Get output
            output = stdout.getvalue()
            if stderr.getvalue():
                output += "\n" + stderr.getvalue()
        except Exception as e:
            # Capture exception information
            output = f"Error: {type(e).__name__}: {str(e)}\n"
            output += traceback.format_exc()
        if return_value is not None:
            output += "\n\n" + str(return_value)

        return output

# definition step Event
class InputEvent(Event):
    chat_history_message: List[ChatMessage]

class StreamEvent(Event):
    delta: str

class CodeExecutionEvent(Event):
    code: str


# code_agent system prompt
# CODE_ACT_SYSTEM_PROMPT = """
# You are a helpful assistant that can execute code.
# Given the chat history, you can write code within <execute>...</execute> tags to help the user with their question.In your code, you can reference any previously used variables or functions.
# The user provides you with some predefined methods for your use:
# {tool_fn_str}
# To execute code, write the code between <execute>...</execute> tags.
# """

CODE_ACT_SYSTEM_PROMPT = """
你是一个助手，可以执行代码。给定聊天记录，您可以在<execute>...</execute>标记来帮助用户解决他们的问题。在您的代码中，您可以引用任何以前使用过的变量或函数。
用户提供了一些预定义的函数供你使用：
{tool_fn_str}
要执行代码，请编写<execute>...</execute>标记。
"""

class CodeAgentWorkflow(Workflow):
    def __init__(
            self,
            tool_fns: List[Callable],
            code_executor_fn: Callable,
            llm: LLM = OpenAILike(),
            **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.tool_fns = tool_fns
        self.code_executor_fn = code_executor_fn
        self.llm = llm
        # 解析工具函数为：截断的函数字符串
        # 也即是说:获取所有工具函数的注解
        self.tools_fn_str = "\n\n".join(
            [
                f'def {tool_fn.__name__}{str(inspect.signature(tool_fn))}:\n """{tool_fn.__doc__}"""'
                for tool_fn in self.tool_fns
            ]
        )
        # 设置系统提示词
        self.system_message = ChatMessage(
            role="system",
            content=CODE_ACT_SYSTEM_PROMPT.format(
                tool_fn_str=self.tools_fn_str,
            )
        )

    @staticmethod
    def _parse_code(response: str) -> str | None:
        """ 从<execute></execute>从解析出llm编写代码"""
        matched = re.findall(r"<execute>(.*?)</execute>", response, re.DOTALL)
        if matched:
            return "\n".join(matched)
        return None

    @step
    async def prepare_chat_history(self, ctx: Context, ev: StartEvent) -> InputEvent:
        # 获取内存，目的是把chat对话放到内存中记忆
        chat_history_message = await ctx.store.get("memory", default=None)
        if chat_history_message is None:
            memory:Memory = Memory.from_defaults(
                token_limit=self.llm.metadata.context_window
            )
        # 获取用户角色输入的内容
        user_query = ev.user_query or None
        # check user_query
        if user_query is None:
            raise ValueError("user_query can't be None.")
        # 创建用户的chat_message
        memory.put(ChatMessage(
            role="user",
            content=user_query,
        ))
        # 更新当前上下文
        await ctx.store.set("memory", memory)

        # get chat_history_message
        chat_history_message = memory.get()
        # 返回结果
        return InputEvent(
            chat_history_message=[self.system_message, *chat_history_message]
        )

    @step
    async def handle_llm_input(self, ctx: Context, ev: InputEvent) -> CodeExecutionEvent | StopEvent:
        # 获取agent对话历史信息
        chat_history_message = ev.chat_history_message
        # 获取llm的响应
        chat_response_generator = await self.llm.astream_chat(chat_history_message)
        # 增量向外trigger stream event
        async for chat_response in chat_response_generator:
            ctx.write_event_to_stream(StreamEvent(
                delta=chat_response.delta or ""
            ))
        # save the final response , which should have all content
        memory = await ctx.store.get("memory")
        memory.put(chat_response.message)
        # update memory
        await ctx.store.set("memory", memory)

        # 获取 tool code执行
        code = CodeAgentWorkflow._parse_code(chat_response.message.content)
        # trigger不同的事件
        if not code:
            return StopEvent(
                result=chat_response
            )
        else:
            return CodeExecutionEvent(
                code=code
            )
    @step
    async def handle_code_execution(self, ctx: Context, ev: CodeExecutionEvent) -> InputEvent:
        # execute the code
        ctx.write_event_to_stream(ev)
        # execute executor_fn
        output = self.code_executor_fn(ev.code)
        # update the memory
        memory = await ctx.store.get("memory")
        memory.put(
            ChatMessage(
                role="assistant", content=output,
            )
        )
        # update memory
        await ctx.store.set("memory", memory)
        # 获取最新的聊天记录并循环回到开始
        chat_history_message = memory.get()
        # 返回输入的事件
        return InputEvent(
            chat_history_message=chat_history_message
        )

code_executor = SimpleCodeExecutor(
    # give access to our functions defined above
    locals={
        "add": add,
        "subtract": subtract,
        "multiply": multiply,
        "divide": divide,
    },
    globals={
        # give access to all builtins
        "builtins": __import__("builtins"),
        # give access to numpy
        "np": __import__("numpy"),
    },
)

async def main():
    code_agent_workflow = CodeAgentWorkflow(
        tool_fns=[add, subtract, multiply, divide],
        code_executor_fn=code_executor.execute,
        llm=OpenAILike(
            model=LLM_NAMESPACE[LLM_NAME],
            api_key=LLM_NAMESPACE[LLM_API_KEY],
            api_base=LLM_NAMESPACE[LLM_URL],
            is_chat_model=True,
            is_function_calling_model=True,
        ),
        timeout=200,
        verbose=False,
    )

    handle = code_agent_workflow.run(
        user_query="Calculate the sum of all numbers from 1 to 10"
    )

    async for event in handle.stream_events():
        if isinstance(event, StreamEvent):
            # logger.info(event.delta, model="a")
            print(f"{event.delta}", end="", flush=True)
        elif isinstance(event, CodeExecutionEvent):
            print(f"\n------------\n Parsed code:\n {event.code}\n")

    # final result
    await handle

if __name__ == "__main__":
    asyncio.run(main())



