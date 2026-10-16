import asyncio
from loguru import logger
from llama_index.core.workflow import step, Context
from llama_index.core.schema import BaseModel
from llama_index.llms.openai_like import OpenAILike
from llama_index.utils.workflow import Event, Workflow, StartEvent, StopEvent
from agents.settings import LLM_NAMESPACE,LLM_URL,LLM_NAME,LLM_API_KEY

class ExtractionDone(Event):
    output: str
    describe: str

class ValidationErrorEvent(Event):
    error: str
    wrong_output: str
    describe: str


class Car(BaseModel):
    brand: str
    model: str
    power: str


class CarCollection(BaseModel):
    cars: list[Car]


EXTRACTION_PROMPT = """
背景信息如下所示:
--------------------------
{describe}
--------------------------

给定上述的背景信息并且没有先前知识的前提下，根据上述背景信息创建一个`JSON`对象。
这个创建的`JSON`对象格式必须遵循下面的JSON schema:
{json_schema}
"""
REFLECTION_PROMPT = """
你之前创建的输出如下所示:
---------------------------
{wrong_output}
---------------------------

导致JSON decode失败的异常如下所示:
{error}

请你再次尝试，响应必须仅包含有效的JSON代码。不要再JSON对象之前或之后添加任何句子。不要重复JSON schema。
"""


class ReflectionWorkflow(Workflow):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 最大尝试数值
        self.max_retries: int = 3
        # llm
        self.llm = OpenAILike(
            model=LLM_NAMESPACE[LLM_NAME],
            api_key=LLM_NAMESPACE[LLM_API_KEY],
            api_base=LLM_NAMESPACE[LLM_URL],
            is_chat_model=True,
            is_function_calling_model=True,
        )
        pass

    @step
    async def extract(self, ctx: Context, ev: StartEvent | ValidationErrorEvent) -> ExtractionDone | StopEvent:
        # 获取最大尝试参数
        current_retries = await ctx.store.get("max_retries", default=0)
        # 判断是否超过尝试次数
        if current_retries < self.max_retries:
            await ctx.store.set("max_retries", current_retries + 1)
        else:
        # 超过最大的尝试次数，直接返回结果
            return StopEvent(
                result="达到的最大重试次数"
            )
        # 获取用户的输入
        if isinstance(ev, StartEvent):
            # 获取用户的输入
            describe = ev.describe or None
            if describe is None:
                return StopEvent(
                    result="请提供一些输入的文本"
                )
            reflection_prompt = None
        elif isinstance(ev, ValidationErrorEvent):
            describe = ev.describe
            reflection_prompt = REFLECTION_PROMPT.format(
                wrong_output = ev.wrong_output,
                error = ev.error
            )
        # call llm structured output
        llm_prompt = EXTRACTION_PROMPT.format(
            describe = describe,
            json_schema = CarCollection.model_json_schema()
        )
        # if reflection_prompt
        if reflection_prompt is not None:
            llm_prompt += "\n" + reflection_prompt

        # call llm structured output
        output = await self.llm.acomplete(prompt=llm_prompt)

        # return event
        return ExtractionDone(
            output = str(output),
            describe=describe
        )

    @step
    async def validate(self, ev: ExtractionDone) -> ValidationErrorEvent | StopEvent:
        try:
            CarCollection.model_validate_json(
                json_data=ev.output
            )
        except Exception as e:
            return ValidationErrorEvent(
                error=str(e),
                wrong_output=ev.output,
                describe=ev.describe
            )
        return StopEvent(
            result=ev.output
        )

async def work_flow_run():
    workflow_handle = ReflectionWorkflow(
        timeout=200,
        verbose=False
    )
    output = await workflow_handle.run(
       describe="我有两辆车，一辆马力30HP的比亚迪海豹和一辆马力60HP的奔驰迈巴赫。"
    )
    logger.success(output)

if __name__ == "__main__":
    asyncio.run(work_flow_run())