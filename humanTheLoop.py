import asyncio
# 人工介入等待/恢复工作流
import uuid
from typing import Any, List, Dict, Optional
from loguru import logger
from llama_index.core import PromptTemplate
from llama_index.core.schema import BaseModel, Field
from llama_index.llms.openai_like import OpenAILike
from llama_index.core.workflow import step, Context, HumanResponseEvent
from llama_index.utils.workflow import Event, Workflow, StopEvent, StartEvent

from agents.settings import LLM_NAMESPACE, LLM_URL, LLM_NAME, LLM_API_KEY

LLM = OpenAILike(
    model=LLM_NAMESPACE[LLM_NAME],
    api_key=LLM_NAMESPACE[LLM_API_KEY],
    api_base=LLM_NAMESPACE[LLM_URL],
    is_chat_model=True,
    is_function_calling_model=True
)


# 片段生成提示词
SEGMENT_GENERATION_TEMPLATE = PromptTemplate(
    template="""
你正在与一个人合作，以“选择你的冒险”的风格创作一个故事。

这个人扮演故事中的主角，而你负责协助编写故事。为了创作故事，我们分步骤进行，每一步产生一个 BLOCK。
每个 BLOCK 包含一个 PLOT、主角可以采取的一组 ACTIONS，以及选择的 ACTION。

下面我们附上迄今为止的冒险历史。

之前的 BLOCKS：
---
{running_history}

通过生成下一个 BLOCK 的 PLOT 和 ACTIONS 集合来继续故事。如果没有之前的 BLOCKS，开始一个有趣的新故事。给主角起一个名字，并给他设定一个有趣的挑战来解决。

使用提供的数据模型来组织你的输出。
"""
)

# 故事最终片段生成提示词
FINAL_SEGMENT_GENERATION_TEMPLATE = PromptTemplate(
    template="""
你正在与一个人合作，以选择你自己的冒险故事风格创作一个故事。

这个人正在扮演故事中的主角，而你则被要求协助编写故事。为了创作故事，我们分步骤进行，每一步都会生成一个 BLOCK。
每个 BLOCK 包含一个 PLOT、一组主角可以采取的行动以及选择的行为。下面我们附上到目前为止的冒险历史。

之前的 BLOCK：
---
{running_history}

故事现在即将结束。根据之前的 BLOCK，用结尾的 PLOT 来总结故事。由于这是一个结尾 PLOT，请勿生成新的行动集。

使用提供的数据模型来组织你的输出。
"""
)

BLOCK_TEMPLATE = """
BLOCK
=====
PLOT: {plot}
ACTIONS: {actions}
CHOICE: {choice}
"""

HUMAN_CHOICE_PROMPT = """
===={plot}====
Choice your adventure: {choices}
"""

# 片段对象
class Segment(BaseModel):
    """ Data Model for generator segments of a story."""

    plot: str = Field(
        description="当前片段的冒险情节。情节不应该超过3句话。"
    )
    actions: List[str] = Field(
        default=[],
        description="主角可以采取的、影响下一段情节和行为的行动列表",
    )

# 需要将生成的block转换为running_story
class Block(BaseModel):
    id: str = Field(
        description="block id",
        default_factory=lambda: str(uuid.uuid4())
    )
    segment: Segment
    choice: Optional[str] = None,
    block_template: str = BLOCK_TEMPLATE

    def __str__(self) -> str:
        return self.block_template.format(
            plot=self.segment.plot,
            actions=", ".join(self.segment.actions),
            choice=self.choice,
        )

class NewBlockEvent(Event):
    block: Block

class HumanChoiceEvent(Event):
    block_id: str

class HumanInputEvent(Event):
    prompt: str


class ChooseYourOwnAdventureWorkflow(Workflow):
    def __init__(self, max_step:int = 3, **kwargs) -> None:
        super().__init__(**kwargs)
        self.max_step = max_step
        self.llm = LLM

    @step
    async def create_segment(self, ctx: Context, ev: StartEvent | HumanChoiceEvent) -> StopEvent | NewBlockEvent:
        # 获取生成running_history
        blocks: List[Block] = await ctx.store.get("blocks", [])
        running_history = "\n".join([str(block) for block in blocks])
        # 判断是否超出了最大step
        if len(blocks) < self.max_step:
            new_segment = self.llm.structured_predict(
                output_cls=Segment,
                prompt=SEGMENT_GENERATION_TEMPLATE,
                running_history=running_history,
            )
            # new block
            new_block = Block(segment=new_segment)
            blocks.append(new_block)
            await ctx.store.set("blocks", blocks)
            return NewBlockEvent(block=new_block)
        else:
            # 超出了最大步骤次数，需要直接总结最后的结果
            final_segment = self.llm.structured_predict(
                output_cls=Segment,
                prompt=FINAL_SEGMENT_GENERATION_TEMPLATE,
                running_history=running_history,
            )
            final_block = Block(segment=final_segment)
            blocks.append(final_block)
            return StopEvent(
                result=blocks
            )
    @step
    async def prompt_human(self, ctx: Context, ev: NewBlockEvent) -> HumanChoiceEvent:
        block: Block = ev.block
        # 提示词
        choice_prompt = HUMAN_CHOICE_PROMPT.format(
            plot=block.segment.plot,
            choices="\n".join([
                f"{index + 1}.{action}"
                for index, action in enumerate(block.segment.actions)
            ])
        )
        # wait_event_humanResponse
        human_response = await ctx.wait_for_event(
            HumanResponseEvent,
            waiter_event=HumanInputEvent(prompt=choice_prompt),
            waiter_id=block.id
        )
        blocks = await ctx.store.get("blocks")
        block.choice = human_response.response
        blocks[-1] = block
        await ctx.store.set("blocks", blocks)
        # 返回事件
        return HumanChoiceEvent(
            block_id=block.id,
        )

async def workflow_run():
    work_flow = ChooseYourOwnAdventureWorkflow(
        timeout=200,
        verbose=False,
    )
    work_flow_handle = work_flow.run()
    async for ev in work_flow_handle.stream_events():
        if isinstance(ev, HumanInputEvent):
            human_choice_prompt = ev.prompt
            user_input = input(human_choice_prompt)
            await work_flow_handle.send_event(
                HumanResponseEvent(response=user_input),
            )
    final_response = await work_flow_handle

    logger.success(final_response)


if __name__ == "__main__":
    asyncio.run(workflow_run())