"""A2A AgentExecutor for the Orchestrator Agent."""

from __future__ import annotations

import logging

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    InternalError,
    InvalidParamsError,
    Part,
    TaskState,
    TextPart,
    UnsupportedOperationError,
)
from a2a.utils import new_agent_text_message, new_task
from a2a.utils.errors import ServerError

from agent import OrchestratorAgent

logger = logging.getLogger(__name__)


class OrchestratorExecutor(AgentExecutor):
    """A2A executor for the Orchestrator Agent."""

    def __init__(self) -> None:
        pass

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        if self._validate_request(context):
            raise ServerError(error=InvalidParamsError())

        query = context.get_user_input()
        task = context.current_task

        if not task:
            task = new_task(context.message)  # type: ignore[arg-type]
            await event_queue.enqueue_event(task)

        # Pass the context_id into the agent so downstream A2A calls share the session
        agent = OrchestratorAgent(context_id=task.context_id)
        updater = TaskUpdater(event_queue, task.id, task.context_id)

        try:
            async for item in agent.stream(query, task.context_id):
                is_complete = item["is_task_complete"]
                needs_input = item["require_user_input"]
                content: str = item["content"]

                if not is_complete and not needs_input:
                    await updater.update_status(
                        TaskState.working,
                        new_agent_text_message(content, task.context_id, task.id),
                    )
                elif needs_input:
                    await updater.update_status(
                        TaskState.input_required,
                        new_agent_text_message(content, task.context_id, task.id),
                        final=True,
                    )
                    break
                else:
                    await updater.add_artifact(
                        [Part(root=TextPart(text=content))],
                        name="orchestrator_result",
                    )
                    await updater.complete()
                    break

        except Exception as exc:
            logger.exception("Orchestrator execution failed")
            raise ServerError(error=InternalError()) from exc

    def _validate_request(self, context: RequestContext) -> bool:
        return not context.get_user_input()

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise ServerError(error=UnsupportedOperationError())
