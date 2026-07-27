"""
src/core/base_agent.py — Base ReAct Agent
All specialist agents inherit from this class.

What is the ReAct loop?
  THOUGHT: "I need to find solar energy statistics"
  ACTION:  call web_search("solar energy capacity 2024")
  OBSERVATION: [results]
  THOUGHT: "I need India data too"
  ACTION:  call web_search("India solar capacity")
  OBSERVATION: [more results]
  THOUGHT: "I have enough"
  FINAL_ANSWER: [full summary]

Specialist agents override:
  - agent_name, system_prompt, tools — their identity and capabilities
  - run() — entry point (receives SubTask, returns result string)
"""

import json
import re
from abc import ABC
from typing import Any, Callable, Optional
from loguru import logger

from config import settings, TaskStatus
from src.core.llm_client import LLMClient
from src.core.memory import get_memory, ChunkType
from src.core.message_bus import get_bus, make_result_message, MessageType
from src.core.task_queue import get_queue, SubTask


class Tool:
    """A named, callable tool the agent can invoke."""
    def __init__(self, name: str, description: str, func: Callable, example: str = ""):
        self.name = name
        self.description = description
        self.func = func
        self.example = example

    def call(self, **kwargs) -> str:
        try:
            result = self.func(**kwargs)
            return result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, indent=2)
        except Exception as e:
            return f"Tool error in '{self.name}': {e}"

    def to_prompt_str(self) -> str:
        base = f"- {self.name}: {self.description}"
        if self.example:
            base += f"\n  Example: {self.example}"
        return base


class BaseAgent(ABC):
    """
    Abstract base class for all AgentForge agents.
    Provides the full ReAct loop, memory access, and message bus integration.

    Subclasses define:
      agent_name   (str)          : unique identifier
      system_prompt (str)         : agent personality
      tools        (dict[str, Tool]): available tools
    """

    agent_name: str = "base_agent"
    system_prompt: str = "You are a helpful AI assistant."
    tools: dict[str, Tool] = {}

    def __init__(self):
        self.memory = get_memory()
        self.bus    = get_bus()
        self.queue  = get_queue()
        self.llm    = LLMClient(
            agent_name=self.agent_name,
            system_prompt=self._build_system_prompt(),
        )
        self._conversation_history: list[dict] = []
        logger.info(f"[{self.agent_name}] Agent initialised")

    def _build_system_prompt(self) -> str:
        tools_desc = "\n".join(t.to_prompt_str() for t in self.tools.values()) if self.tools                      else "No tools — reason from existing knowledge."
        react_instructions = f"""
You operate in a THOUGHT → ACTION → OBSERVATION loop.

Format A — to use a tool:
THOUGHT: [your reasoning]
ACTION: tool_name
ACTION_INPUT: {{"arg1": "value1"}}

Format B — when you have the final answer:
THOUGHT: [your reasoning]
FINAL_ANSWER: [complete response]

Available tools:
{tools_desc}

Rules:
- Always start with THOUGHT
- ACTION_INPUT must be valid JSON
- On tool error, try a different approach
- Stop when you have enough info
- FINAL_ANSWER must fully address the task
- Max {settings.max_react_iterations} iterations
"""
        return f"{self.system_prompt}\n\n{react_instructions}"

    def react(self, task_description: str, task_id: str, subtask_id: Optional[str] = None) -> str:
        """Run the ReAct loop. Returns the final answer string."""
        self._conversation_history = [{"role": "user", "content": f"Task: {task_description}"}]
        iterations = 0
        final_answer = None

        while iterations < settings.max_react_iterations:
            iterations += 1
            logger.debug(f"[{self.agent_name}] ReAct iteration {iterations}")

            response = self.llm.complete(self._conversation_history)
            self._conversation_history.append({"role": "assistant", "content": response})

            thought = self._extract_thought(response)
            if thought and task_id:
                self.queue.log_agent_action(task_id=task_id, agent_name=self.agent_name,
                                            log_type="thought", content=thought,
                                            subtask_id=subtask_id)

            if "FINAL_ANSWER:" in response:
                final_answer = self._extract_final_answer(response)
                logger.info(f"[{self.agent_name}] FINAL_ANSWER after {iterations} iterations")
                break

            action_name, action_input = self._parse_action(response)
            if not action_name:
                logger.warning(f"[{self.agent_name}] No valid action on iteration {iterations}")
                final_answer = response.strip()
                break

            observation = self._execute_tool(action_name, action_input, task_id, subtask_id)
            self._conversation_history.append({"role": "user", "content": f"OBSERVATION: {observation}"})

        if final_answer is None:
            logger.warning(f"[{self.agent_name}] Max iterations reached, using last response")
            last = self._conversation_history[-1]["content"]
            final_answer = self._extract_final_answer(last) or last

        return final_answer

    def _parse_action(self, text: str):
        action_match = re.search(r"ACTION:\s*(\w+)", text)
        if not action_match:
            return None, {}
        action_name = action_match.group(1).strip()
        input_match = re.search(r"ACTION_INPUT:\s*(\{.*?\})", text, re.DOTALL)
        if not input_match:
            return action_name, {}
        try:
            return action_name, json.loads(input_match.group(1))
        except json.JSONDecodeError:
            return action_name, {}

    def _execute_tool(self, action_name, action_input, task_id, subtask_id=None) -> str:
        if action_name not in self.tools:
            return (f"Error: Unknown tool '{action_name}'. "
                    f"Available: {list(self.tools.keys())}")
        tool = self.tools[action_name]
        logger.debug(f"[{self.agent_name}] Calling {action_name} with {action_input}")
        if task_id:
            self.queue.log_agent_action(task_id=task_id, agent_name=self.agent_name,
                log_type="action",
                content=f"Tool: {action_name}, Input: {json.dumps(action_input)}",
                subtask_id=subtask_id)
        observation = tool.call(**action_input)
        if task_id:
            self.queue.log_agent_action(task_id=task_id, agent_name=self.agent_name,
                log_type="observation",
                content=f"Tool: {action_name}, Result: {observation[:500]}",
                subtask_id=subtask_id)
        return observation

    def _extract_thought(self, text) -> Optional[str]:
        m = re.search(r"THOUGHT:\s*(.+?)(?=\nACTION:|FINAL_ANSWER:|$)", text, re.DOTALL)
        return m.group(1).strip() if m else None

    def _extract_final_answer(self, text) -> Optional[str]:
        m = re.search(r"FINAL_ANSWER:\s*(.+)", text, re.DOTALL)
        return m.group(1).strip() if m else None

    def store_result(self, content, task_id, chunk_type=ChunkType.TOOL_OUTPUT, metadata=None) -> str:
        return self.memory.store(content=content, agent_name=self.agent_name,
                                 task_id=task_id, chunk_type=chunk_type, metadata=metadata)

    def retrieve_context(self, query, task_id, k=3) -> str:
        chunks = self.memory.retrieve(query=query, task_id=task_id, k=k)
        if not chunks:
            return "No relevant context found in memory."
        return "\n\n".join(
            f"[Context {i+1} from {c['metadata'].get('agent_name','unknown')}]:\n{c['content']}"
            for i, c in enumerate(chunks)
        )

    def run(self, subtask: SubTask) -> str:
        """Main entry point. Runs ReAct, stores result, sends via bus."""
        task_id, subtask_id = subtask.task_id, subtask.subtask_id
        logger.info(f"[{self.agent_name}] Starting {subtask_id}: '{subtask.description[:60]}'")
        self.queue.update_subtask(subtask_id, status=TaskStatus.IN_PROGRESS)

        try:
            context_str = ""
            if subtask.context:
                context_str = f"\n\nContext from previous agents:\n{json.dumps(subtask.context, indent=2)}"
            task_description = (f"{subtask.description}\n\n"
                                f"Expected output: {subtask.expected_output}{context_str}")

            result = self.react(task_description, task_id, subtask_id)
            result = self._post_process(result, subtask)

            chunk_id = self.store_result(content=result, task_id=task_id,
                chunk_type=self._chunk_type(),
                metadata={"subtask_id": subtask_id, "description": subtask.description})

            self.queue.complete_subtask(subtask_id, result=result, chunk_ids=[chunk_id])
            self.bus.send(make_result_message(
                sender=self.agent_name, recipient="planner",
                task_id=task_id, output=result, chunk_ids=[chunk_id], success=True))

            logger.info(f"[{self.agent_name}] {subtask_id} completed")
            return result

        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}"
            logger.error(f"[{self.agent_name}] {subtask_id} failed: {error_msg}")
            self.queue.fail_subtask(subtask_id, error=error_msg)
            self.queue.log_agent_action(task_id=task_id, agent_name=self.agent_name,
                                        log_type="error", content=error_msg,
                                        subtask_id=subtask_id)
            self.bus.send(make_result_message(
                sender=self.agent_name, recipient="planner",
                task_id=task_id, output="", success=False, error=error_msg))
            return ""

    def _post_process(self, result: str, subtask: SubTask) -> str:
        return result

    def _chunk_type(self) -> str:
        return ChunkType.TOOL_OUTPUT


if __name__ == "__main__":
    from rich import print as rprint
    rprint("[bold cyan]Testing BaseAgent...[/bold cyan]")

    class EchoAgent(BaseAgent):
        agent_name = "echo_agent"
        system_prompt = "You are a simple echo agent for testing."
        tools = {}

    agent = EchoAgent()
    rprint(f"Agent name   : {agent.agent_name}")
    rprint(f"System prompt: {len(agent.llm.system_prompt)} chars built")
    rprint("[bold green]✓ BaseAgent instantiation passed![/bold green]")
    rprint("[dim](Full ReAct test needs API keys — run src/core/llm_client.py)[/dim]")
