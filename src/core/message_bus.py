"""
src/core/message_bus.py — Agent-to-Agent Message Bus
All agent communication goes through here. No agent imports another agent.

Design decisions:
1. DECOUPLING: Planner posts a Message. Bus delivers it. Agents swappable.
2. IN-MEMORY QUEUE: Simple dict of deques — one inbox per agent.
3. FULL HISTORY: All messages logged for Streamlit activity feed.
4. SYNC BY DEFAULT: Phase 4 can add async without changing the interface.
"""

import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional
from loguru import logger


class MessageType(str, Enum):
    TASK             = "task"
    RESULT           = "result"
    REVISION_REQUEST = "revision"
    STATUS_UPDATE    = "status"
    MEMORY_NOTIFY    = "memory_notify"


class MessagePriority(int, Enum):
    LOW    = 0
    NORMAL = 1
    HIGH   = 2
    URGENT = 3


@dataclass
class Message:
    sender:       str
    recipient:    str
    message_type: MessageType
    task_id:      str
    content:      dict[str, Any] = field(default_factory=dict)
    priority:     MessagePriority = MessagePriority.NORMAL
    message_id:   str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp:    str = field(default_factory=lambda: datetime.utcnow().isoformat())
    reply_to:     Optional[str] = None

    def __repr__(self):
        return (f"Message(id={self.message_id}, {self.sender}→{self.recipient}, "
                f"type={self.message_type.value}, task={self.task_id})")


class MessageBus:
    """
    Central in-memory message bus.
    Each agent has its own inbox (deque). Bus routes messages to the right inbox.
    """

    def __init__(self):
        self._inboxes: dict[str, deque[Message]] = {}
        self._history: list[Message] = []
        logger.info("MessageBus initialised")

    def _ensure_inbox(self, agent_name: str):
        if agent_name not in self._inboxes:
            self._inboxes[agent_name] = deque()

    def send(self, message: Message) -> str:
        self._ensure_inbox(message.recipient)
        self._inboxes[message.recipient].append(message)
        self._history.append(message)
        logger.debug(f"[bus] {message.sender} → {message.recipient} [{message.message_type.value}]")
        return message.message_id

    def broadcast(self, sender, recipients, message_type, task_id, content, priority=MessagePriority.NORMAL):
        ids = []
        for recipient in recipients:
            msg = Message(sender=sender, recipient=recipient, message_type=message_type,
                          task_id=task_id, content=content, priority=priority)
            ids.append(self.send(msg))
        return ids

    def receive(self, agent_name: str, message_type: Optional[MessageType] = None,
                drain: bool = True) -> list[Message]:
        self._ensure_inbox(agent_name)
        inbox = self._inboxes[agent_name]
        if not inbox:
            return []
        if drain:
            if message_type is None:
                messages = list(inbox)
                inbox.clear()
            else:
                matching, remaining = [], []
                while inbox:
                    msg = inbox.popleft()
                    (matching if msg.message_type == message_type else remaining).append(msg)
                for msg in remaining:
                    inbox.appendleft(msg)
                messages = matching
        else:
            messages = [m for m in inbox if message_type is None or m.message_type == message_type]
        logger.debug(f"[bus] {agent_name} received {len(messages)} messages")
        return messages

    def receive_one(self, agent_name: str, message_type: Optional[MessageType] = None):
        messages = self.receive(agent_name, message_type=message_type, drain=True)
        return messages[0] if messages else None

    def pending_count(self, agent_name: str) -> int:
        self._ensure_inbox(agent_name)
        return len(self._inboxes[agent_name])

    def get_history(self, task_id=None, sender=None, recipient=None):
        h = self._history
        if task_id:   h = [m for m in h if m.task_id == task_id]
        if sender:    h = [m for m in h if m.sender == sender]
        if recipient: h = [m for m in h if m.recipient == recipient]
        return h

    def clear_task(self, task_id: str):
        for inbox in self._inboxes.values():
            to_remove = [m for m in inbox if m.task_id == task_id]
            for msg in to_remove:
                inbox.remove(msg)

    def reset(self):
        self._inboxes.clear()
        self._history.clear()
        logger.info("[bus] Message bus reset")


_bus_instance: Optional[MessageBus] = None

def get_bus() -> MessageBus:
    global _bus_instance
    if _bus_instance is None:
        _bus_instance = MessageBus()
    return _bus_instance


def make_task_message(sender, recipient, task_id, description,
                      context=None, expected_output="",
                      priority=MessagePriority.NORMAL) -> Message:
    return Message(
        sender=sender, recipient=recipient, message_type=MessageType.TASK,
        task_id=task_id,
        content={"description": description, "context": context or {},
                 "expected_output": expected_output},
        priority=priority,
    )


def make_result_message(sender, recipient, task_id, output,
                        chunk_ids=None, tool_calls=None,
                        success=True, error=None) -> Message:
    return Message(
        sender=sender, recipient=recipient, message_type=MessageType.RESULT,
        task_id=task_id,
        content={"output": output, "chunk_ids": chunk_ids or [],
                 "tool_calls": tool_calls or [], "success": success, "error": error},
    )


if __name__ == "__main__":
    from rich import print as rprint
    from rich.table import Table
    rprint("[bold cyan]Testing Message Bus...[/bold cyan]")
    bus = get_bus()
    task_id = "demo_001"
    bus.send(make_task_message("planner", "researcher", task_id, "Find solar data"))
    messages = bus.receive("researcher")
    rprint(f"Researcher received: {messages[0].content['description']}")
    bus.send(make_result_message("researcher", "planner", task_id, "China: 430 GW"))
    history = bus.get_history(task_id=task_id)
    table = Table(title="Message History")
    table.add_column("From"); table.add_column("To"); table.add_column("Type")
    for msg in history:
        table.add_row(msg.sender, msg.recipient, msg.message_type.value)
    rprint(table)
    rprint("[bold green]✓ Message Bus test passed![/bold green]")
