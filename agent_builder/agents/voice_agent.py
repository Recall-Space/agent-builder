import asyncio
import json
import websockets

from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, AsyncGenerator, Callable, Coroutine, Dict, List

from pydantic import BaseModel, PrivateAttr
from langchain_core.tools import BaseTool
from langchain_core._api import beta
import logging


# Events to ignore during processing
EVENTS_TO_IGNORE = {
    "response.function_call_arguments.delta",
    "rate_limits.updated",
    "response.audio_transcript.delta",
    "response.created",
    "response.content_part.added",
    "response.content_part.done",
    "conversation.item.created",
    "response.audio.done",
    "session.created",
    "session.updated",
    "response.done",
    "response.output_item.done",
}


@asynccontextmanager
async def connect(
    *,
    api_key: str,
    model_url: str = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01",
):
    """
    Asynchronous context manager to handle the connection to the OpenAI real-time API.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "OpenAI-Beta": "realtime=v1",
    }

    async with websockets.connect(model_url, additional_headers=headers) as websocket:

        async def send_event(event: Dict[str, Any] | str) -> None:
            formatted_event = json.dumps(event) if isinstance(event, dict) else event
            await websocket.send(formatted_event)

        async def event_stream() -> AsyncIterator[Dict[str, Any]]:
            async for raw_event in websocket:
                yield json.loads(raw_event)

        yield send_event, event_stream()


class VoiceToolExecutor(BaseModel):
    """
    Manages tool function calls and emits their outputs to a stream.
    """

    tools_by_name: Dict[str, BaseTool]
    _trigger_future: asyncio.Future = PrivateAttr(default_factory=asyncio.Future)
    _lock: asyncio.Lock = PrivateAttr(default_factory=asyncio.Lock)

    async def _trigger_func(self) -> Dict:
        # Waits for a tool call to be added
        return await self._trigger_future

    async def add_tool_call(self, tool_call: Dict) -> None:
        async with self._lock:
            if self._trigger_future.done():
                raise ValueError("Tool call already in progress")
            self._trigger_future.set_result(tool_call)

    async def _create_tool_call_task(self, tool_call: Dict) -> asyncio.Task:
        tool = self.tools_by_name.get(tool_call["name"])
        if tool is None:
            raise ValueError(
                f"Tool '{tool_call['name']}' not found. Available tools: {list(self.tools_by_name.keys())}"
            )
        # Parse arguments
        try:
            args = json.loads(tool_call["arguments"])
        except json.JSONDecodeError:
            raise ValueError(
                f"Failed to parse arguments '{tool_call['arguments']}'. Must be valid JSON."
            )

        async def run_tool() -> Dict:
            result = await tool.ainvoke(args)
            try:
                result_str = json.dumps(result)
            except TypeError:
                result_str = str(result)
            return {
                "type": "conversation.item.create",
                "item": {
                    "id": tool_call["call_id"],
                    "call_id": tool_call["call_id"],
                    "type": "function_call_output",
                    "output": result_str,
                },
            }

        return asyncio.create_task(run_tool())

    async def output_iterator(self) -> AsyncIterator[Dict]:
        trigger_task = asyncio.create_task(self._trigger_func())
        tasks = {trigger_task}
        while True:
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                tasks.remove(task)
                if task == trigger_task:
                    async with self._lock:
                        self._trigger_future = asyncio.Future()
                    trigger_task = asyncio.create_task(self._trigger_func())
                    tasks.add(trigger_task)
                    tool_call = task.result()
                    try:
                        new_task = await self._create_tool_call_task(tool_call)
                        tasks.add(new_task)
                    except ValueError as e:
                        yield {
                            "type": "conversation.item.create",
                            "item": {
                                "id": tool_call["call_id"],
                                "call_id": tool_call["call_id"],
                                "type": "function_call_output",
                                "output": f"Error: {str(e)}",
                            },
                        }
                else:
                    yield task.result()


@beta()
class VoiceAgent:
    """
    An agent designed for voice interactions, managing audio input/output streams and tool execution.
    """

    def __init__(
        self,
        api_key: str,
        model_url: str,
        goal: str,
        voice: str = "shimmer",
        tools: list = None,
        role: str = "",
        temperature: float = 0.7,
    ):
        self.api_key = api_key
        self.model_url = model_url
        self.goal = goal
        self.tools = tools
        self.role = role
        self.voice = voice
        self.temperature = temperature

        tool_names = ", ".join(
            [each_tool.__class__.__name__ for each_tool in self.tools]
        ).replace(", Tool", "")
        logging.warning(
            f"""

                Initializing VoiceAgent
                --------------------------
                name: {self.__class__.__name__}
                tools: {tool_names}
                model_url: {self.model_url}
                voice: {self.voice}
                """
        )

    async def aconnect(
        self,
        input_stream: AsyncIterator[str],
        send_output_chunk: Callable[[str], Coroutine[Any, Any, None]],
    ) -> None:
        """
        Connects to the OpenAI API and handles the communication.

        Parameters:
        -----------
        input_stream : AsyncIterator[str]
            Stream of input events to send to the model.
        send_output_chunk : Callable[[str], Coroutine[Any, Any, None]]
            Callback to handle output events from the model.
        """
        tools_by_name = {tool.name: tool for tool in self.tools}
        tool_executor = VoiceToolExecutor(tools_by_name=tools_by_name)

        async with connect(
            model_url=self.model_url,
            api_key=self.api_key,
        ) as (model_send, model_receive_stream):
            # Send initial session update with tools and instructions
            tool_defs = [
                {
                    "type": "function",
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": {"type": "object", "properties": tool.args},
                }
                for tool in tools_by_name.values()
            ]
            await model_send(
                {
                    "type": "session.update",
                    "session": {
                        "instructions": self.goal,
                        "voice": self.voice,
                        "temperature": self.temperature,
                        # "input_audio_transcription": {
                        #     "model": "whisper-1",
                        # },
                        "tools": tool_defs,
                    },
                }
            )
            # Merge input streams and process events
            async for stream_key, data_raw in self.amerge_streams(
                input_mic=input_stream,
                output_speaker=model_receive_stream,
                tool_outputs=tool_executor.output_iterator(),
            ):
                try:
                    data = (
                        json.loads(data_raw) if isinstance(data_raw, str) else data_raw
                    )
                except json.JSONDecodeError:
                    print("Error decoding data:", data_raw)
                    continue

                if stream_key == "input_mic":
                    await model_send(data)
                elif stream_key == "tool_outputs":
                    await model_send(data)
                    await model_send({"type": "response.create", "response": {}})
                elif stream_key == "output_speaker":
                    await self._process_output_speaker(
                        data, send_output_chunk, tool_executor
                    )

    async def _process_output_speaker(
        self,
        data: Dict,
        send_output_chunk: Callable[[str], Coroutine[Any, Any, None]],
        tool_executor: VoiceToolExecutor,
    ) -> None:
        event_type = data.get("type")
        if event_type == "response.audio.delta":
            await send_output_chunk(json.dumps(data))
        elif event_type == "input_audio_buffer.speech_started":
            await send_output_chunk(json.dumps(data))
        elif event_type == "error":
            print("Error:", data)
        elif event_type == "response.function_call_arguments.done":
            await tool_executor.add_tool_call(data)
        elif event_type == "response.audio_transcript.done":
            print("Assistant transcript:", data.get("transcript", ""))
        elif event_type == "conversation.item.input_audio_transcription.completed":
            print("User transcript:", data.get("transcript", ""))
        elif event_type in EVENTS_TO_IGNORE:
            pass
        else:
            print("Unhandled event type:", event_type)

    async def ainvoke(
        self,
        input_stream: AsyncIterator[str],
        send_output_chunk: Callable[[str], Coroutine[Any, Any, None]],
    ) -> None:
        """
        Asynchronously invokes the agent with the given input and output handlers.

        Parameters:
        -----------
        input_stream : AsyncIterator[str]
            The input audio stream to be processed by the agent.
        send_output_chunk : Callable[[str], Coroutine[Any, Any, None]]
            The callback function to handle output events.
        """
        await self.aconnect(input_stream, send_output_chunk)

    async def amerge_streams(self, **streams) -> AsyncIterator[tuple]:
        """
        Merges multiple asynchronous iterators into a single iterator.

        Parameters:
        -----------
        streams : dict
            A dictionary of named async iterators.

        Yields:
        -------
        tuple
            A tuple containing the stream name and the data from the stream.
        """
        tasks = {
            name: asyncio.create_task(self._wrap_stream(name, stream))
            for name, stream in streams.items()
        }
        while tasks:
            done, pending = await asyncio.wait(
                tasks.values(), return_when=asyncio.FIRST_COMPLETED
            )
            for task in done:
                result = task.result()
                if result is None:
                    # Stream has ended
                    name = next(name for name, t in tasks.items() if t == task)
                    del tasks[name]
                else:
                    name, data = result
                    yield name, data
                    # Re-create the task for the next item
                    tasks[name] = asyncio.create_task(
                        self._wrap_stream(name, streams[name])
                    )

    async def _wrap_stream(
        self, name: str, stream: AsyncIterator[Any]
    ) -> tuple[str, Any] | None:
        try:
            data = await stream.__anext__()
            return name, data
        except StopAsyncIteration:
            return None
