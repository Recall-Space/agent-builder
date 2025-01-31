"""

"""

import json
import websockets

from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, Tuple, Callable
from urllib.parse import urlparse


@asynccontextmanager
async def websocket_connection(
    *,
    api_key: str,
    model_url: str = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01",
) -> AsyncIterator[
    Tuple[Callable[[Dict[str, Any] | str], None], AsyncIterator[Dict[str, Any]]]
]:
    """
    Asynchronous context manager to establish and manage a WebSocket connection
    with the OpenAI Realtime API.

    This function provides an interface to send events to the API and receive responses
    as an event stream.

    Parameters
    ----------
    api_key : str
        The API key used for authentication with OpenAI's Realtime API.
    model_url : str, optional
        The WebSocket URL of the OpenAI Realtime API model (default is GPT-4o preview).
        Can also be an Azure OpenAI Realtime hosted model URL.

    Yields
    ------
    Tuple[Callable[[Dict[str, Any] | str], None], AsyncIterator[Dict[str, Any]]]
        A tuple containing:
        - `send_message`: A coroutine function to send messages to the server.
        - `receive_messages`: An asynchronous generator yielding responses from the server.
    """
    parsed_url = urlparse(model_url)
    headers = {}

    if "api.openai.com" in parsed_url.netloc:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "OpenAI-Beta": "realtime=v1",
        }
    else:
        # For Azure Services
        headers = {
            "api-key": api_key,
        }

    async with websockets.connect(model_url, additional_headers=headers) as websocket:

        async def send_message(message: Dict[str, Any] | str) -> None:
            """
            Sends a message to the WebSocket server.

            Parameters
            ----------
            message : Dict[str, Any] | str
                The message data to be sent, either as a dictionary (which is
                serialized to JSON) or a preformatted JSON string.
            """
            formatted_message = (
                json.dumps(message) if isinstance(message, dict) else message
            )
            await websocket.send(formatted_message)

        async def receive_messages() -> AsyncIterator[Dict[str, Any]]:
            """
            Asynchronous generator that listens for incoming messages from the WebSocket.

            Yields
            ------
            Dict[str, Any]
                Parsed JSON responses received from the WebSocket connection.
            """
            async for raw_message in websocket:
                yield json.loads(raw_message)

        yield send_message, receive_messages()
