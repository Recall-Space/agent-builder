import logging
from typing import Any, Dict, List, Optional

from langchain_core.tools import BaseTool

from agent_builder.agents.voice_agent import VoiceAgent


class VoiceAgentBuilder:
    """
    A builder class for constructing `VoiceAgent` instances with customizable configurations.

    This class follows the builder pattern, providing methods to configure various components
    of the `VoiceAgent` and ensure the agent is fully customizable.

    **Attributes:**

    - **api_key** (`str`): The API key used to authenticate with the OpenAI Realtime API.
    - **model_url** (`str`): The WebSocket URL of the OpenAI Realtime API model.
    - **goal** (`str`): The assistant's instructions or goal.
    - **voice** (`str`): The voice model to use for audio output.
    - **tools** (`List[BaseTool]`): A list of tools that the agent can use.
    - **temperature** (`float`): The sampling temperature for response generation.
    - **modalities** (`List[str]`): Modes of input/output, e.g., ["text", "audio"].
    - **input_audio_format** (`str`): The format of the input audio.
    - **output_audio_format** (`str`): The format of the output audio.
    - **input_audio_transcription** (`Optional[Dict[str, Any]]`): Configuration for input audio transcription.
    - **turn_detection** (`Optional[Dict[str, Any]]`): Configuration for turn detection.
    - **tool_choice** (`str`): How the agent chooses tools, e.g., "auto".
    - **max_response_output_tokens** (`Optional[int]`): The maximum number of tokens in the response.

    **Methods:**

    - `reset()`: Resets the builder to its initial state.
    - `set_api_key(api_key: str)`: Sets the API key.
    - `set_model_url(model_url: str)`: Sets the model URL.
    - `set_goal(goal: str)`: Sets the agent's goal.
    - `set_voice(voice: str)`: Sets the voice model.
    - `add_tool(tool: BaseTool)`: Adds a tool to the agent's toolset.
    - `set_tools(tools: List[BaseTool])`: Sets the entire list of tools.
    - `set_temperature(temperature: float)`: Sets the sampling temperature.
    - `set_modalities(modalities: List[str])`: Sets the modalities.
    - `set_input_audio_format(format: str)`: Sets the input audio format.
    - `set_output_audio_format(format: str)`: Sets the output audio format.
    - `set_input_audio_transcription(config: Dict[str, Any])`: Sets input audio transcription config.
    - `set_turn_detection(config: Dict[str, Any])`: Sets turn detection config.
    - `set_tool_choice(choice: str)`: Sets the tool choice strategy.
    - `set_max_response_output_tokens(max_tokens: int)`: Sets max output tokens.
    - `validate()`: Validates all necessary components before building the `VoiceAgent`.
    - `build() -> VoiceAgent`: Builds and returns the fully configured `VoiceAgent`.

    **Usage Example:**

    ```python
    from agent_builder.agents.voice_agent_builder import VoiceAgentBuilder
    from langchain_core.tools import BaseTool

    # Define a custom tool
    class MyCustomTool(BaseTool):
        name = "custom_tool"
        description = "A custom tool for demonstration."
        args = {
            "param1": {"type": "string", "description": "A string parameter."}
        }

        async def ainvoke(self, args):
            param1 = args.get("param1", "")
            # Perform the tool's action
            return {"result": f"Processed {param1}"}

    # Initialize the builder
    builder = VoiceAgentBuilder()

    # Configure the agent
    agent = (
        builder
        .set_api_key("your-api-key")
        .set_model_url("wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01")
        .set_goal("You are an assistant that helps with various tasks.")
        .set_voice("shimmer")
        .add_tool(MyCustomTool())
        .set_temperature(0.7)
        .set_modalities(["text", "audio"])
        .set_input_audio_format("pcm16")
        .set_output_audio_format("pcm16")
        .set_input_audio_transcription({"model": "whisper-1"})
        .set_turn_detection({
            "type": "server_vad",
            "threshold": 0.5,
            "prefix_padding_ms": 300,
            "silence_duration_ms": 500
        })
        .set_tool_choice("auto")
        .set_max_response_output_tokens(1024)
        .build()
    )

    # Now you can use the agent in your application
    ```
    """

    def __init__(self):
        """Initialize the builder and reset all attributes to default values."""
        self.reset()

    def reset(self):
        """Resets the builder to its initial state."""
        self.api_key: Optional[str] = None
        self.model_url: str = (
            "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01"
        )
        self.goal: Optional[str] = None
        self.voice: str = "shimmer"
        self.tools: List[BaseTool] = []
        self.temperature: float = 0.7
        self.modalities: Optional[List[str]] = None
        self.input_audio_format: str = "pcm16"
        self.output_audio_format: str = "pcm16"
        self.input_audio_transcription: Optional[Dict[str, Any]] = None
        self.turn_detection: Optional[Dict[str, Any]] = None
        self.tool_choice: str = "auto"
        self.max_response_output_tokens: Optional[int] = None

    def set_api_key(self, api_key: str) -> "VoiceAgentBuilder":
        """Sets the API key used to authenticate with the OpenAI Realtime API."""
        self.api_key = api_key
        return self

    def set_model_url(self, model_url: str) -> "VoiceAgentBuilder":
        """Sets the model URL for the VoiceAgent."""
        self.model_url = model_url
        return self

    def set_goal(self, goal: str) -> "VoiceAgentBuilder":
        """Sets the primary goal or instructions for the agent."""
        self.goal = goal
        return self

    def set_voice(self, voice: str) -> "VoiceAgentBuilder":
        """Sets the voice model to use for audio output."""
        self.voice = voice
        return self

    def add_tool(self, tool: BaseTool) -> "VoiceAgentBuilder":
        """Adds a tool to the agent's toolset."""
        self.tools.append(tool)
        return self

    def set_tools(self, tools: List[BaseTool]) -> "VoiceAgentBuilder":
        """Sets the entire list of tools for the agent."""
        self.tools = tools
        return self

    def set_temperature(self, temperature: float) -> "VoiceAgentBuilder":
        """Sets the sampling temperature for response generation."""
        self.temperature = temperature
        return self

    def set_modalities(self, modalities: List[str]) -> "VoiceAgentBuilder":
        """Sets the modalities (e.g., ["text", "audio"])."""
        self.modalities = modalities
        return self

    def set_input_audio_format(self, format: str) -> "VoiceAgentBuilder":
        """Sets the input audio format."""
        self.input_audio_format = format
        return self

    def set_output_audio_format(self, format: str) -> "VoiceAgentBuilder":
        """Sets the output audio format."""
        self.output_audio_format = format
        return self

    def set_input_audio_transcription(
        self, config: Dict[str, Any]
    ) -> "VoiceAgentBuilder":
        """Sets the input audio transcription configuration."""
        self.input_audio_transcription = config
        return self

    def set_turn_detection(self, config: Dict[str, Any]) -> "VoiceAgentBuilder":
        """Sets the turn detection configuration."""
        self.turn_detection = config
        return self

    def set_tool_choice(self, choice: str) -> "VoiceAgentBuilder":
        """Sets how the agent chooses tools."""
        self.tool_choice = choice
        return self

    def set_max_response_output_tokens(self, max_tokens: int) -> "VoiceAgentBuilder":
        """Sets the maximum number of tokens in the response."""
        self.max_response_output_tokens = max_tokens
        return self

    def validate(self):
        """Validates that all necessary components are properly set before building the agent."""
        if not self.api_key:
            raise ValueError("API key must be set before building the VoiceAgent.")
        if not self.goal:
            raise ValueError("Goal must be set before building the VoiceAgent.")
        if not isinstance(self.tools, list):
            raise ValueError("Tools must be a list of BaseTool instances.")

    def build(self) -> "VoiceAgent":
        """
        Constructs and returns the fully built `VoiceAgent` based on the configured parameters.

        Returns
        -------
        VoiceAgent
            The fully constructed `VoiceAgent` with the specified configuration.
        """
        self.validate()  # Ensure all required components are set

        # Build the VoiceAgent with all configured parameters
        agent = VoiceAgent(
            api_key=self.api_key,
            model_url=self.model_url,
            goal=self.goal,
            voice=self.voice,
            tools=self.tools,
            temperature=self.temperature,
            modalities=self.modalities,
            input_audio_format=self.input_audio_format,
            output_audio_format=self.output_audio_format,
            input_audio_transcription=self.input_audio_transcription,
            turn_detection=self.turn_detection,
            tool_choice=self.tool_choice,
            max_response_output_tokens=self.max_response_output_tokens,
        )

        # Reset the builder to allow for reuse
        self.reset()

        return agent
