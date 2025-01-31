"""
Utility module for handling audio input and output for the VoiceAgent.

This module provides functions to:
- Capture audio from the microphone.
- Play audio through the speakers.
- Handle async communication using a queue.
"""

import asyncio
import base64
import json

# Attempt to import pyaudio and handle the case where it's missing
try:
    import pyaudio

    AUDIO_ENABLED = True
except ImportError:
    AUDIO_ENABLED = False
    print("Warning: pyaudio is not installed. Audio functionality will be disabled.")


class AudioHandler:
    def __init__(self, frame_size=3200, rate=24000, channels=1):
        """
        Initialize the audio handler with user-defined parameters.

        :param frame_size: Number of audio frames per buffer (lower for lower latency)
        :param rate: Sampling rate in Hz (24kHz recommended)
        :param channels: Number of channels (1 for mono, 2 for stereo)
        """
        self.frame_size = frame_size
        self.rate = rate
        self.channels = channels
        self.output_queue = asyncio.Queue()

        if AUDIO_ENABLED:
            # Initialize PyAudio if available
            self.p = pyaudio.PyAudio()

            # Open Input (Microphone)
            self.stream_in = self.p.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.frame_size,
            )

            # Open Output (Speaker)
            self.stream_out = self.p.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.rate,
                output=True,
                frames_per_buffer=self.frame_size,
            )
        else:
            print(
                "Warning: Audio functionality will be skipped due to missing pyaudio."
            )

    async def audio_input_generator(self):
        """
        Asynchronous generator to capture audio from the microphone and yield as base64-encoded strings.
        """
        if not AUDIO_ENABLED:
            return  # Skip if audio is disabled

        try:
            while True:
                data = self.stream_in.read(self.frame_size, exception_on_overflow=False)
                audio_base64 = base64.b64encode(data).decode("utf-8")
                event_json = json.dumps(
                    {"type": "input_audio_buffer.append", "audio": audio_base64}
                )
                yield event_json
                await asyncio.sleep(0.001)  # Lower sleep time for faster processing
        except asyncio.CancelledError:
            pass

    async def output_handler(self, event_json_str):
        """
        Handles audio output from the VoiceAgent.
        """
        try:
            event = json.loads(event_json_str)
            if event.get("type") == "response.audio.delta":
                audio_bytes = base64.b64decode(event.get("delta", ""))
                await self.output_queue.put(audio_bytes)
        except Exception as e:
            print(f"Error processing agent output: {e}")

    async def play_audio(self):
        """
        Plays audio smoothly from the output queue with pre-buffering.
        """
        if not AUDIO_ENABLED:
            return  # Skip if audio is disabled

        try:
            pre_buffer = [
                await self.output_queue.get() for _ in range(3)
            ]  # Pre-buffer a few chunks
            for audio_bytes in pre_buffer:
                self.stream_out.write(audio_bytes)

            while True:
                audio_bytes = await self.output_queue.get()
                self.stream_out.write(audio_bytes)
        except asyncio.CancelledError:
            pass

    def close(self):
        """
        Closes all audio streams and terminates PyAudio.
        """
        if AUDIO_ENABLED:
            self.stream_in.stop_stream()
            self.stream_in.close()
            self.stream_out.stop_stream()
            self.stream_out.close()
            self.p.terminate()
        else:
            print("Warning: No audio resources to close due to missing pyaudio.")
