# Event types constants
SESSION_UPDATE = "session.update"
RESPONSE_CREATE = "response.create"
RESPONSE_CANCEL = "response.cancel"
CONVERSATION_ITEM_CREATE = "conversation.item.create"
INPUT_AUDIO_BUFFER_APPEND = "input_audio_buffer.append"
RESPONSE_AUDIO_DELTA = "response.audio.delta"
RESPONSE_AUDIO_DONE = "response.audio.done"
RESPONSE_FUNCTION_CALL_ARGUMENTS_DONE = "response.function_call_arguments.done"
INPUT_AUDIO_BUFFER_SPEECH_STARTED = "input_audio_buffer.speech_started"
ERROR = "error"
INPUT_AUDIO_TRANSCRIPTION_COMPLETED = (
    "conversation.item.input_audio_transcription.completed"
)
RESPONSE_AUDIO_TRANSCRIPT_DONE = "response.audio_transcript.done"

# Events to ignore during processing
EVENTS_TO_IGNORE = {
    "response.function_call_arguments.delta",
    "rate_limits.updated",
    "response.created",
    "response.content_part.added",
    "response.content_part.done",
    "conversation.item.created",
    "input_audio_buffer.cleared",
    "conversation.item.truncated",
    "conversation.item.deleted",
    "response.done",
    "response.output_item.done",
}
