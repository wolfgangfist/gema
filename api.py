from typing import List, Dict, Optional
from dataclasses import dataclass, field

class RequestOperation:
    """Enum for request operations."""
    CLEAR = "clear"
    EOS = "eos"

class ResponseType:
    """Enum for response types."""
    ERR = "error"
    AUDIO_CHUNK = "chunk"
    METADATA = "timestamps"
    ACK = "ack"

@dataclass
class Request:
    text: Optional[str] = None
    contextId: Optional[str] = None
    operation: Optional[RequestOperation] = None

@dataclass
class Timestamps:
    words: List[str] = field(default_factory=list)
    start: List[float] = field(default_factory=list)
    end: List[float] = field(default_factory=list)

@dataclass
class Response:
    type: ResponseType = None
    contextId: Optional[str] = None
    data: Optional[str] = None
    metadata: Optional[Dict] = None
    timestamps: Optional[Timestamps] = None
    error: Optional[str] = None