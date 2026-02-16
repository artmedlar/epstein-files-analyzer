"""Data models for the Epstein Analyzer."""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class DocumentStatus(str, Enum):
    """Status of a document in the pipeline."""
    FOUND = "found"           # URL discovered via search
    DOWNLOADING = "downloading"
    DOWNLOADED = "downloaded"
    EXTRACTING = "extracting"
    EXTRACTED = "extracted"   # Text extracted, PDF deleted
    FAILED = "failed"


class Document(BaseModel):
    """A DOJ document tracked in the database."""
    id: Optional[int] = None
    filename: str                          # e.g. EFTA00026703.pdf
    url: str                               # Full DOJ URL
    data_set: Optional[str] = None         # e.g. "DataSet 9"
    status: DocumentStatus = DocumentStatus.FOUND
    text_path: Optional[str] = None        # Relative path to extracted text
    page_count: Optional[int] = None
    ocr_needed: bool = False
    ocr_confidence: Optional[float] = None
    file_hash: Optional[str] = None
    found_via_query: Optional[str] = None  # Search query that found this doc
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class SearchResult(BaseModel):
    """A result from a DOJ site search."""
    filename: str
    url: str
    data_set: Optional[str] = None
    title: Optional[str] = None


class AnalysisResult(BaseModel):
    """LLM analysis result for a query."""
    id: Optional[int] = None
    query: str                             # The search query / person name
    document_ids: list[int] = []           # Documents analyzed
    analysis_text: str                     # The LLM's analysis
    dates_found: list[str] = []
    people_found: list[str] = []
    locations_found: list[str] = []
    document_types: list[str] = []
    model_used: str = ""
    created_at: datetime = Field(default_factory=datetime.now)


class SearchProgress(BaseModel):
    """Real-time progress update for the UI."""
    stage: str                             # "searching", "downloading", "extracting", "analyzing"
    current: int = 0
    total: int = 0
    message: str = ""
    documents_found: int = 0


class OllamaStatus(BaseModel):
    """Status of the Ollama connection."""
    available: bool = False
    models: list[str] = []
    has_llm: bool = False
    has_embed: bool = False
