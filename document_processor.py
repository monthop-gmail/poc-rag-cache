import re
from dataclasses import dataclass
from pypdf import PdfReader
from io import BytesIO


@dataclass
class ChunkResult:
    """A child chunk with reference to its parent context."""
    child_text: str
    parent_text: str
    section: str
    chunk_index: int


def extract_text_from_pdf(file_bytes: bytes) -> str:
    reader = PdfReader(BytesIO(file_bytes))
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return "\n".join(pages)


def extract_text_from_txt(file_bytes: bytes) -> str:
    return file_bytes.decode("utf-8")


# ---------------------------------------------------------------------------
# Section-based splitting
# ---------------------------------------------------------------------------

# Patterns that indicate a new section/heading
SECTION_PATTERNS = re.compile(
    r'(?:^|\n)'
    r'(?:'
    r'#{1,6}\s+'           # Markdown headings: ## Heading
    r'|หมวด(?:ที่)?\s*\d+' # Thai sections: หมวด 1, หมวดที่ 2
    r'|ข้อ\s*\d+'          # Thai clauses: ข้อ 1
    r'|บทที่\s*\d+'        # Thai chapters: บทที่ 1
    r'|\d+\.\s+[A-Zก-ฮ]'  # Numbered headings: 1. Policy
    r'|[A-Z][A-Z\s]{3,}$'  # ALL CAPS lines (likely headings)
    r')'
)


def split_into_sections(text: str) -> list[tuple[str, str]]:
    """
    Split text by section headings.
    Returns list of (section_name, section_body).
    """
    text = text.strip()
    if not text:
        return []

    # Find all section heading positions
    matches = list(SECTION_PATTERNS.finditer(text))

    if not matches:
        return [("(untitled)", text)]

    sections = []

    # Text before first heading
    if matches[0].start() > 0:
        pre_text = text[:matches[0].start()].strip()
        if pre_text:
            sections.append(("(intro)", pre_text))

    # Each heading and its body
    for i, match in enumerate(matches):
        # Section name = the heading line itself
        heading_end = text.find("\n", match.start() + 1)
        if heading_end == -1:
            heading_end = len(text)
        section_name = text[match.start():heading_end].strip()

        # Section body = text until next heading
        body_start = heading_end
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[body_start:body_end].strip()

        sections.append((section_name, f"{section_name}\n{body}" if body else section_name))

    return sections


# ---------------------------------------------------------------------------
# Parent-child chunking
# ---------------------------------------------------------------------------

def chunk_document(
    text: str,
    child_size: int = 500,
    parent_size: int = 2000,
    overlap: int = 50,
) -> list[ChunkResult]:
    """
    Smart chunking with parent-child strategy:
    1. Split text into sections by headings
    2. Each section becomes a parent (or split if too long)
    3. Each parent is split into smaller child chunks for search
    4. Child carries reference to its parent for full context
    """
    text = text.strip()
    if not text:
        return []

    sections = split_into_sections(text)
    results = []
    global_index = 0

    for section_name, section_body in sections:
        # Split section into parent-sized chunks
        parents = _chunk_text(section_body, chunk_size=parent_size, overlap=100)

        for parent in parents:
            # Split each parent into child-sized chunks
            children = _chunk_text(parent, chunk_size=child_size, overlap=overlap)

            for child in children:
                results.append(ChunkResult(
                    child_text=child,
                    parent_text=parent,
                    section=section_name,
                    chunk_index=global_index,
                ))
                global_index += 1

    return results


# ---------------------------------------------------------------------------
# Basic chunking (internal helper, also kept for backward compatibility)
# ---------------------------------------------------------------------------

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Simple chunking — kept for backward compatibility. Use chunk_document() for RAG."""
    return _chunk_text(text, chunk_size, overlap)


def _chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split text into overlapping chunks, trying to break at sentence boundaries."""
    text = text.strip()
    if not text:
        return []

    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        if end >= len(text):
            chunks.append(text[start:].strip())
            break

        # Try to break at sentence boundary near the end
        boundary = _find_sentence_boundary(text, end, search_range=80)
        if boundary:
            end = boundary

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = end - overlap

    return chunks


def _find_sentence_boundary(text: str, pos: int, search_range: int = 80) -> int | None:
    """Find the nearest sentence-ending punctuation around pos."""
    search_start = max(0, pos - search_range)
    search_end = min(len(text), pos + search_range)
    region = text[search_start:search_end]

    # Find all sentence boundaries in the search region
    boundaries = [m.end() for m in re.finditer(r'[.!?。]\s', region)]

    if not boundaries:
        return None

    # Pick the boundary closest to the target position
    target = pos - search_start
    closest = min(boundaries, key=lambda b: abs(b - target))
    return search_start + closest
