from typing import Any

class TextChunker:
    """Split text into overlapping chunks"""
    
    def __init__(self, chunk_size: int =1000, overlap: int = 200) -> None:
        
        if chunk_size <= 0:
            raise ValueError(
                "Chunk size must be greater than zero."
            )
            
        if overlap < 0:
            raise ValueError(
                "Overlap must be greater than zero."
            )
            
        if overlap >= chunk_size:
            raise ValueError("Overlap must be smaller than chunk sizze")
        
        self.chunk_size = chunk_size
        self.overlap = overlap
        
    def chunk_text(
        self,
        text: str,
        page_number: int | None = None,
    ) -> list[dict[str, Any]]:
        """Split text into overlapping character-based chunks."""
        cleaned_text = text.strip()

        if not cleaned_text:
            return []
        
        chunks: list[dict[str, Any]] = []
        step = self.chunk_size - self.overlap
        start = 0
        chunk_index = 0
        
        while start < len(cleaned_text):
            chunk_text = cleaned_text[start : start + self.chunk_size].strip()

            if chunk_text:
                chunks.append(
                    {
                        "chunk_id": f"chunk-{chunk_index}",
                        "chunk_index": chunk_index,
                        "text": chunk_text,
                        "page_number": page_number,
                    }
                )

            chunk_index += 1
            start += step

        return chunks