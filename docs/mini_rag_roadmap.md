# Mini RAG Pipeline - FastAPI Project Roadmap

## 📋 Project Overview
A lightweight RAG system that ingests PDFs, splits text into chunks, generates embeddings, stores in a vector database, and retrieves relevant context for LLM responses. **Focus: Core RAG mechanism only.**

---

## 🔧 Tech Stack

### Backend
- **Framework**: FastAPI (async Python)
- **PDF Processing**: PyPDF2 or pdfplumber (text extraction)
- **Chunking**: LangChain (semantic splitting) or custom logic
- **Embeddings**: OpenAI API, Ollama (local), or HuggingFace transformers
- **Vector Database**: 
  - **Option A** (Recommended for simplicity): Pinecone (managed, free tier)
  - **Option B** (Local): Milvus (Docker) or FAISS (in-memory, no server)
  - **Option C** (PostgreSQL-based): pgvector extension
- **LLM**: OpenAI (gpt-4-mini) or LiteLLM (multi-provider)
- **Validation**: Pydantic v2
- **Async task handling**: Celery (optional) or simple threading for CPU-bound PDF processing

### Database (optional for metadata)
- SQLite (development) or PostgreSQL (production)
- Store: document metadata, chunk references, upload history

### Deployment
- Docker + Docker Compose
- Uvicorn ASGI server
- Optional: AWS Lambda / Render / Railway for serverless

---

## 📅 Phase Breakdown

### **Phase 1: Project Setup & Architecture (Days 1-2)**

#### 1.1 Initialize FastAPI Project
```
mini-rag/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI app
│   ├── config.py               # Settings (env vars, paths)
│   ├── requirements.txt
│   └── routers/
│       ├── documents.py        # Upload, list documents
│       └── query.py            # Query endpoint
├── core/
│   ├── pdf_processor.py        # PDF text extraction
│   ├── chunker.py              # Text chunking logic
│   ├── embeddings.py           # Embedding generation
│   └── vector_store.py         # Vector DB interface
|   |___ indexer.py  
|    
├── models/
│   ├── schemas.py              # Pydantic models
│   └── db.py                   # SQLAlchemy models (optional)
├── services/
│   └── rag_service.py          # Core RAG orchestration
├── utils/
│   └── logging_config.py
├── .env.example
├── docker-compose.yml
└── README.md
```

#### 1.2 Environment Setup
- Create `.env` with:
  - `OPENAI_API_KEY`
  - `VECTOR_DB_URL` (Pinecone/Milvus connection)
  - `EMBEDDING_MODEL` (e.g., "text-embedding-3-small")
  - `LLM_MODEL` (e.g., "gpt-4-mini")
  - `CHUNK_SIZE`, `CHUNK_OVERLAP`

#### 1.3 Project Structure
- Set up virtual environment
- Install core dependencies (see Phase 2)
- Create simple FastAPI skeleton with health check endpoint

**Deliverable**: Running FastAPI server with `/health` endpoint

---

### **Phase 2: PDF Processing & Chunking (Days 3-4)**

#### 2.1 PDF Text Extraction
```python
# core/pdf_processor.py
class PDFProcessor:
    def extract_text(pdf_path: str) -> str
    def extract_with_metadata(pdf_path: str) -> Dict[str, Any]
```

**Requirements**:
- Handle various PDF types (text, scanned images with OCR optional)
- Return: raw text, page numbers, metadata
- Error handling for corrupted PDFs

**Tool Choice**:
- `pdfplumber`: Best for structured text extraction
- Alternative: `PyPDF2` (simpler) or `PyMuPDF` (faster)

#### 2.2 Text Chunking Strategy
```python
# core/chunker.py
class TextChunker:
    def chunk_text(text: str, chunk_size: int, overlap: int) -> List[Chunk]
    def semantic_chunk(text: str) -> List[Chunk]  # Optional
```

**Strategy**:
- **Simple approach**: Fixed-size chunks (1000 tokens) with 200-token overlap
- **Semantic approach**: Use sentence/paragraph boundaries (LangChain's RecursiveCharacterTextSplitter)
- Store: chunk ID, text, source document, page number

**Key Parameters**:
- `CHUNK_SIZE`: 1000-1500 tokens (≈ 3000-4500 characters)
- `OVERLAP`: 200-300 tokens (for context continuity)
- `MIN_CHUNK_SIZE`: 100 tokens (skip very small chunks)

#### 2.3 Database Schema (if using metadata store)
```sql
-- documents table
CREATE TABLE documents (
    id UUID PRIMARY KEY,
    filename VARCHAR(255),
    uploaded_at TIMESTAMP,
    text_length INT,
    status ENUM('pending', 'processing', 'completed', 'failed')
);

-- chunks table
CREATE TABLE chunks (
    id UUID PRIMARY KEY,
    document_id UUID FOREIGN KEY,
    chunk_index INT,
    text TEXT,
    tokens INT,
    created_at TIMESTAMP
);
```

**Deliverable**: 
- `PDFProcessor` class that extracts text from PDFs
- `TextChunker` class that splits text into chunks
- Unit tests for both

---

### **Phase 3: Embeddings & Vector Store (Days 5-6)**

#### 3.1 Embedding Generation
```python
# core/embeddings.py
class EmbeddingService:
    async def embed_text(text: str) -> List[float]
    async def embed_batch(texts: List[str]) -> List[List[float]]
```

**Embedding Options**:
- **OpenAI** (recommended): `text-embedding-3-small` ($0.02/1M tokens, best quality)
- **Local** (free): 
  - Ollama + `nomic-embed-text` or `sentence-transformers`
  - HuggingFace: `all-MiniLM-L6-v2` (fast, 384-dim)
- **Hybrid**: Use local embeddings for development, OpenAI for production

#### 3.2 Vector Store Setup
```python
# core/vector_store.py
class VectorStore:
    async def add_embeddings(
        embeddings: List[List[float]], 
        metadata: List[Dict]
    ) -> List[str]
    
    async def search(
        query_embedding: List[float], 
        k: int = 5
    ) -> List[Chunk]
    
    async def delete_document(document_id: str)
```

**Vector DB Comparison**:

| Option | Pros | Cons | Setup |
|--------|------|------|-------|
| **Pinecone** | Managed, scalable, free tier | Requires API key, data privacy | API + Python SDK |
| **FAISS** | Fast, in-memory, free | No persistence (need serialization) | `pip install faiss-cpu` |
| **Milvus** | Open-source, persistent, self-hosted | More complex setup | Docker Compose |
| **pgvector** | SQL + vectors, PostgreSQL | Learning curve | PostgreSQL extension |

**Recommendation for learning**: Start with **Pinecone** (easiest) or **FAISS** (full control, local)

#### 3.3 Batch Processing
- Embed chunks in batches (e.g., 100 at a time) to optimize API costs
- Use async batching to avoid rate limits
- Store embedding metadata: chunk_id, source_doc, timestamp

**Deliverable**:
- `EmbeddingService` class
- `VectorStore` interface with implementation (Pinecone/FAISS/Milvus)
- Script to test embedding + retrieval

---

### **Phase 4: Core RAG Service (Days 7-8)**

#### 4.1 RAG Orchestration Service
```python
# services/rag_service.py
class RAGService:
    async def ingest_document(
        file: UploadFile
    ) -> Document
    
    async def query(
        question: str,
        k: int = 5  # top-k chunks
    ) -> RAGResponse
    
    async def delete_document(document_id: str)
```

**Workflow**:
1. **Ingest**:
   - Extract PDF text
   - Chunk the text
   - Generate embeddings for chunks
   - Store in vector DB + metadata DB
   - Return document_id

2. **Query**:
   - Embed user question
   - Search vector DB for top-5 similar chunks
   - Format retrieved chunks as context
   - Send to LLM: `context + question → answer`
   - Return answer + source chunks

#### 4.2 LLM Integration
```python
# services/rag_service.py
class LLMService:
    async def generate_answer(
        question: str,
        context: str,  # Retrieved chunks
        max_tokens: int = 500
    ) -> str
```

**Prompt Engineering** (keep simple):
```
Context:
{context}

Question: {question}

Answer (cite sources):
```

**LLM Options**:
- OpenAI (`gpt-4-mini` or `gpt-3.5-turbo`)
- LiteLLM (multi-provider support)
- Ollama (local, free)

#### 4.3 Response Structure
```python
# models/schemas.py
class RAGResponse(BaseModel):
    answer: str
    source_chunks: List[SourceChunk]
    confidence: float  # Optional: based on relevance scores
    model_used: str
    query_time_ms: float

class SourceChunk(BaseModel):
    text: str
    document_name: str
    page_number: Optional[int]
    relevance_score: float
```

**Deliverable**:
- Full RAG service orchestration
- Integration tests
- Example queries working end-to-end

---

### **Phase 5: FastAPI Endpoints (Days 9-10)**

#### 5.1 Document Management Endpoints
```python
# routers/documents.py

@router.post("/documents/upload")
async def upload_document(file: UploadFile) -> DocumentResponse

@router.get("/documents")
async def list_documents() -> List[DocumentInfo]

@router.get("/documents/{doc_id}")
async def get_document_chunks(doc_id: str) -> List[ChunkInfo]

@router.delete("/documents/{doc_id}")
async def delete_document(doc_id: str) -> DeleteResponse
```

**Features**:
- Validate file type (only PDFs)
- Stream large file uploads
- Return processing status
- Handle errors gracefully

#### 5.2 Query Endpoints
```python
# routers/query.py

@router.post("/query")
async def query_documents(
    question: str,
    k: int = 5,  # top-k chunks
    model: Optional[str] = None
) -> RAGResponse

@router.post("/query/stream")  # Streaming response
async def query_stream(question: str) -> StreamingResponse
```

**Features**:
- Basic query endpoint
- Streaming response for real-time feedback (optional)
- Toggle between models
- Track query metadata

#### 5.3 Utilities & Health
```python
@router.get("/health")
async def health_check() -> HealthStatus

@router.get("/stats")
async def get_stats() -> SystemStats
    # Documents indexed, total chunks, avg query time, etc.
```

**Deliverable**: All endpoints functional with OpenAPI docs

---

### **Phase 6: Error Handling & Logging (Days 11-12)**

#### 6.1 Error Handling
```python
# Custom exceptions
class PDFProcessingError(Exception): pass
class EmbeddingError(Exception): pass
class VectorStoreError(Exception): pass

# Global exception handlers in main.py
@app.exception_handler(PDFProcessingError)
async def pdf_error_handler(request, exc): ...
```

#### 6.2 Logging & Monitoring
```python
# utils/logging_config.py
- Structured logging (JSON format)
- Track: API response times, errors, chunk retrieval metrics
- Log to file + console
- Optional: Send to external service (DataDog, etc.)
```

#### 6.3 Input Validation
- File size limits (e.g., max 50MB)
- Question length limits
- Rate limiting (optional, simple approach with dict counter)

**Deliverable**: Robust error handling, detailed logs

---

### **Phase 7: Testing & Documentation (Days 13-14)**

#### 7.1 Unit Tests
```
tests/
├── test_pdf_processor.py
├── test_chunker.py
├── test_embeddings.py
├── test_vector_store.py
└── test_rag_service.py
```

**Coverage**:
- Happy path + error cases
- Edge cases (empty PDFs, very long documents, special characters)

#### 7.2 Integration Tests
```python
# tests/test_api.py
@pytest.mark.asyncio
async def test_upload_and_query():
    # Upload PDF → Query → Verify answer
```

#### 7.3 Documentation
- **README.md**: Setup, usage examples
- **API docs**: Auto-generated by FastAPI (`/docs`)
- **Code comments**: Docstrings for key functions
- **Architecture diagram**: Visual flow

**Deliverable**: >80% test coverage, comprehensive docs

---

### **Phase 8: Optimization & Deployment (Days 15-16)**

#### 8.1 Performance Optimization
- **Batch embeddings**: Process multiple chunks concurrently
- **Caching**: Cache embedding results (avoid re-embedding same text)
- **Connection pooling**: Reuse vector DB connections
- **Async everything**: Ensure no blocking I/O

#### 8.2 Docker Setup
```dockerfile
# Dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0"]
```

```yaml
# docker-compose.yml (with Milvus example)
version: '3.8'
services:
  fastapi:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - VECTOR_DB_URL=http://milvus:19530
  
  milvus:
    image: milvusdb/milvus:latest
    ports:
      - "19530:19530"
```

#### 8.3 Local Deployment
- Run with Uvicorn: `uvicorn app.main:app --reload`
- Docker: `docker-compose up`
- Test all endpoints

**Deliverable**: Running system, deployable container

---

## 🎯 Key Milestones

| Week | Phase | Milestone |
|------|-------|-----------|
| W1 | 1-2 | ✅ PDF extraction + chunking working |
| W2 | 3-4 | ✅ Embeddings + vector store integrated |
| W2-3 | 5-6 | ✅ Full RAG pipeline end-to-end |
| W3 | 7-8 | ✅ FastAPI endpoints complete |
| W3-4 | 9-10 | ✅ Testing + documentation done |
| W4 | 11 | ✅ Deployed & ready to extend |

---

## 💡 Pro Tips

1. **Start Simple**: Use OpenAI embeddings + Pinecone. Don't optimize prematurely.
2. **Iterate Fast**: Build MVP in Week 1, refine in Weeks 2-3.
3. **Test with Real PDFs**: Use 5-10 sample PDFs early.
4. **Monitor Costs**: Track embedding API costs (OpenAI can add up).
5. **Chunk Size Matters**: Too small = noise, too large = loss of precision. Experiment.
6. **LLM Prompting**: Simple context + question works well. Complex prompts often hurt.
7. **Logging is Your Friend**: Instrument everything early.

---

## 🔗 Essential Dependencies

```txt
fastapi==0.104.1
uvicorn==0.24.0
pydantic==2.4.2
pydantic-settings==2.1.0

# PDF Processing
pdfplumber==0.10.3
PyPDF2==4.0.1

# Embeddings & LLM
openai==1.3.5
litellm==1.0.0  # Optional

# Vector DB (choose one)
pinecone-client==2.2.3  # OR
pymilvus==2.3.0  # OR
faiss-cpu==1.7.4

# Database (optional)
sqlalchemy==2.0.23
psycopg2-binary==2.9.9

# Async & Tasks
httpx==0.25.1
python-multipart==0.0.6

# Testing
pytest==7.4.3
pytest-asyncio==0.21.1

# Utils
python-dotenv==1.0.0
structlog==23.2.0  # Structured logging
```

---

## 📊 Expected System Architecture

```
┌─────────────────────────────────────┐
│        FastAPI Server               │
│  (Async, 8 workers)                 │
└────────────┬────────────────────────┘
             │
    ┌────────┴────────┐
    │                 │
    ▼                 ▼
┌─────────────┐  ┌─────────────┐
│  Document   │  │   Query     │
│  Ingestion  │  │  Processing │
│  Router     │  │  Router     │
└────────┬────┘  └────────┬────┘
         │                │
    ┌────▼────────────────▼───┐
    │   RAG Service Core      │
    │ (Orchestration Logic)   │
    └────┬────────────────┬───┘
         │                │
    ┌────▼────┐      ┌────▼────────┐
    │ PDF     │      │ LLM Service  │
    │Processor│      │(OpenAI/Ollama)
    └────┬────┘      └─────────────┘
         │
    ┌────▼──────────────────────┐
    │ Core Services:            │
    │ • PDF extraction          │
    │ • Text chunking           │
    │ • Embedding generation    │
    │ • Vector search           │
    └────┬──────────────────────┘
         │
    ┌────▼──────────────┐
    │  External Deps:   │
    │ • OpenAI API      │
    │ • Vector DB       │
    │ • Metadata DB     │
    └───────────────────┘
```

---

## ✅ Success Criteria (MVP)

- [ ] Upload PDF → Instant ingestion
- [ ] Ask question → Relevant answer in <2 seconds
- [ ] Source citation: Shows which chunks were used
- [ ] Error handling: No crashes, user-friendly errors
- [ ] API docs: Full OpenAPI documentation
- [ ] Test coverage: >75% of core logic
- [ ] Logging: Can trace any issue
- [ ] Docker: One-command deployment

---

**Next Step**: Start with Phase 1 & 2. Build PDF processor + chunker first. Get data flowing before touching APIs.
