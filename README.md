# poc-rag-cache

RAG (Retrieval-Augmented Generation) + 2-Level Semantic Cache API — ตอบคำถามจากเอกสารภายในองค์กร พร้อมประหยัด LLM API Quota

## Features

- **2-Level Cache** — Redis (exact match) + Onyx Vector DB (semantic similarity)
- **RAG** — อัปโหลดเอกสาร PDF/TXT แล้วถามคำถามจากเนื้อหาได้
- **Smart Chunking** — ตัดเอกสารตาม section + parent-child ลด hallucination
- **Batch Embedding** — embed ทีละ batch พร้อม rate limiting
- **Cache Invalidation** — อัปเดตเอกสารใหม่ cache เก่าถูกล้างอัตโนมัติ
- **Metrics** — ดู hit rate, latency, จำนวนเอกสารผ่าน API

## Quick Start

```bash
# 1. ตั้งค่า
cp .env.example .env
# แก้ไข GEMINI_API_KEY ใน .env

# 2. รัน
docker compose up -d

# 3. ตรวจสอบ
curl http://localhost:8000/health

# 4. อัปโหลดเอกสาร
curl -X POST http://localhost:8000/rag/upload -F "file=@your_doc.pdf"

# 5. ถามคำถาม
curl -X POST http://localhost:8000/rag/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "สรุปเนื้อหาให้หน่อย"}'
```

## Project Structure

```
app/
├── main.py               ← FastAPI entry point
├── config.py             ← centralized configuration
├── cache/
│   ├── redis_cache.py    ← L1: exact match
│   └── semantic_cache.py ← L2: vector similarity
├── rag/
│   ├── routes.py         ← /rag/* endpoints
│   ├── knowledge_base.py ← Onyx knowledge base
│   ├── chunking.py       ← section + parent-child chunking
│   └── metrics.py        ← stats tracking
├── llm/
│   └── gemini.py         ← embedding + generation
└── tests/                ← 29 unit tests
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/ask` | ถาม Gemini ผ่าน 2-level cache |
| `POST` | `/rag/upload` | อัปโหลดเอกสาร PDF/TXT |
| `POST` | `/rag/ask` | ถามจากเอกสาร + cache |
| `GET` | `/rag/sources` | ดูรายชื่อเอกสาร |
| `DELETE` | `/rag/sources/{filename}` | ลบเอกสาร |
| `GET` | `/rag/stats` | ดูสถิติ |
| `POST` | `/rag/invalidate-cache` | ล้าง cache |
| `GET` | `/health` | Health check |

## Tests

```bash
python3 -m pytest app/tests/ -v
```

## Documentation

รายละเอียดเชิงลึก สถาปัตยกรรม, flow, chunking strategy, และแผนพัฒนาต่อ:

**[RAG_OVERVIEW.md](RAG_OVERVIEW.md)**

## Tech Stack

- **Python 3.12** + **FastAPI** + **Uvicorn**
- **Gemini** (text-embedding-004 + 1.5-flash)
- **Onyx** (Vector DB)
- **Redis 7** (Cache)
- **Docker Compose**
