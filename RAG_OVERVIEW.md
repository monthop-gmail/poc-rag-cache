# POC: RAG + Semantic Cache System

ระบบ RAG (Retrieval-Augmented Generation) พร้อม 2-Level Cache สำหรับประหยัด API Quota และตอบคำถามจากเอกสารภายในองค์กร

---

## สารบัญ

- [ภาพรวมระบบ](#ภาพรวมระบบ)
- [สถาปัตยกรรม](#สถาปัตยกรรม)
- [Flow การทำงาน](#flow-การทำงาน)
- [โครงสร้างโปรเจค](#โครงสร้างโปรเจค)
- [API Endpoints](#api-endpoints)
- [การ Chunk เอกสาร](#การ-chunk-เอกสาร)
- [Cache Strategy](#cache-strategy)
- [Tech Stack](#tech-stack)
- [การตั้งค่า](#การตั้งค่า)
- [การรัน](#การรัน)
- [ตัวอย่างการใช้งาน](#ตัวอย่างการใช้งาน)
- [Metrics & Monitoring](#metrics--monitoring)
- [แนวทางพัฒนาต่อ](#แนวทางพัฒนาต่อ)

---

## ภาพรวมระบบ

### ปัญหา
- เรียก LLM API ทุกครั้ง = เสีย quota/เงิน ทุกครั้ง
- คำถามซ้ำ/คล้ายกัน ไม่ควรต้องเรียก API ซ้ำ
- ต้องการตอบคำถามจากเอกสารภายในองค์กร (HR policy, คู่มือ, etc.)

### แนวทาง
ระบบ **3 ชั้น** ที่ทำงานร่วมกัน:

| ชั้น | เทคโนโลยี | หน้าที่ | ความเร็ว |
|------|-----------|---------|----------|
| **L1** | Redis | Exact match cache | ~1ms |
| **L2** | Onyx (Vector DB) | Semantic cache — คำถามคล้ายกัน | ~10-50ms |
| **RAG** | Onyx + Gemini | ค้นเอกสาร + สรุปคำตอบ | ~500-2000ms |

---

## สถาปัตยกรรม

```
                         ┌─────────────────────────────────────────────┐
                         │              FastAPI Application            │
                         │                                             │
  User ──POST /ask──────►│  ┌─────────┐    ┌──────────┐    ┌────────┐ │
                         │  │  Redis   │    │  Onyx    │    │ Gemini │ │
  User ──POST /rag/ask──►│  │  (L1)   │    │  (L2)    │    │  API   │ │
                         │  └────┬────┘    └────┬─────┘    └───┬────┘ │
  User ──POST /rag/upload►│      │              │              │      │
                         └──────┼──────────────┼──────────────┼──────┘
                                │              │              │
                         ┌──────▼──────┐ ┌─────▼──────┐      │
                         │ Redis 7     │ │ Onyx       │      │
                         │ Alpine      │ │ Vector DB  │      │
                         │             │ │            │      │
                         │ exact cache │ │ 2 collections:    │
                         │             │ │ - semantic_cache   │
                         │             │ │ - knowledge_base   │
                         └─────────────┘ └────────────┘      │
                                                              │
                                                    ┌─────────▼────────┐
                                                    │ Google Gemini    │
                                                    │ - embedding-004  │
                                                    │ - 1.5-flash     │
                                                    └──────────────────┘
```

---

## Flow การทำงาน

### 1. POST /ask (Cache เท่านั้น — ไม่มี RAG)

```
คำถาม: "Python คืออะไร"
    │
    ▼
┌─ L1: Redis ─────────────────┐
│ key ตรงเป๊ะ? ───── Yes ────►│ return ⚡ (~1ms)
└──────────┬──────────────────┘
           │ No
           ▼
┌─ L2: Semantic Cache ────────┐
│ similarity ≥ 0.92? ── Yes ─►│ return + sync L1 🔍 (~10-50ms)
└──────────┬──────────────────┘
           │ No
           ▼
┌─ Gemini API ────────────────┐
│ เรียก LLM ──────────────────│ return 💰 (~500-2000ms)
│ ★ เก็บ L1 + L2              │
└─────────────────────────────┘
```

### 2. POST /rag/ask (Cache + RAG จากเอกสาร)

```
คำถาม: "นโยบายลาป่วยเป็นอย่างไร"
    │
    ▼
┌─ L1: Redis ──────────────────────────────┐
│ exact match? ─────── Yes ───────────────►│ return ⚡
└──────────┬───────────────────────────────┘
           │ No
           ▼
┌─ L2: Semantic Cache ─────────────────────┐
│ similarity ≥ 0.92? ─── Yes ────────────►│ return 🔍
└──────────┬───────────────────────────────┘
           │ No
           ▼
┌─ RAG: Knowledge Base ────────────────────┐
│ ค้นเอกสาร (similarity ≥ 0.7, top 5)     │
│                                          │
│ เจอ? ──► สร้าง prompt:                   │
│          "จากข้อมูลนี้ ตอบคำถาม:         │
│           [hr_policy.pdf | หมวด 1]:      │
│           พนักงานลาป่วยได้ 30 วัน..."     │
│                                          │
│ ไม่เจอ? ──► ใช้คำถามตรงๆ                 │
└──────────┬───────────────────────────────┘
           │
           ▼
┌─ Gemini API ─────────────────────────────┐
│ สรุปคำตอบจากเอกสาร                       │
│ ★ เก็บ L1 + L2 (ครั้งหน้าไม่ต้อง RAG)    │
└──────────────────────────────────────────┘
```

### 3. POST /rag/upload (อัปโหลดเอกสาร)

```
PDF / TXT
    │
    ▼
┌─ Extract Text ──────────────────────────┐
│ PDF → pypdf → plain text                │
│ TXT → UTF-8 decode                      │
└──────────┬──────────────────────────────┘
           │
           ▼
┌─ Section Splitting ─────────────────────┐
│ ตัดตาม heading:                          │
│ ## Heading, หมวด 1, ข้อ 1, บทที่ 1      │
└──────────┬──────────────────────────────┘
           │
           ▼
┌─ Parent-Child Chunking ─────────────────┐
│ Parent (2000 chars): บริบทครบถ้วน        │
│   └── Child (500 chars): ใช้ค้นหา       │
│   └── Child (500 chars): ใช้ค้นหา       │
└──────────┬──────────────────────────────┘
           │
           ▼
┌─ Batch Embedding ───────────────────────┐
│ Gemini text-embedding-004               │
│ ทีละ 20 chunks + rate limiting          │
└──────────┬──────────────────────────────┘
           │
           ▼
┌─ Store in Onyx (knowledge_base) ────────┐
│ vector + child_text + parent_text       │
│ + section + source_filename             │
└─────────────────────────────────────────┘
```

---

## โครงสร้างโปรเจค

```
app/
├── main.py                ← FastAPI entry point, /ask + /health
├── config.py              ← ค่า config ทุกอย่างอยู่ที่เดียว
│
├── cache/                 ← ระบบ Cache 2 ชั้น
│   ├── redis_cache.py     ← L1: Exact match (Redis)
│   └── semantic_cache.py  ← L2: Semantic similarity (Onyx)
│
├── rag/                   ← ระบบ RAG
│   ├── routes.py          ← /rag/* endpoints ทั้งหมด
│   ├── knowledge_base.py  ← จัดการ Onyx knowledge_base collection
│   ├── chunking.py        ← ตัดเอกสาร (section + parent-child)
│   └── metrics.py         ← เก็บสถิติ hit rate, latency
│
├── llm/                   ← LLM Integration
│   └── gemini.py          ← Embedding + Generation
│
└── tests/                 ← Unit Tests (29 tests)
    ├── test_cache.py
    └── test_rag.py

Dockerfile                 ← Python 3.12 + uvicorn
docker-compose.yml         ← 3 services: app, onyx, redis
requirements.txt           ← dependencies
.env.example               ← ตัวอย่างค่า environment
```

### แต่ละ module ทำอะไร

| Module | ไฟล์ | หน้าที่ |
|--------|------|---------|
| **config** | `config.py` | รวม env vars, thresholds, ค่า default ทุกอย่าง |
| **cache** | `redis_cache.py` | L1 — get/set ด้วย query string ตรงๆ |
| | `semantic_cache.py` | L2 — upsert/query ด้วย vector, clear_all |
| **rag** | `chunking.py` | แปลง PDF/TXT → sections → parent-child chunks |
| | `knowledge_base.py` | เก็บ/ค้น chunks ใน Onyx, deduplicate parents |
| | `routes.py` | API endpoints: upload, ask, sources, stats, invalidate |
| | `metrics.py` | นับ queries, hits, latency แบบ in-memory |
| **llm** | `gemini.py` | Embedding (single + batch), text generation |

---

## API Endpoints

### Cache Endpoints (ไม่ใช้ RAG)

| Method | Path | คำอธิบาย |
|--------|------|----------|
| `POST` | `/ask` | ถาม Gemini ผ่าน 2-level cache |
| `GET` | `/health` | Health check |

### RAG Endpoints

| Method | Path | คำอธิบาย |
|--------|------|----------|
| `POST` | `/rag/upload` | อัปโหลดเอกสาร PDF/TXT เข้า knowledge base |
| `POST` | `/rag/ask` | ถามจากเอกสาร + cache |
| `GET` | `/rag/sources` | ดูรายชื่อเอกสารที่อัปโหลด |
| `DELETE` | `/rag/sources/{filename}` | ลบเอกสาร + invalidate cache |
| `GET` | `/rag/stats` | ดูสถิติ hit rate, latency |
| `POST` | `/rag/invalidate-cache` | ล้าง cache ทั้งหมด (manual) |

### Request/Response ตัวอย่าง

**POST /rag/upload**
```bash
curl -X POST http://localhost:8000/rag/upload \
  -F "file=@hr_policy.pdf" \
  -F "child_size=500" \
  -F "parent_size=2000" \
  -F "overlap=50"
```
```json
{"filename": "hr_policy.pdf", "chunks": 42, "message": "Ingested 42 chunks from hr_policy.pdf"}
```

**POST /rag/ask**
```bash
curl -X POST http://localhost:8000/rag/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "นโยบายลาป่วยเป็นอย่างไร", "top_k": 5}'
```
```json
{
  "response": "พนักงานลาป่วยได้ 30 วันต่อปี แต่ถ้าทำงานไม่ถึง 1 ปี ลาได้ไม่เกิน 15 วัน",
  "cache_level": "MISS (RAG)",
  "latency_ms": 1250.5,
  "saved_api_call": false,
  "sources": ["hr_policy.pdf"]
}
```

**GET /rag/stats**
```json
{
  "total_queries": 150,
  "l1_hits": 80,
  "l2_hits": 30,
  "rag_hits": 25,
  "cache_misses": 15,
  "cache_hit_rate": 73.33,
  "avg_latency_ms": 45.2,
  "total_documents": 5,
  "total_chunks_ingested": 210
}
```

---

## การ Chunk เอกสาร

### ทำไมไม่ตัดแบบธรรมดา?

ตัดทุก 500 ตัวอักษรแบบสุ่มสี่สุ่มห้า อาจทำให้ **LLM หลอน (hallucinate)**:

```
เอกสาร: "ลาป่วยได้ 30 วัน แต่ถ้าทำงานไม่ถึง 1 ปี ลาได้ 15 วัน"

❌ ตัดแบบธรรมดา:
   chunk 1: "ลาป่วยได้ 30 วัน"           ← ข้อมูลไม่ครบ!
   chunk 2: "แต่ถ้าทำงานไม่ถึง 1 ปี..."   ← เงื่อนไขหลุดไปอีก chunk
   → LLM ตอบ "30 วัน" โดยไม่มีเงื่อนไข = ผิด!
```

### วิธีที่ใช้: Section-Based + Parent-Child

```
✅ ขั้นตอนที่ 1: ตัดตาม Section
   เอกสาร → [หมวด 1: การลา] [หมวด 2: สวัสดิการ] [หมวด 3: ...]

✅ ขั้นตอนที่ 2: Parent-Child Chunking
   แต่ละ section:
   ┌─ Parent (2000 chars): "ลาป่วยได้ 30 วัน แต่ถ้าไม่ถึง 1 ปี ลาได้ 15 วัน"
   │     ← บริบทครบ ส่งให้ LLM
   ├── Child (500 chars): "ลาป่วยได้ 30 วัน"
   │     ← เล็ก ใช้ค้นหา (vector search)
   └── Child (500 chars): "ถ้าไม่ถึง 1 ปี ลาได้ 15 วัน"
         ← เล็ก ใช้ค้นหา

✅ ขั้นตอนที่ 3: ค้นเจอ child → ส่ง parent ให้ LLM
   → เห็นเงื่อนไขครบ → ตอบถูก!
```

### รองรับ Heading ภาษาไทยและ Markdown

```
## Heading          ← Markdown
หมวด 1 / หมวดที่ 1  ← Thai section
ข้อ 1               ← Thai clause
บทที่ 1             ← Thai chapter
1. Title            ← Numbered heading
ALL CAPS LINE       ← All-caps heading
```

### ปรับค่าได้ตามประเภทเอกสาร

| ประเภท | child_size | parent_size | เหตุผล |
|--------|-----------|-------------|---------|
| HR policy (สั้น) | 300 | 1500 | ข้อความกระชับ |
| Legal document (ยาว) | 500 | 3000 | ต้องการบริบทมาก |
| FAQ | 200 | 800 | แต่ละคำถามสั้น |

---

## Cache Strategy

### Onyx ใช้ 2 Collection แยกกัน

```
Onyx Vector DB
│
├── semantic_cache              ← Cache คำถาม-คำตอบ
│   ├── id: query string
│   ├── vector: embedding ของคำถาม
│   ├── payload: {query, response}
│   ├── threshold: 0.92 (เข้มงวด)
│   └── เขียนโดย: /ask, /rag/ask (ตอน cache miss)
│
└── knowledge_base              ← เอกสารจริง
    ├── id: filename_chunkIndex
    ├── vector: embedding ของ child chunk
    ├── payload: {child_text, parent_text, section, source_filename}
    ├── threshold: 0.7 (กว้าง — หาเอกสารที่เกี่ยวข้อง)
    └── เขียนโดย: /rag/upload
```

### Cache Invalidation

ป้องกันปัญหา cache เก่าตอบผิดหลังอัปเดตเอกสาร:

| เหตุการณ์ | สิ่งที่เกิดขึ้น |
|-----------|----------------|
| Re-upload เอกสารเดิม | ลบ chunks เก่า + ล้าง cache อัตโนมัติ |
| ลบเอกสาร | ลบ chunks + ล้าง cache อัตโนมัติ |
| Manual invalidate | `POST /rag/invalidate-cache` |

### Batch Embedding

ป้องกัน rate limit ตอนอัปโหลดเอกสารใหญ่:

```
PDF 100 หน้า → 400 chunks → 400 embedding calls

❌ ทีละ 1: อาจโดน rate limit
✅ ทีละ batch (20): 20 calls ต่อ batch + หน่วงเวลาระหว่าง batch
```

ตั้งค่าได้ผ่าน environment:
- `EMBEDDING_BATCH_SIZE=20` (จำนวน chunks ต่อ batch)
- `EMBEDDING_RATE_LIMIT_DELAY=0.5` (วินาทีระหว่าง batch)

---

## Tech Stack

| Component | Technology | ทำไมถึงเลือก |
|-----------|-----------|-------------|
| **API** | FastAPI + Uvicorn | async, เร็ว, auto-docs (Swagger) |
| **LLM** | Gemini 1.5 Flash | เร็ว, ถูก, embedding 768d |
| **Embedding** | text-embedding-004 | 768 dimensions, รองรับไทย |
| **Vector DB** | Onyx | Cosine similarity, หลาย collection |
| **Cache** | Redis 7 | In-memory, ไม่เกิน 1ms |
| **PDF** | pypdf | Pure Python, ไม่ต้อง system deps |
| **Container** | Docker Compose | 3 services, deploy ง่าย |

---

## การตั้งค่า

### Environment Variables (.env)

```env
# Required
GEMINI_API_KEY=your-api-key-here

# Optional (มีค่า default)
ONYX_HOST=onyx
ONYX_PORT=8080
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_CACHE_EXPIRE=3600
EMBEDDING_BATCH_SIZE=20
EMBEDDING_RATE_LIMIT_DELAY=0.5
```

### Thresholds (ปรับใน config.py)

| ค่า | Default | ความหมาย |
|-----|---------|----------|
| `CACHE_THRESHOLD` | 0.92 | Semantic cache — ต้องคล้ายมาก |
| `RAG_THRESHOLD` | 0.7 | Knowledge base — กว้างกว่า |
| `RAG_TOP_K` | 5 | จำนวน chunks สูงสุดที่ดึงมา |

---

## การรัน

### Docker Compose (แนะนำ)

```bash
# 1. สร้าง .env
cp .env.example .env
# แก้ไข GEMINI_API_KEY

# 2. Start ทุก service
docker compose up -d

# 3. ตรวจสอบ
curl http://localhost:8000/health
# {"status": "healthy"}
```

### Run Tests

```bash
python3 -m pytest app/tests/ -v
# 29 passed
```

---

## ตัวอย่างการใช้งาน

### Scenario: ระบบ HR Bot

```bash
# 1. อัปโหลดเอกสาร HR
curl -X POST http://localhost:8000/rag/upload -F "file=@hr_policy.pdf"
curl -X POST http://localhost:8000/rag/upload -F "file=@leave_rules.pdf"

# 2. ถามคำถาม (ครั้งแรก — RAG + Gemini)
curl -X POST http://localhost:8000/rag/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "ลาป่วยได้กี่วัน"}'
# → cache_level: "MISS (RAG)", sources: ["hr_policy.pdf"]

# 3. ถามซ้ำ (ครั้งที่ 2 — Redis cache hit)
curl -X POST http://localhost:8000/rag/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "ลาป่วยได้กี่วัน"}'
# → cache_level: "L1 (Redis)" ⚡

# 4. ถามคล้ายๆ (Semantic cache hit)
curl -X POST http://localhost:8000/rag/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "วันลาป่วยมีกี่วันครับ"}'
# → cache_level: "L2 (Semantic Cache)" 🔍

# 5. ดูสถิติ
curl http://localhost:8000/rag/stats
# → cache_hit_rate: 66.67%

# 6. อัปเดตเอกสาร (cache ถูก invalidate อัตโนมัติ)
curl -X POST http://localhost:8000/rag/upload -F "file=@hr_policy.pdf"
```

---

## Metrics & Monitoring

### GET /rag/stats

| Metric | ความหมาย | ดีถ้า |
|--------|----------|------|
| `cache_hit_rate` | % ที่ตอบจาก cache | สูง = ประหยัด quota |
| `l1_hits` | Redis exact match | สูง = คำถามซ้ำเยอะ |
| `l2_hits` | Semantic cache | สูง = chunking/embedding ดี |
| `rag_hits` | ตอบจากเอกสาร | สูง = knowledge base มีข้อมูลครบ |
| `cache_misses` | ต้องเรียก Gemini ตรง | ต่ำ = ดี |
| `avg_latency_ms` | เวลาเฉลี่ยต่อ query | ต่ำ = เร็ว |

---

## แนวทางพัฒนาต่อ

### ระยะสั้น
- [ ] เพิ่ม MCP Server สำหรับเชื่อม Claude Code / AI Agent
- [ ] เพิ่ม authentication (API key / JWT)
- [ ] Persistent metrics (ตอนนี้เป็น in-memory หาย ตอน restart)
- [ ] รองรับไฟล์ .docx, .csv

### ระยะกลาง
- [ ] Web UI สำหรับ upload เอกสาร + ถามคำถาม
- [ ] Cache invalidation แบบ selective (ล้างเฉพาะ cache ที่เกี่ยวกับเอกสารที่เปลี่ยน)
- [ ] Multi-tenant support (แยก knowledge base ต่อทีม)
- [ ] Evaluation pipeline วัดคุณภาพคำตอบ

### ระยะยาว
- [ ] รองรับหลาย LLM (Claude, GPT, Llama)
- [ ] Hybrid search (vector + keyword)
- [ ] Streaming response
- [ ] Auto-reindex เมื่อเอกสารเปลี่ยน (file watcher)
