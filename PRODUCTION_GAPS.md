# Production Gaps & Recommendations

สิ่งที่ต้องแก้ก่อนนำ POC นี้ไปใช้งานจริง เรียงตามความสำคัญ

---

## สารบัญ

- [สถานะปัจจุบัน](#สถานะปัจจุบัน)
- [1. Vector DB — เปลี่ยนจาก Onyx](#1-vector-db--เปลี่ยนจาก-onyx)
- [2. Embedding — ลด API dependency](#2-embedding--ลด-api-dependency)
- [3. Authentication — ป้องกัน API](#3-authentication--ป้องกัน-api)
- [4. Cache Invalidation — แม่นยำขึ้น](#4-cache-invalidation--แม่นยำขึ้น)
- [5. Metrics — Persistent storage](#5-metrics--persistent-storage)
- [6. Semantic Cache — คุ้มจริงไหม](#6-semantic-cache--คุ้มจริงไหม)
- [Migration Plan](#migration-plan)

---

## สถานะปัจจุบัน

| ด้าน | POC | Production ต้องการ | Gap |
|------|-----|-------------------|-----|
| Vector DB | Onyx (ไม่มั่นใจว่า stable) | Qdrant / ChromaDB / pgvector | สูง |
| Embedding | Gemini API ทุกครั้ง | Local + API mix | สูง |
| Auth | ไม่มี | API key / JWT | สูง |
| Cache Invalidation | ล้างทั้งหมด | ล้างเฉพาะที่เกี่ยวข้อง | กลาง |
| Metrics | In-memory (หายตอน restart) | Redis / Prometheus | กลาง |
| Semantic Cache ROI | ยังไม่ได้วัด | ต้องพิสูจน์ว่าคุ้ม | กลาง |
| Error handling | Basic try/catch | Retry, circuit breaker | ต่ำ |
| Logging | Console only | Structured logging (JSON) | ต่ำ |

---

## 1. Vector DB — เปลี่ยนจาก Onyx

### ปัญหา

`onyx-database` ไม่ใช่ package ที่มี community ใหญ่หรือ production track record ชัดเจน ถ้า library มีบั๊กหรือหยุดพัฒนา ระบบทั้งหมดจะพัง

### ทางเลือก

| Vector DB | ข้อดี | ข้อเสีย | เหมาะกับ |
|-----------|------|---------|----------|
| **Qdrant** | เร็ว, filter ดี, gRPC, Docker ready, community ใหญ่ | ต้อง host เอง | Production ทั่วไป |
| **ChromaDB** | ง่ายสุด, embed-in-process, ไม่ต้อง server แยก | ช้าเมื่อ data เยอะ | POC / data น้อย |
| **pgvector** | ใช้ Postgres ที่มีอยู่, SQL query ได้ | ช้ากว่า dedicated vector DB | ทีมที่ใช้ PG อยู่แล้ว |
| **Weaviate** | Hybrid search (vector + keyword), GraphQL | ซับซ้อน, RAM เยอะ | ต้องการ hybrid search |
| **Pinecone** | Fully managed, scale อัตโนมัติ | ค่าใช้จ่าย, vendor lock-in | ไม่อยาก manage infra |

### แนะนำ: Qdrant

```
เหตุผล:
- Open source, community ใหญ่, update บ่อย
- Docker image พร้อมใช้ (เปลี่ยน docker-compose แค่ 5 บรรทัด)
- Python client เสถียร (qdrant-client)
- รองรับ filter by payload (ใช้ selective invalidation ได้)
- Performance ดีกว่า ChromaDB เมื่อ data เยอะ
```

### ผลกระทบต่อ code

```
ไฟล์ที่ต้องแก้:
├── app/cache/semantic_cache.py  ← เปลี่ยน OnyxClient → QdrantClient
├── app/rag/knowledge_base.py   ← เปลี่ยน OnyxClient → QdrantClient
├── app/config.py               ← เปลี่ยน ONYX_* → QDRANT_*
├── docker-compose.yml           ← เปลี่ยน onyx image → qdrant image
└── requirements.txt             ← เปลี่ยน onyx-database → qdrant-client

ไฟล์ที่ไม่ต้องแก้:
├── app/rag/routes.py            ← ไม่ยุ่งกับ DB โดยตรง
├── app/rag/chunking.py          ← ไม่เกี่ยว
├── app/llm/gemini.py            ← ไม่เกี่ยว
└── app/main.py                  ← ไม่เกี่ยว (ใช้ผ่าน class)
```

**Effort: 1-2 วัน** — เปลี่ยน client library, ปรับ API calls, ทดสอบ

---

## 2. Embedding — ลด API dependency

### ปัญหา

```
ปัจจุบัน:
ทุก operation → เรียก Gemini Embedding API
├── /ask          → 1 API call (embed คำถาม)
├── /rag/ask      → 1 API call (embed คำถาม)
└── /rag/upload   → N API calls (embed ทุก chunk)

ความเสี่ยง:
1. Gemini API ล่ม → ระบบทั้งหมดล่ม (แม้แต่ cache ก็ใช้ไม่ได้)
2. Rate limit → upload เอกสารใหญ่ไม่ได้
3. ค่าใช้จ่าย → embedding ไม่ฟรี
4. Latency → ทุก query ต้องรอ network round trip
```

### แนวทาง: Local Embedding + API Generation

```
เปลี่ยนเป็น:
├── Embedding: sentence-transformers (local, ฟรี, เร็ว)
│   └── Model: all-MiniLM-L6-v2 (384d) หรือ multilingual-e5-large (1024d)
│
└── Generation: Gemini API (เรียกเฉพาะตอน cache miss)
```

### เปรียบเทียบ

| | Gemini API Embedding | Local Embedding |
|---|---|---|
| **ความเร็ว** | 100-300ms (network) | 5-20ms (local) |
| **ค่าใช้จ่าย** | จ่ายตาม usage | ฟรี |
| **Availability** | ขึ้นกับ Google | ขึ้นกับเครื่องเรา |
| **ภาษาไทย** | รองรับ | ต้องเลือก model ที่รองรับ |
| **Dimension** | 768 | 384-1024 (แล้วแต่ model) |

### Model ที่แนะนำ (รองรับภาษาไทย)

| Model | Dimension | Size | ภาษาไทย |
|-------|-----------|------|---------|
| `multilingual-e5-large` | 1024 | 2.2 GB | ดี |
| `paraphrase-multilingual-MiniLM-L12-v2` | 384 | 470 MB | ดี |
| `all-MiniLM-L6-v2` | 384 | 80 MB | พอใช้ (อังกฤษเด่น) |

### ผลกระทบต่อ code

```
ไฟล์ที่ต้องแก้:
├── app/llm/gemini.py         ← แยก embedding ออกไป
├── app/llm/embedder.py       ← ใหม่: local embedding class
├── app/config.py             ← เพิ่ม EMBEDDING_MODEL_NAME
├── requirements.txt          ← เพิ่ม sentence-transformers
└── Dockerfile                ← เพิ่ม torch (image จะใหญ่ขึ้น ~1-2 GB)
```

**Effort: 1-2 วัน** — แต่ต้องทดสอบคุณภาพ embedding ภาษาไทย

**ข้อควรระวัง:** Docker image จะใหญ่ขึ้นมากเพราะ PyTorch ถ้า deploy บน resource จำกัด ให้พิจารณา ONNX runtime แทน

---

## 3. Authentication — ป้องกัน API

### ปัญหา

```
ปัจจุบัน: ใครก็เรียก API ได้
├── ถ้า deploy บน public network → ใครก็ถามได้ → quota Gemini หมด
├── ไม่รู้ว่าใครเป็นคนถาม → debug ยาก
└── ไม่มี rate limit per user → user คนเดียวกิน resource ทั้งหมด
```

### แนวทาง (เรียงจากง่ายไปยาก)

**ระดับ 1: API Key (ง่ายสุด)**
```python
# Header: X-API-Key: secret-key-here
# เหมาะสำหรับ internal service
```

**ระดับ 2: JWT Token**
```python
# Header: Authorization: Bearer eyJ...
# เหมาะสำหรับ multi-user, มี expiration
```

**ระดับ 3: OAuth2 + RBAC**
```python
# Role-based: admin ได้ upload, user ได้แค่ ask
# เหมาะสำหรับ production จริงจัง
```

### แนะนำเริ่มที่ระดับ 1

**Effort: ครึ่งวัน** — FastAPI มี dependency injection สำหรับ API key อยู่แล้ว

---

## 4. Cache Invalidation — แม่นยำขึ้น

### ปัญหา

```
ปัจจุบัน:
Re-upload hr_policy.pdf → ล้าง cache ทั้งหมด

ผลกระทบ:
- คำถามเรื่อง IT support ที่ไม่เกี่ยวกับ HR ก็ถูกล้าง
- ต้องสร้าง cache ใหม่ทั้งหมด → เสีย quota + latency สูง
```

### แนวทาง: Selective Invalidation

```
Re-upload hr_policy.pdf → ล้างเฉพาะ cache ที่ source มาจาก hr_policy.pdf

วิธีทำ:
1. ตอน cache ใน semantic_cache → เก็บ sources ที่ใช้ตอบไว้ด้วย
2. ตอน invalidate → ค้นหา cache entries ที่มี source = hr_policy.pdf → ลบเฉพาะนั้น
```

### ผลกระทบต่อ code

```
ไฟล์ที่ต้องแก้:
├── app/cache/semantic_cache.py  ← เพิ่ม sources ใน payload, เพิ่ม delete_by_source()
├── app/rag/routes.py            ← เปลี่ยน _invalidate_cache → selective
```

**Effort: ครึ่ง-1 วัน** — ขึ้นกับว่า Vector DB รองรับ filter delete ได้ดีแค่ไหน (Qdrant รองรับ)

---

## 5. Metrics — Persistent storage

### ปัญหา

```
ปัจจุบัน: _metrics = dict ใน memory
├── Restart → หายหมด
├── Multiple workers → แต่ละ worker นับแยกกัน
└── ดูย้อนหลังไม่ได้
```

### แนวทาง (เลือก 1)

**Option A: เก็บใน Redis (ง่ายสุด — มี Redis อยู่แล้ว)**
```python
# ใช้ Redis INCR / HSET
# ข้อดี: ไม่ต้องเพิ่ม service
# ข้อเสีย: ดู dashboard ไม่ได้
```

**Option B: Prometheus + Grafana (production-grade)**
```python
# ใช้ prometheus-fastapi-instrumentator
# ข้อดี: dashboard สวย, alert ได้, ดูย้อนหลังได้
# ข้อเสีย: เพิ่ม 2 services ใน docker-compose
```

### แนะนำ

- **ระยะสั้น:** Option A (Redis) — เปลี่ยนได้ใน 2-3 ชม.
- **ระยะยาว:** Option B (Prometheus) — ถ้าทีม oncall ต้อง monitor

---

## 6. Semantic Cache — คุ้มจริงไหม

### ปัญหา

Semantic cache (L2) ฟังดูดี แต่ในทางปฏิบัติอาจไม่คุ้ม:

```
Threshold สูง (0.92):
  "Python คืออะไร" vs "Python คืออะไรครับ" → 0.96 ✅ hit
  "Python คืออะไร" vs "อธิบาย Python ให้หน่อย" → 0.85 ❌ miss
  → hit rate ต่ำ ประหยัดไม่มาก

Threshold ต่ำ (0.80):
  "นโยบายลาป่วย" vs "นโยบายลาพักร้อน" → 0.83 ✅ hit (ผิด!)
  → ตอบผิด
```

### ควร evaluate ก่อนตัดสินใจ

```
ขั้นตอน:
1. เก็บ query log จากการใช้งานจริง 1-2 สัปดาห์
2. วิเคราะห์:
   - กี่ % เป็นคำถามซ้ำเป๊ะ (L1 จัดการได้)
   - กี่ % เป็นคำถามคล้าย (L2 ช่วยได้)
   - กี่ % เป็นคำถามใหม่ (ต้องเรียก API)
3. ถ้า L1 hit rate > 70% → อาจไม่ต้องมี L2 เลย
4. ถ้า L2 hit rate < 10% → ไม่คุ้มกับ complexity ที่เพิ่ม
```

### ทางเลือก

| สถานการณ์ | แนะนำ |
|-----------|------|
| User ถามซ้ำเป๊ะเยอะ | Redis (L1) เพียงพอ |
| User ถามคล้ายเยอะ, เป็น FAQ | Semantic cache คุ้ม |
| User ถามหลากหลาย ไม่ค่อยซ้ำ | ไม่ต้องมี cache เลย เน้น RAG |

---

## Migration Plan

ลำดับการแก้ไข เรียงตาม impact และ effort:

```
Phase 1: แก้ความเสี่ยงสูง (1-2 สัปดาห์)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
☐ 1.1 เปลี่ยน Onyx → Qdrant              (1-2 วัน)
☐ 1.2 เพิ่ม API Key authentication        (ครึ่งวัน)
☐ 1.3 ย้าย metrics ไป Redis               (ครึ่งวัน)

Phase 2: ปรับปรุงคุณภาพ (2-3 สัปดาห์)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
☐ 2.1 เปลี่ยน embedding เป็น local        (1-2 วัน + ทดสอบภาษาไทย)
☐ 2.2 Selective cache invalidation         (ครึ่ง-1 วัน)
☐ 2.3 Evaluate semantic cache ROI          (1 สัปดาห์ เก็บ data)

Phase 3: Production readiness (ต่อเนื่อง)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
☐ 3.1 Structured logging (JSON)
☐ 3.2 Error retry / circuit breaker
☐ 3.3 Prometheus + Grafana dashboard
☐ 3.4 Load testing
☐ 3.5 CI/CD pipeline
```

---

## สรุป

> POC นี้ **concept ถูกต้อง** และ **โครงสร้าง code ดี** แต่ยังไม่พร้อม production
>
> สิ่งที่ต้องทำก่อน deploy จริง: **เปลี่ยน Vector DB, ลด API dependency, เพิ่ม auth**
>
> สิ่งที่ควร evaluate: **Semantic cache คุ้มจริงไหมกับ use case ของเรา**
