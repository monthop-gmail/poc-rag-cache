# 🧪 Testing Guide: Gemini Semantic Cache

เอกสารนี้รวบรวมวิธีการทดสอบระบบ Semantic Cache เพื่อตรวจสอบความถูกต้องของ L1 (Redis), L2 (Onyx) และการเรียก Gemini API

---

## 1. การเตรียมสภาพแวดล้อม (Prerequisites)

มั่นใจว่ารันระบบผ่าน Docker เรียบร้อยแล้ว:
```bash
docker-compose up -d
```
ตรวจสอบสถานะ API:
```bash
curl http://localhost:8000/health
```

---

## 2. การทดสอบแบบ Manual (Manual Testing via cURL)

เราจะทำการทดสอบ 3 Step เพื่อดูพฤติกรรมของ Cache

### Step 1: Cache Miss (ครั้งแรกที่ถาม)
ระบบจะไปเรียก Gemini API และบันทึกลง Onyx/Redis
```bash
curl -X POST "http://localhost:8000/ask" \
     -H "Content-Type: application/json" \
     -d '{"query": "แนะนำวิธีประหยัดไฟในบ้านหน่อย", "threshold": 0.9}'
```
**ผลลัพธ์ที่คาดหวัง:** `cache_level: "MISS"`, `saved_api_call: false`

### Step 2: L1 Cache Hit (ถามคำเดิมเป๊ะๆ)
ระบบจะดึงจาก Redis ทันที (Latency ควรจะต่ำมาก < 10ms)
```bash
curl -X POST "http://localhost:8000/ask" \
     -H "Content-Type: application/json" \
     -d '{"query": "แนะนำวิธีประหยัดไฟในบ้านหน่อย"}'
```
**ผลลัพธ์ที่คาดหวัง:** `cache_level: "L1 (Redis)"`, `saved_api_call: true`

### Step 3: L2 Cache Hit (ถามด้วยความหมายที่ใกล้เคียง)
ลองเปลี่ยนคำถามเล็กน้อยแต่ความหมายเดิม ระบบจะคำนวณ Vector และหาเจอใน Onyx
```bash
curl -X POST "http://localhost:8000/ask" \
     -H "Content-Type: application/json" \
     -d '{"query": "ขอเทคนิคการลดค่าไฟในบ้าน"}'
```
**ผลลัพธ์ที่คาดหวัง:** `cache_level: "L2 (Onyx)"`, `saved_api_call: true`

---

## 3. การทดสอบแบบ Automated (Unit Tests)

เราใช้ `unittest` ร่วมกับ `unittest.mock` เพื่อทดสอบ Logic โดยไม่ต้องต่อ API จริง

### วิธีการรัน Test (ภายใน Docker):
```bash
docker-compose exec app python test_cache.py
```

### สิ่งที่ Test ตรวจสอบ:
- **Onyx Logic**: ตรวจสอบว่าถ้า Similarity Score ต่ำกว่า Threshold ระบบจะถือว่าเป็น Cache Miss
- **Flow Logic**: ตรวจสอบว่าถ้าเจอใน L1 จะต้องไม่เรียก Gemini API
- **Mocking**: จำลองการทำงานของ `onyx-database` และ `google-generativeai` เพื่อความรวดเร็วในการทดสอบ

---

## 4. การปรับจูน Threshold (Fine-tuning)

ในไฟล์ `main.py` หรือตอนส่ง Request คุณสามารถปรับค่า `threshold` ได้:
- **สูง (0.95+)**: เน้นคำตอบที่ตรงประเด็นมากที่สุด (มีโอกาส Miss สูงขึ้น)
- **ต่ำ (0.85 - 0.90)**: เน้นการประหยัด API Quota (อาจได้คำตอบที่ใกล้เคียงแต่ไม่เป๊ะ 100%)

---

## 5. การตรวจสอบ Logs

ดูการทำงานเบื้องหลังและการ Hit/Miss ของ Cache ได้ผ่าน:
```bash
docker-compose logs -f app
```
