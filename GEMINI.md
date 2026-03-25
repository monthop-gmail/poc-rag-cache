# 🛠️ Setup Semantic Cache with Onyx & Gemini

ช่วยสร้างระบบ Caching เพื่อประหยัด API Quota โดยมีรายละเอียดดังนี้:

## 1. Requirements
สร้างไฟล์ `requirements.txt` ที่รวม:
- google-generativeai
- onyx-client (หรือตัวเชื่อมต่อ Onyx ที่เสถียรที่สุดในปี 2026)
- numpy (สำหรับคำนวณ similarity เบื้องต้น)

## 2. Onyx Client Setup
สร้างไฟล์ `onyx_provider.py`:
- สร้าง Class `OnyxCache` สำหรับจัดการ Connection
- มีฟังก์ชัน `upsert_cache(vector, query, response)`
- มีฟังก์ชัน `query_cache(vector, threshold=0.9)`

## 3. Gemini Integration with Cache
สร้างไฟล์ `main.py`:
- ใช้ `models/text-embedding-004` สำหรับแปลงคำถามเป็น Vector
- เช็ค Cache ใน Onyx ก่อนทุกครั้ง
- ถ้าเจอ (Cache Hit) ให้ดึงคำตอบมาแสดง
- ถ้าไม่เจอ (Cache Miss) ให้เรียก Gemini Pro/Flash แล้วบันทึกลง Onyx ทันที

## 4. Environment Config
สร้างไฟล์ `.env.example` สำหรับใส่:
- GEMINI_API_KEY
- ONYX_HOST
- ONYX_PORT
