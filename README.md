# P2P_YZTA — Kendi Dokümanlarınla Sohbet Et

Bu uygulama, kullanıcıların sisteme yüklediği PDF, DOCX ve TXT dosyalarını analiz eden ve bu dokümanlar üzerinden yapay zeka ile sohbet edilmesini sağlayan bir RAG (Retrieval-Augmented Generation) projesidir.

## Temel Özellikler

* **Hızlı Dosya Yükleme:** PDF, DOCX, DOC ve TXT dosyalarını sürükle-bırak ile yükleme ve anında işleme.
* **Kullanıcı Bazlı Doküman Yönetimi:** Her kullanıcı yalnızca kendi yüklediği dosyaları görür ve sorgular. Çıkış yapıp tekrar giriş yapıldığında dosyalar kaybolmaz.
* **Modern Chat Arayüzü:** Streamlit ile geliştirilmiş, kullanıcı dostu ve akıcı sohbet ekranı.
* **Cevap Akışı (Streaming):** Yapay zeka yanıtlarının ChatGPT'deki gibi kelime kelime akarak ekrana gelmesi.
* **Kaynak Gösterme:** Her cevabın altında hangi dosyadan üretildiği gösterilir.
* **Doküman Özetleme:** Seçili dosya veya tüm dosyalar için tek tıkla özet üretimi.
* **FastAPI Backend:** Yüksek performanslı ve ölçeklenebilir arka plan motoru.
* **Docker Desteği:** Tek komutla tüm stack'i ayağa kaldırma.

## Kullanılan Teknolojiler

* **Arayüz (Frontend):** Streamlit
* **Sunucu (Backend):** FastAPI / Uvicorn
* **Vektör Veritabanı:** ChromaDB (kalıcı, disk tabanlı)
* **Embedding Modeli:** sentence-transformers / all-MiniLM-L6-v2
* **LLM:** Groq API (llama-3.1-8b-instant)
* **Dil:** Python 3.9+
* **Konteyner:** Docker / Docker Compose

---

## Kurulum ve Çalıştırma

### 1. Yerel Kurulum (venv)

**Gereksinimler:** Python 3.9+, pip, antiword (DOC dosyaları için)

```bash
# antiword kur (macOS)
brew install antiword

# Projeyi klonla
git clone https://github.com/myy16/P2P_YZTA.git
cd P2P_YZTA

# ── Backend ──
cd backend
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Sunucuyu başlat
uvicorn main:app --reload
```

> **Not:** `.env` dosyasını proje **kök dizininde** (`P2P_YZTA/.env`) oluştur:
> ```
> GROQ_API_KEY=your_key_here
> ```

Swagger UI: http://127.0.0.1:8000/docs

```bash
# ── Frontend (ayrı terminal) ──
cd P2P_YZTA
pip install -r requirements.txt
streamlit run app.py
```

Streamlit UI: http://localhost:8501

---

### 2. Docker ile Çalıştırma

**Gereksinim:** Docker Desktop

```bash
# Proje kök dizininden
cd P2P_YZTA
docker compose up --build
```

Swagger UI: http://localhost:8000/docs  
Streamlit UI: http://localhost:8501

Durdurmak için:
```bash
docker compose down
```

Veri (Chroma + uploads) Docker volume'larında kalıcı olarak saklanır. Sıfırlamak için:
```bash
docker compose down -v
```

Chroma vektör verisi Docker volume içinde tutulur ve `backend/data/chroma` altında kalıcıdır.

---

## API Endpoints

### POST /api/upload

Bir veya birden fazla doküman yükler. Sonuçları tümünü birden döner.

**Desteklenen formatlar:** PDF, DOCX, DOC, TXT
**Maksimum dosya boyutu:** 20 MB

**Örnek response:**
```json
{
  "uploaded_files": [
    {
      "file_id": "uuid",
      "original_name": "rapor.pdf",
      "file_type": "pdf",
      "size_mb": 0.057,
      "extracted_text": "...",
      "chunks": [
        {
          "chunk_id": "uuid",
          "file_id": "uuid",
          "source_file": "rapor.pdf",
          "file_type": "pdf",
          "chunk_index": 0,
          "total_chunks": 7,
          "text": "...",
          "char_count": 491
        }
      ],
      "chunk_count": 7
    }
  ],
  "count": 1
}
```

---

### POST /api/upload/stream

Aynı işlemi Server-Sent Events (SSE) ile yapar. Her dosya işlenince anlık event gönderir.

**Event formatı:**
```
data: {"event": "file_done", "file": {...}}
data: {"event": "error", "filename": "...", "detail": "..."}
data: {"event": "done", "count": 1}
```

---

### GET /api/files

Kullanıcının daha önce yüklediği dosyaların listesini döner.

**Query parametresi:** `username`

**Örnek response:**
```json
{
  "files": [
    {"file_id": "uuid", "original_name": "rapor.pdf", "chunk_count": 7, "size_mb": 0.0}
  ]
}
```

---

### POST /api/chat

Yüklenen dokümanlara göre soru-cevap üretir. Kaynakları da response içinde döner.

**Örnek request:**
```json
{
  "question": "Bu dokümanda ana konu nedir?",
  "top_k": 5,
  "username": "kullanici_adi"
}
```

### POST /api/chat/stream

Aynı soru-cevap işlemini Server-Sent Events ile akışlı yapar.

**Event formatı:**
```
data: {"type": "token", "content": "kelime"}
data: {"type": "sources", "content": [...]}
data: {"type": "error", "detail": "..."}
```

---

### POST /api/summarize

Seçili doküman veya tüm indeks üzerinden özet üretir.

**Örnek request:**
```json
{
  "source_file": "rapor.pdf",
  "max_chunks": 8
}
```

---

### DELETE /api/upload

Yüklenen bir dokümanı vektör veritabanından ve diskten siler.

**Örnek request:**
```json
{
  "file_id": "uuid"
}
```

---

## Doküman İşleme Pipeline

Yüklenen her dosya şu adımlardan geçer:

```
Yükleme → Parse → Temizleme → Chunking → Response
```

| Adım | Modül | Açıklama |
|------|-------|----------|
| Parse | `app/core/parser.py` | PDF (pdfplumber), DOCX/DOC (python-docx, antiword), TXT (encoding detection) |
| Temizleme | `app/core/cleaner.py` | Kontrol karakterleri, Unicode NFC, header/footer pattern'ları, fazla boşluk |
| Chunking | `app/core/chunker.py` | Recursive character splitting, chunk_size=500, overlap=50 |

---

## Testleri Çalıştırma

```bash
cd backend
source venv/bin/activate
pytest tests/test_chunker.py -v
```

---

## Proje Yapısı

```
P2P_YZTA/
├── app.py                       # Streamlit frontend arayüzü
├── requirements.txt             # Frontend bağımlılıkları (streamlit, requests)
├── Dockerfile                   # Frontend Docker image
├── docker-compose.yml           # Backend + Frontend stack
├── .env                         # GROQ_API_KEY (git'e ekleme!)
└── backend/
    ├── main.py                  # FastAPI uygulama giriş noktası
    ├── requirements.txt         # Backend bağımlılıkları
    ├── Dockerfile               # Backend Docker image
    ├── app/
    │   ├── api/
    │   │   ├── upload.py        # POST/DELETE /api/upload
    │   │   ├── chat.py          # POST /api/chat, /api/chat/stream
    │   │   ├── summarize.py     # POST /api/summarize
    │   │   └── files.py         # GET /api/files
    │   └── core/
    │       ├── config.py        # Uygulama ayarları
    │       ├── parser.py        # Doküman parser'ları (PDF, DOCX, TXT)
    │       ├── cleaner.py       # Metin temizleme
    │       ├── chunker.py       # Recursive character splitting
    │       ├── embeddings.py    # sentence-transformers embedding servisi
    │       ├── vector_store.py  # ChromaDB wrapper
    │       ├── retriever.py     # Semantic search + metadata filtreleme
    │       └── rag_service.py   # RAG orchestration (index, retrieve, generate)
    └── tests/
        ├── test_chunker.py
        ├── test_rag_service.py
        └── test_api_integration.py
```

---

## GitHub Commit Formatı

```
(type) scope : description
```

Bu format, dokümanların nasıl işlendiğini, vektörleştiğini ve yapay zeka tarafından nasıl anlamlandırıldığını adım adım izlememizi sağlar.

### Types

| Type | Kullanım |
|------|----------|
| `(feat)` | Yeni özellikler veya arayüz bileşenleri |
| `(fix)` | Hata düzeltmeleri |
| `(style)` | Sadece görsel değişiklikler |
| `(refactor)` | RAG mantığını veya kod yapısını iyileştirme |
| `(chore)` | API anahtarları, kütüphane kurulumları veya ayar dosyaları |
| `(rag)` | Embedding, vektör veritabanı veya doküman işleme süreçleri |
| `(docs)` | README, kurulum kılavuzu veya yorum satırı eklemeleri |
| `(data)` | Veri temizleme ve ön işleme |
| `(test)` | Test ekleme veya güncelleme |
| `(config)` | Yapılandırma dosyaları |

### Örnekler

```
(feat) upload : add multi-format file support for PDF and DOCX uploads
(rag) chunking : implement recursive character splitting for better context
(rag) vector-db : integrate FAISS for efficient similarity search
(fix) parser : resolve encoding issues while reading Turkish characters in TXT files
(refactor) prompt : optimize system prompt to include source citations in responses
(data) cleaning : remove redundant white spaces and headers from parsed text
(chore) deps : add langchain and groq-sdk to requirements.txt
(style) chat : apply scrolling effect to chat window for long conversations
(docs) readme : add architecture diagram and local setup instructions
```
