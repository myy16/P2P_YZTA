# P2P_YZTA
🛠️ GitHub Commit Format: (type) scope : description
Bu format, dokümanların nasıl işlendiğini, vektörleştiğini ve yapay zeka tarafından nasıl anlamlandırıldığını adım adım izlememizi sağlar.

Types
(feat): Yeni özellikler veya arayüz bileşenleri (PDF yükleme, chat ekranı vb.).

(fix): Hata düzeltmeleri (dosya okuma hatası, LLM bağlantı kopması vb.).

(style): Sadece görsel değişiklikler (CSS, buton renkleri, chat balonları).

(refactor): RAG mantığını veya kod yapısını iyileştirme (chunking stratejisi değişimi).

(chore): API anahtarları, kütüphane kurulumları veya ayar dosyaları.

(rag): Embedding, vektör veritabanı veya doküman işleme süreçlerine özel güncellemeler.

(docs): README, kurulum kılavuzu veya yorum satırı eklemeleri.

RAG Projesi İçin Örnekler
(feat) upload : add multi-format file support for PDF and DOCX uploads

(rag) chunking : implement recursive character splitting for better context

(rag) vector-db : integrate FAISS for efficient similarity search

(fix) parser : resolve encoding issues while reading Turkish characters in TXT files

(refactor) prompt : optimize system prompt to include source citations in responses

(data) cleaning : remove redundant white spaces and headers from parsed text

(chore) deps : add langchain and groq-sdk to requirements.txt

(style) chat : apply scrolling effect to chat window for long conversations

(docs) readme : add architecture diagram and local setup instructions
