# 📄 Dokümanlarla Akıllı Sohbet (P2P_YZTA)

Bu uygulama, kullanıcıların sisteme yüklediği PDF, DOCX ve TXT dosyalarını analiz eden ve bu dokümanlar üzerinden yapay zeka ile sohbet edilmesini sağlayan bir RAG (Retrieval-Augmented Generation) projesidir.

## ✨ Temel Özellikler
* **Hızlı Dosya Yükleme:** Dosyaları sürükle-bırak yöntemiyle sisteme aktarma ve anında işleme.
* **Modern Chat Arayüzü:** Streamlit ile geliştirilmiş, kullanıcı dostu ve akıcı sohbet ekranı.
* **Cevap Akışı (Streaming):** Yapay zeka yanıtlarının ChatGPT'deki gibi kelime kelime akarak ekrana gelmesi.
* **FastAPI Backend:** Yüksek performanslı ve ölçeklenebilir arka plan motoru.

## 🛠️ Kullanılan Teknolojiler
* **Arayüz (Frontend):** Streamlit
* **Sunucu (Backend):** FastAPI / Uvicorn
* **Dil:** Python 3.9+

## 🚀 Kurulum ve Çalıştırma

### 1. Kütüphaneleri Kurma
```bash
pip install streamlit requests