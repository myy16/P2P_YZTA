import json
import os
import streamlit as st
import requests

BASE_URL = os.getenv("BASE_URL", "http://localhost:8000/api")

st.set_page_config(page_title="Doküman Sohbet", page_icon="📄", layout="wide")

# ── Session state başlangıç ───────────────────────────────────────────────────
for key, default in [
    ("messages", []),
    ("uploaded_files_info", []),
    ("username", ""),
    ("username_set", False),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Kullanıcı adı girişi ──────────────────────────────────────────────────────
def _load_user_files(username: str):
    """Backend'den kullanıcının daha önce yüklediği dosyaları çek."""
    try:
        resp = requests.get(f"{BASE_URL}/files", params={"username": username}, timeout=10)
        if resp.status_code == 200:
            return resp.json().get("files", [])
    except requests.exceptions.ConnectionError:
        pass
    return []


if not st.session_state.username_set:
    st.title("Kendi Dokümanların ile Sohbet Et 📄")
    st.markdown("### Başlamak için kullanıcı adını gir")
    col1, col2 = st.columns([3, 1])
    with col1:
        name_input = st.text_input("Kullanıcı adı", label_visibility="collapsed")
    with col2:
        if st.button("Giriş Yap", use_container_width=True) and name_input.strip():
            st.session_state.username = name_input.strip()
            st.session_state.username_set = True
            st.session_state.uploaded_files_info = _load_user_files(name_input.strip())
            st.rerun()
    st.stop()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"**Kullanıcı:** {st.session_state.username}")
    if st.button("Çıkış Yap", use_container_width=True):
        for key in ["messages", "uploaded_files_info", "username", "username_set"]:
            st.session_state[key] = [] if key in ("messages", "uploaded_files_info") else ("" if key == "username" else False)
        st.rerun()

    st.divider()
    st.header("1. Doküman Yükle")
    uploaded_files = st.file_uploader(
        "PDF, DOC, DOCX veya TXT",
        type=["pdf", "docx", "txt", "doc"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    if st.button("Yükle ve İndeksle", disabled=not uploaded_files, use_container_width=True):
        files_to_send = [("files", (f.name, f.getvalue(), f.type)) for f in uploaded_files]
        try:
            with st.spinner("İşleniyor..."):
                resp = requests.post(
                    f"{BASE_URL}/upload",
                    files=files_to_send,
                    data={"username": st.session_state.get("username", "")},
                )
            if resp.status_code == 200:
                data = resp.json()
                existing_names = {f["original_name"] for f in st.session_state.uploaded_files_info}
                added = 0
                for fi in data["uploaded_files"]:
                    if fi["original_name"] not in existing_names:
                        st.session_state.uploaded_files_info.append({
                            "file_id": fi["file_id"],
                            "original_name": fi["original_name"],
                            "chunk_count": fi["chunk_count"],
                            "size_mb": fi["size_mb"],
                        })
                        added += 1
                st.success(f"{added} dosya yüklendi!")
                st.rerun()
            else:
                st.error(f"Hata {resp.status_code}: {resp.text[:200]}")
        except requests.exceptions.ConnectionError:
            st.error("Backend'e ulaşılamadı (port 8000).")

    # Yüklü dosyalar listesi + silme
    if st.session_state.uploaded_files_info:
        st.divider()
        st.subheader("Yüklü Dosyalar")
        file_options = {"Tüm Dosyalar": None}
        for f in st.session_state.uploaded_files_info:
            file_options[f["original_name"]] = f["original_name"]

        selected_label = st.selectbox("Sorgulama kapsamı:", list(file_options.keys()))
        selected_source = file_options[selected_label]

        for f in st.session_state.uploaded_files_info:
            col_name, col_del = st.columns([5, 1])
            with col_name:
                st.markdown(f"📄 **{f['original_name']}**  \n"
                            f"<small>{f['chunk_count']} chunk · {f['size_mb']} MB</small>",
                            unsafe_allow_html=True)
            with col_del:
                if st.button("🗑", key=f"del_{f['file_id']}", help="Sil"):
                    try:
                        r = requests.delete(f"{BASE_URL}/upload", json={"file_id": f["file_id"]})
                        if r.status_code == 200:
                            st.session_state.uploaded_files_info = [
                                x for x in st.session_state.uploaded_files_info
                                if x["file_id"] != f["file_id"]
                            ]
                            st.rerun()
                        else:
                            st.error(f"Silinemedi: {r.status_code}")
                    except requests.exceptions.ConnectionError:
                        st.error("Backend'e ulaşılamadı.")

        st.divider()
        if st.button("Seçili Dosyayı Özetle", use_container_width=True):
            payload = {"max_chunks": 8, "username": st.session_state.get("username") or None}
            if selected_source:
                payload["source_file"] = selected_source
            try:
                with st.spinner("Özetleniyor..."):
                    r = requests.post(f"{BASE_URL}/summarize", json=payload)
                if r.status_code == 200:
                    summary_text = r.json().get("summary", "Özet alınamadı.")
                    label = selected_label if selected_label != "Tüm Dosyalar" else "Tüm Dosyalar"
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"**Özet — {label}:**\n\n{summary_text}",
                        "sources": [],
                    })
                    st.rerun()
                else:
                    st.error(f"Özetleme hatası: {r.status_code}")
            except requests.exceptions.ConnectionError:
                st.error("Backend'e ulaşılamadı.")
    else:
        selected_source = None

    # Mimari akış bilgisi
    st.divider()
    with st.expander("RAG Mimarisi"):
        st.markdown("""
1. Doküman yükleme
2. Metne çevirme
3. Chunking
4. Embedding oluşturma
5. Chroma vektör DB kaydı
6. Kullanıcı sorusu
7. İlgili chunk'ları retrieve et
8. Groq LLM ile cevap üret
        """)

# ── Ana alan ──────────────────────────────────────────────────────────────────
header_col, clear_col = st.columns([6, 1])
with header_col:
    st.title(f"Merhaba, {st.session_state.username} 👋")
with clear_col:
    if st.button("Sohbeti Temizle", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Geçmiş mesajlar
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            source_names = list({s.get("source_file", "") for s in msg["sources"] if s.get("source_file")})
            st.caption("Kaynak: " + " · ".join(f"📄 {n}" for n in source_names))

# Yeni soru
if prompt := st.chat_input("Dokümanlarla ilgili ne öğrenmek istersin?"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt, "sources": []})

    if not st.session_state.uploaded_files_info:
        reply = "Henüz doküman yüklemedin. Sol panelden dosya yükleyerek başlayabilirsin."
        with st.chat_message("assistant"):
            st.markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply, "sources": []})
    else:
        payload = {"question": prompt, "top_k": 5, "username": st.session_state.get("username") or None}
        if selected_source:
            payload["source_file"] = selected_source

        try:
            with st.chat_message("assistant"):
                placeholder = st.empty()
                full_response = ""
                sources = []

                with requests.post(
                    f"{BASE_URL}/chat/stream",
                    json=payload,
                    stream=True,
                    timeout=60,
                ) as r:
                    for line in r.iter_lines():
                        if not line:
                            continue
                        line = line.decode("utf-8")
                        if not line.startswith("data:"):
                            continue
                        raw = line[len("data:"):].strip()
                        try:
                            chunk = json.loads(raw)
                        except Exception:
                            continue
                        if chunk.get("type") == "token":
                            full_response += chunk.get("content", "")
                            placeholder.markdown(full_response + "▌")
                        elif chunk.get("type") == "sources":
                            sources = chunk.get("content", [])
                        elif chunk.get("type") == "error":
                            full_response = f"Hata: {chunk.get('detail', 'Bilinmeyen hata')}"

                placeholder.markdown(full_response)

                if sources:
                    source_names = list({s.get("source_file", "") for s in sources if s.get("source_file")})
                    st.caption("Kaynak: " + " · ".join(f"📄 {n}" for n in source_names))

            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "sources": sources,
            })

        except requests.exceptions.ConnectionError:
            err = "Backend'e ulaşılamadı (port 8000)."
            with st.chat_message("assistant"):
                st.markdown(err)
            st.session_state.messages.append({"role": "assistant", "content": err, "sources": []})
