import re

with open('tests/test_rag_service.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix test_answer_question_builds_context_and_sources
old = '''def test_answer_question_builds_context_and_sources(monkeypatch):
    monkeypatch.setattr(service, "_call_groq", lambda prompt, system_prompt: "Yanıt üretildi")
    monkeypatch.setattr(service, "_call_groq", lambda prompt: "Yanıt üretildi")

    result = service.answer_question("Belge ne anlatıyor?")

    assert result["answer"] == "Yanıt üretildi"
    assert result["sources"][0]["source_file"] == "dosya.pdf"
    assert result["context"][0]["text"] == "Chunk 1 içerik"
    assert "evaluation" in result'''

new = '''def test_answer_question_builds_context_and_sources(monkeypatch):
    service, _, _, _ = build_service(monkeypatch)
    monkeypatch.setattr(service, "_call_groq", lambda prompt, system_prompt: "Yanıt üretildi")

    result = service.answer_question("Belge ne anlatıyor?")

    assert result["answer"] == "Yanıt üretildi"
    assert result["sources"][0]["source_file"] == "dosya.pdf"
    assert result["context"][0]["text"] == "Chunk 1 içerik"
    assert "evaluation" in result'''

content = content.replace(old, new)

with open('tests/test_rag_service.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('Fixed test_answer_question_builds_context_and_sources')
