lines = open('tests/test_rag_service.py').readlines()
nl = lines[:126]
nl.append('def test_answer_question_builds_context_and_sources(monkeypatch):\n')
nl.append('    service, _, _, _ = build_service(monkeypatch)\n')
nl.append('    monkeypatch.setattr(service, \
_call_groq\, lambda prompt, system_prompt: \Yanıt
üretildi\)\n')
nl.append('\n')
nl.append('    result = service.answer_question(\Belge
ne
anlatıyor?\)\n')
nl.append('\n')
nl.append('    assert result[\answer\] == \Yanıt
üretildi\\n')
nl.append('    assert result[\sources\][0][\source_file\] == \dosya.pdf\\n')
nl.append('    assert result[\context\][0][\text\] == \Chunk
1
i�erik\\n')
nl.append('    assert \evaluation\ in result\n')
nl.extend(lines[134:])
with open('tests/test_rag_service.py','w') as f:
    f.writelines(nl)
print('Fixed')
