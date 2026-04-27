import re

with open('tests/test_rag_service.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Fix line 128 (index 127) - change 8 spaces to 4
lines[127] = '    monkeypatch.setattr(service, "_call_groq", lambda prompt, system_prompt: "Yanıt üretildi")\n'

with open('tests/test_rag_service.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print('Fixed line 128')
