import re

with open('tests/test_rag_service.py', 'r') as f:
    lines = f.readlines()

# Fix line 128 (index 127) - it has 8 spaces, change to 4
lines[127] = '    monkeypatch.setattr(service, "_call_groq", lambda prompt, system_prompt: "Yanıt üretildi")\n'

with open('tests/test_rag_service.py', 'w') as f:
    f.writelines(lines)

print('Fixed line 128')
