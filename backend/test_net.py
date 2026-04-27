import socket
import ssl
try:
    ip = socket.gethostbyname("api.groq.com")
    print(f"DNS OK: api.groq.com -> {ip}")
    # Try TCP connect
    s = socket.create_connection(("api.groq.com", 443), timeout=10)
    s.close()
    print("TCP OK: port 443 reachable")
except Exception as e:
    print(f"FAILED: {e}")
