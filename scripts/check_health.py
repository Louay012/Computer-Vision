import urllib.request

URL = 'http://127.0.0.1:8000/api/health'
print('Checking', URL)
try:
    with urllib.request.urlopen(URL, timeout=10) as r:
        print('Status:', r.status)
        print('Body:', r.read().decode())
except Exception as e:
    print('Health check failed:', e)
