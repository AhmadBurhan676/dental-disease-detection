import urllib.request, urllib.error, json
req = urllib.request.Request(
    'http://127.0.0.1:8000/api/retrain',
    data=b'{"epochs":1}',
    headers={'Content-Type': 'application/json'},
    method='POST'
)
try:
    urllib.request.urlopen(req)
    print("FAIL: Expected 403 but got 200")
except urllib.error.HTTPError as e:
    body = json.loads(e.read())
    print(f"PASS: Status {e.code} - {e.reason}")
    print(f"Detail: {body.get('detail', '')}")
