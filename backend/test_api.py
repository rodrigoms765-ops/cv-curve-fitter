import urllib.request
import urllib.error
from urllib.parse import urlencode
import json
import uuid

# Very basic multipart/form-data encoding
boundary = uuid.uuid4().hex

filepath = r"C:\Users\rodri\OneDrive\Documents\Chem Project\Rodrigo CV\2000rpm_1p0gL_3cm2\2p0k 30s 1p0g 3cm2_-1V1V_10mVs.csv"
try:
    with open(filepath, 'rb') as f:
        file_content = f.read()
except FileNotFoundError:
    print("File not found, skipping API test")
    exit(0)

body = bytearray()
def add_field(name, value):
    body.extend(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode('utf-8'))

def add_file(name, filename, content):
    body.extend(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"; filename="{filename}"\r\nContent-Type: text/csv\r\n\r\n'.encode('utf-8'))
    body.extend(content)
    body.extend(b'\r\n')

add_field('scan_rate', '0.010')
add_field('film_thickness', '0.0001')
add_field('v_min', '-1.0')
add_field('v_max', '1.0')
add_field('skip_factor', '10')
add_field('num_peaks', '30')
add_field('pot_col', '8')
add_field('cur_col', '9')
add_file('file', 'test.csv', file_content)

body.extend(f'--{boundary}--\r\n'.encode('utf-8'))

req = urllib.request.Request('http://127.0.0.1:8000/solve', data=body)
req.add_header('Content-Type', f'multipart/form-data; boundary={boundary}')

try:
    with urllib.request.urlopen(req) as response:
        res = json.loads(response.read().decode('utf-8'))
        print("Status Code:", response.status)
        print("Response Status:", res.get('status'))
        if res.get('status') == 'success':
            print("Parameters:", res.get('data', {}).get('parameters'))
        else:
            print("Error:", res.get('message'))
except urllib.error.URLError as e:
    print("URLError:", e)
