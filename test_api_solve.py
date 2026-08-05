import urllib.request, json

boundary = '----WebKitFormBoundary7MA4YWxkTrZu0gW'
file_path = r'C:\Users\rodri\OneDrive\Documents\Chem Project\Rodrigo CV\1000rpm_0p1gL_2cm2\1p0k 30s 0p1g 2cm2_-1V1V_10mVs.csv'

with open(file_path, 'rb') as f:
    file_bytes = f.read()

body = bytearray()
def add_field(name, val):
    global body
    body.extend(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{val}\r\n'.encode('utf-8'))

add_field('film_thickness', '1e-4')
add_field('scan_rate_v_s', '0.010')
add_field('v_min', '-1.0')
add_field('v_max', '1.0')
add_field('num_peaks', '4')
add_field('num_terms', '25')
add_field('skip_factor', '10')
add_field('loss_weight_const', '1.0')
add_field('max_iter', '50')
add_field('tol_ftol', '1e-8')
add_field('tol_gtol', '1e-7')
add_field('pot_col', '8')
add_field('cur_col', '9')

body.extend(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="data.csv"\r\nContent-Type: text/csv\r\n\r\n'.encode('utf-8'))
body.extend(file_bytes)
body.extend(f'\r\n--{boundary}--\r\n'.encode('utf-8'))

req = urllib.request.Request('http://127.0.0.1:8000/api/solve', data=bytes(body))
req.add_header('Content-Type', f'multipart/form-data; boundary={boundary}')

with urllib.request.urlopen(req) as resp:
    for line in resp:
        line_str = line.decode('utf-8').strip()
        if line_str.startswith('data:'):
            payload = json.loads(line_str[5:].strip())
            if payload.get('type') == 'init':
                print('[SSE] Data loaded: points =', len(payload['exp_potential']))
            elif payload.get('type') == 'iter':
                if payload['iter'] % 10 == 0 or payload['iter'] == 1:
                    print(f'[SSE] Iter {payload["iter"]}: loss={payload["loss"]:.5f}')
            elif payload.get('type') == 'done':
                print('[SSE] COMPLETED SUCCESSFULLY!')
                print('Params:', payload['data']['parameters'])
