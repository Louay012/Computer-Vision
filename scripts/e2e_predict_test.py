import io
import os
import sys
from pathlib import Path

try:
    from PIL import Image
except Exception:
    print('Pillow is required for this test. Please install pillow in the environment.')
    sys.exit(2)

URL = os.environ.get('PREDICT_URL', 'http://127.0.0.1:8000/api/predict')
MODEL_TYPE = os.environ.get('PREDICT_MODEL_TYPE', 'dl')
MODEL_NAME = os.environ.get('PREDICT_MODEL_NAME')
TMP = Path(__file__).parent / 'tmp_test_img.jpg'

def make_image(path: Path):
    img = Image.new('RGB', (224, 224), color=(200, 40, 40))
    img.save(path)

def run_with_requests(path: Path):
    try:
        import requests
    except Exception:
        return False
    with open(path, 'rb') as f:
        files = {'file': ('test.jpg', f, 'image/jpeg')}
        data = {'model_type': MODEL_TYPE}
        if MODEL_NAME:
            data['model_name'] = MODEL_NAME
        try:
            r = requests.post(URL, files=files, data=data, timeout=60)
            print('Status:', r.status_code)
            print('Response body:')
            print(r.text)
            return True
        except Exception as e:
            print('Request failed:', e)
            return True

def run_with_urllib(path: Path):
    import mimetypes
    from urllib import request

    boundary = '----WebKitFormBoundary7MA4YWxkTrZu0gW'
    data = []
    data.append(f'--{boundary}')
    data.append('Content-Disposition: form-data; name="model_type"')
    data.append('')
    data.append(MODEL_TYPE)
    data.append(f'--{boundary}')
    filename = path.name
    content_type = mimetypes.guess_type(str(path))[0] or 'application/octet-stream'
    data.append(f'Content-Disposition: form-data; name="file"; filename="{filename}"')
    data.append(f'Content-Type: {content_type}')
    data.append('')
    body_pre = '\r\n'.join(data).encode('utf-8')
    with open(path, 'rb') as f:
        body = body_pre + f.read() + (f'\r\n--{boundary}--\r\n').encode('utf-8')
    req = request.Request(URL, data=body)
    req.add_header('Content-Type', f'multipart/form-data; boundary={boundary}')
    try:
        with request.urlopen(req, timeout=60) as resp:
            print('Status:', resp.status)
            print('Response body:')
            print(resp.read().decode('utf-8'))
    except Exception as e:
        print('urllib request failed:', e)

def main():
    make_image(TMP)
    print('Created test image at', TMP)
    # CLI args: [model_type] [model_name]
    if len(sys.argv) > 1:
        global MODEL_TYPE, MODEL_NAME
        MODEL_TYPE = sys.argv[1]
        if len(sys.argv) > 2:
            MODEL_NAME = sys.argv[2]
    if run_with_requests(TMP):
        return
    print('requests not available or failed; trying urllib')
    run_with_urllib(TMP)

if __name__ == '__main__':
    main()
