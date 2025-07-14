from flask import Flask, request, send_file, render_template
from flask_cors import CORS
import cv2
import numpy as np
import io

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/sketch', methods=['POST'])
def sketch():
    file = request.files['image']
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    inv = 255 - gray
    blur = cv2.GaussianBlur(inv, (21, 21), 0)
    sketch = cv2.divide(gray, 255 - blur, scale=256.0)
    _, buffer = cv2.imencode('.png', sketch)
    return send_file(io.BytesIO(buffer), mimetype='image/png')

if __name__ == "__main__":
    app.run(debug=True)