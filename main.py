from flask import Flask, render_template, send_from_directory
from flask_socketio import SocketIO, emit
from flask_cors import CORS
from matching import *
from time import time
import numpy as np
import base64
import cv2
import os


# https://colab.research.google.com/drive/1RoKdGm3sNbQMlvocY_zxFa7wfl4l-NjC#scrollTo=iMOz0e9Tld6-&uniqifier=1
# book1 - https://universe.roboflow.com/eltanma/bookdetection-yoeop/dataset/2#
# book2 - https://universe.roboflow.com/bookdetection-lgtpa/book-spine-detector <- 제일 잘 됨!
currDir = os.path.dirname(os.path.realpath(__file__))
app = Flask(__name__, static_folder="./templates/static")
app.config["SECRET_KEY"] = "secret!"
socketio = SocketIO(app)
CORS(app)

model = YOLO('model/yolov8n_book2_epoch400.pt')
model.fuse()

@app.route("/favicon.ico")
def favicon():
    return send_from_directory(
        os.path.join(app.root_path, "static"),
        "favicon.ico",
        mimetype="image/vnd.microsoft.icon",
    )



@socketio.on("connect")
def test_connect():
    print("Connected")
    emit("my response", {"data": "Connected"})
    

def base64_to_image(base64_string):
    base64_data = base64_string.split(",")[1]
    image_bytes = base64.b64decode(base64_data)
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    return image


@socketio.on("image")
def receive_image(image):
    sTime = time()
    image = base64_to_image(image)
    # image = cv2.flip(image, 1)
    # res = match(image, template, cv2.TM_CCOEFF_NORMED, [0.6, 1])
    frame = detectYOLO(model, image, template, confThreshold=0.55, HSVThreshold=0.5)
    
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    result, frame_encoded = cv2.imencode(".jpg", frame, encode_param)
    processed_img_data = base64.b64encode(frame_encoded).decode()
    b64_src = "data:image/jpg;base64,"
    processed_img_data = b64_src + processed_img_data
    print(1/(time()-sTime))
    emit("processed_image", processed_img_data)


@app.route("/")
def index():
    return render_template("index.html")


if __name__ == "__main__":
    isbn = "9788901260716"
    template = cv2.imread(f'{currDir}/static/books/images/{isbn}/spine.jpg', cv2.IMREAD_COLOR)
    socketio.run(app, debug=True, port=5000, host="0.0.0.0", ssl_context=(currDir+'/SSL/cert.pem', currDir+'/SSL/key.pem'))