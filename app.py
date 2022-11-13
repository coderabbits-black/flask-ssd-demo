from flask import Flask, flash, request, jsonify
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import tensorflow as tf
import numpy as np


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)

detector = hub.load(
    "https://hub.tensorflow.google.cn/tensorflow/ssd_mobilenet_v2/fpnlite_320x320/1")


def image_to_tensor(image_data):
    pil_image = Image.open(image_data)
    pil_image_rgb = pil_image.convert("RGB")
    row = np.array(pil_image_rgb)
    return tf.expand_dims(tf.image.convert_image_dtype(row, tf.uint8), axis=0)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return 'No file part', 400
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return 'No selected file', 400 
        if file and allowed_file(file.filename):
            tensor = image_to_tensor(file.stream)
            detector_output = detector(tensor)
            item_count = int(detector_output['num_detections'][0].numpy())
            items = list(map(lambda index: ({
                "class": detector_output["detection_classes"][0][index].numpy().astype(int).tolist(),
                "score": detector_output["detection_scores"][0][index].numpy().tolist(),
                "box": detector_output["detection_boxes"][0][index].numpy().tolist(),
            }), range(item_count)))

            return jsonify({
                "count": item_count,
                "items": items                
            })

    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''


if __name__ == "__main__":
    app.run()
