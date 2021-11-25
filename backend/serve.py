from flask import Flask, request, jsonify
from flask_cors import CORS
import json

import base64
import matplotlib.pyplot as plt
import skimage.io
from skimage.transform import resize
import cv2

import torch
import torch.nn.functional as F

import numpy as np
from triton import triton_infer
from detection import Detect, prep_display

from mysql.connector import connect, Error
from getpass import getpass

img_size = (768, 768)
means = (123.68, 116.78, 103.94)
std = (58.40, 57.12, 57.38)
class_names = ["cp", "ALB", 'BET', 'BUM', 'DOL', 'LEC', 'MLS', 'OIL', 'SBT', 'SFA', 'SKJ', 'SSP', 'SWO', 'Shark', 'WAH', 'YFT']

def base64_to_img(base64_str):
    if isinstance(base64_str, bytes):
        base64_str = base64_str.decode("utf-8")

    imgdata = base64.b64decode(base64_str)
    img = skimage.io.imread(imgdata, plugin='imageio')

    return img

def img_to_base64(img):
    if isinstance(img, np.ndarray):
        retval, buffer = cv2.imencode('.jpg', img)
        jpg_as_text = base64.b64encode(buffer).decode()
        #print(jpg_as_text)
        return jpg_as_text
    else:
        return None

def preprocessing(img):
    # Resize image and convert to numpy array
    img = (resize(img, img_size)*255).astype(np.int)
    # Normalize
    img = (img - np.asarray(means)) / np.asarray(std)
    # Swap to CHW
    img = np.transpose(img, (2, 0, 1)).astype(np.float32)

    return img

app = Flask(__name__)
app.config["DEBUG"] = True
CORS(app)

@app.route("/predict", methods=['POST'])
def post_classify():
    print(request)
    result = request.get_json(silent=True)

    img = base64_to_img(result)
    h, w, _ = img.shape
    processed_img = preprocessing(img)

    # NCHW
    batch_img = processed_img[np.newaxis, :, :, :]

    prediction = single_predict(batch_img, h, w)
    #print(prediction["class"])

    return jsonify(prediction)


def single_predict(batch_img, h, w):
    outputs = triton_infer(batch_img)
    detect = Detect(17 , bkg_label=0, top_k=200, conf_thresh=0.05, nms_thresh=0.5)
    preds = detect({'loc': torch.from_numpy(outputs['loc']), 'conf': torch.from_numpy(outputs['conf']), 'mask': torch.from_numpy(outputs['mask']),
                    'proto': torch.from_numpy(outputs['proto']), 'priors': torch.from_numpy(outputs['priors'])})
    
    squeeze_img = torch.Tensor(np.squeeze(batch_img, axis=0)) # RGB to BGR
    img_numpy, classes, scores = prep_display(preds, squeeze_img, h, w, undo_transform=True)
    cv2.imwrite('result.jpg', img_numpy)
    img_base64 = img_to_base64(img_numpy)

    create_db_query = f"SELECT CommonName, Link FROM FishMetaData WHERE Id={classes.tolist()[0]}"
    with connection.cursor() as cursor:
        cursor.execute(create_db_query)
        sql_result = cursor.fetchall()
        commonname = sql_result[0][0]
        link = sql_result[0][1]

    result = {'score': scores.tolist(), 'class': classes.tolist(), 'commonname': [commonname], 'link': [link], 'image': img_base64}

    return result


if __name__ == '__main__':
    # app.debug = True
    try:
        connection = connect(
            host="10.233.106.105",
            user=input("Enter username: "),
            password=getpass("Enter password: "),
            database="FishSpecies"
        )
    except Error as e:
        print(e)
    app.run(host='localhost')