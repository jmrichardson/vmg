## Imports for making predictions
import os
from keras.models import load_model
import cv2
import numpy as np
import pickle
import configparser
from datetime import datetime, time

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# Get config
config = configparser.ConfigParser()
config.read("config.ini")
rtsp = config.get("vars", "rtsp")

top_model_weights_path = 'spots.h5'
model = load_model(top_model_weights_path)

with open(r"spots.pickle", "rb") as file:
    spots = pickle.load(file)

class_dictionary = {}
class_dictionary[0] = 'empty'
class_dictionary[1] = 'occupied'

def date_diff_in_seconds(dt2, dt1):
    timedelta = dt2 - dt1
    return timedelta.days * 24 * 3600 + timedelta.seconds

def make_prediction(image):
    #Rescale image
    img = image/255.

    #Convert to a 4D tensor
    image = np.expand_dims(img, axis=0)
    #print(image.shape)

    # make predictions on the preloaded model
    class_predicted = model.predict(image)
    inID = np.argmax(class_predicted[0])
    label = class_dictionary[inID]
    return label


color = [0, 255, 0]
alpha = 0.5

while True:

    cnt_empty = 0
    all_spots = 0

    # cap = cv2.VideoCapture('rtsp://admin:ez4me2no@192.168.1.217:554/11')
    cap = cv2.VideoCapture('http://127.0.0.1:8080/')
    ret, image = cap.read()
    if not ret:
        print ("Error getting image...")
        continue
    del(cap)
    date1 = datetime.now()

    print("Captured parking lot image ..")

    new_image = np.copy(image)
    overlay = np.copy(image)

    for spot in spots.values():
        all_spots += 1
        (x1, y1, x2, y2) = spot

        # crop this image
        spot_img = image[y1:y2, x1:x2]
        spot_img = cv2.resize(spot_img, (48, 80))

        label = make_prediction(spot_img)
        # print(label)
        if label == 'empty':
            cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), color, -1)
            cnt_empty += 1

    cnt_occupied = all_spots - cnt_empty

    cv2.addWeighted(overlay, alpha, new_image, 1 - alpha, 0, new_image)

    cv2.imwrite('../static/img/parking_lot.jpg', new_image)

    template = """
        <div id="info" class="card-deck">
          <div class="card border-dark mb-3" style="max-width: 18rem;">
            <div class="card-header">Total Parking Spots</div>
            <div class="card-body text-dark">
              <strong><font size="70">{all_spots}</font></strong>
            </div>
          </div>
          <div class="card border-danger mb-3" style="max-width: 18rem;">
            <div class="card-header">Total Spots Occupied</div>
            <div class="card-body text-danger">
              <strong><font size="70">{cnt_occupied}</font></strong>
            </div>
          </div>
          <div class="card border-success mb-3" style="max-width: 18rem;">
            <div class="card-header">Total Spots Available</div>
            <div class="card-body text-success">
              <strong><font size="70">{cnt_empty}</font></strong>
            </div>
          </div>
        </div>
    """.format(
        all_spots =all_spots,
        cnt_occupied=cnt_occupied,
        cnt_empty=cnt_empty
    )
    print(template, file=open("../templates/spots.html", 'w'))

    date2 = datetime.now()
    print("Completed in %d seconds" % (date_diff_in_seconds(date2, date1)))

