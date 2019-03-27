## Imports for making predictions
import os
from keras.models import load_model
import cv2
import numpy as np
import pickle
import configparser
from datetime import datetime, time
import multiprocessing
from functools import partial
# from keras import backend as K; K.set_session(K.tf.Session(config=K.tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)))


def init():
    global class_dictionary
    class_dictionary = {}
    class_dictionary[0] = 'empty'
    class_dictionary[1] = 'occupied'

    global model
    model = load_model('spots.h5')


def classify_image(image, spot):

    # from keras.models import load_model
    process = multiprocessing.current_process()

    # Get coordinates
    (x1, y1, x2, y2) = spot

    # crop the image to just parking spot
    spot_img = image[y1:y2, x1:x2]
    spot_img = cv2.resize(spot_img, (210, 380))
    spot_img = spot_img / 255
    spot_img = np.expand_dims(spot_img, axis=0)

    # Classify parking spot as empty or occupied
    start = datetime.now()
    class_predicted = model.predict(spot_img)
    now = datetime.now()
    print(str(process.pid) + ":" + str(now - start))

    inID = np.argmax(class_predicted[0])
    label = class_dictionary[inID]
    return label



if __name__ == '__main__':
    multiprocessing.freeze_support()

    first_iter = True
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

    # Get config
    config = configparser.ConfigParser()
    config.read("config.ini")
    rtsp = config.get("vars", "rtsp")


    with open(r"spots.pickle", "rb") as file:
        spots = pickle.load(file)


    color = [0, 255, 0, 255]
    alpha = 0.5
    # height = 720
    # width = 1280
    # num_cores = multiprocessing.cpu_count()
    # num_cores = 4
    # print(num_cores)


    while True:

        cnt_empty = 0
        all_spots = 0

        # cap = cv2.VideoCapture(rtsp)
        # ret, image = cap.read()
        image = cv2.imread('parking_lot.jpg')
        # if not ret:
            # print ("Error getting image...")
            # continue
        # del(cap)

        print("Captured parking lot image ..")

        # new_image = np.copy(image)
        overlay = np.copy(image)
        # overlay = np.zeros((height, width, 4), dtype=np.uint8)

        # results = Parallel(n_jobs=num_cores)(delayed(classify_image)(spot, image) for spot in spots.values())

        if first_iter:
            pool = multiprocessing.Pool(processes=4, initializer=init)
            first_iter = False


        date1 = datetime.now()
        # classify = partial(classify_image, model, image)
        classify = partial(classify_image, image)
        results = pool.map(classify, spots.values())
        print(results)
        date2 = datetime.now()
        print(str(date2 - date1))


        # for spot in spots.values():
        continue

        print(label)
        if label == 'empty':
            cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), color, -1)
            cnt_empty += 1

        cnt_occupied = all_spots - cnt_empty

        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

        cv2.imwrite('parking_lot_new.jpg', image)

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
        print(str(date2 - date1))

