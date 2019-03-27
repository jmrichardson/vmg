from django.shortcuts import render
from django.http import StreamingHttpResponse, HttpResponse
from keras.models import load_model
import cv2
import glob, time
from django.views.decorators import gzip
import numpy as np
import pickle
from .models import Stats
from datetime import datetime

# Load trained parking spot locator model
top_model_weights_path = 'parking_spot/spots.h5'
model = load_model(top_model_weights_path)
model._make_predict_function()

# Load predefined parking spot locations
with open(r"parking_spot/spots.pickle", "rb") as f:
    spots = pickle.load(f)

# Define parking spot classification labels
class_dictionary = {}
class_dictionary[0] = 'empty'
class_dictionary[1] = 'occupied'

# Define variables
color = [0, 255, 0]
alpha = 0.5
spots_total = 38

# Set defaults
stat = Stats.objects.first()
stat.spots_empty = 0
stat.spots_total = 0
stat.spots_occupied = 0
stat.save()

# Process video camera image
def parking_lot_jpeg():


    warmup = True

    # print("Captured parking lot image ..")

    while True:

        # Capture video image from video stream
        cap = cv2.VideoCapture('rtsp://admin:ez4me2no@192.168.1.10:554/11')
        ret, image = cap.read()
        if not ret:
            print("Error getting image...")
            continue
        del(cap)

        print("Processing image ...")

        # Warmup browser with image
        if warmup:
            frame = cv2.imencode('.jpg', image)[1].tobytes()
            yield (b'--frame\r\n' + b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
            yield (b'--frame\r\n' + b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
            warmup = False

        date_image_start = datetime.now()

        # Initialize count variables
        spots_empty = 0

        # Copy image for overlay
        image_overlay = np.copy(image)

        # Loop through all defined spots
        for spot in spots.values():

            # Get spot coordinates and assign to x,y coordinates
            (x1, y1, x2, y2) = spot

            # Crop the image to the particular parking spot
            image_spot = image[y1:y2, x1:x2]

            # Resize spot so all spots have the same size and rescale
            # image_spot = cv2.resize(image_spot, (210, 380))
            # image_spot = cv2.resize(image_spot, (32, 32))
            image_spot = cv2.resize(image_spot, (105, 190))

            # Normalize RGB values 255/255/255 to 0-1/0-1/0-1
            image_scale = image_spot/255

            # Convert to a 4D tensor
            image_tf = np.expand_dims(image_scale, axis=0)

            # Classify parking spot image and label
            spot_class = model.predict(image_tf)

            inID = np.argmax(spot_class[0])
            label = class_dictionary[inID]
            # print(label)

            # Highlight all empty parking spots on overlay image
            if label == 'empty':
                cv2.rectangle(image_overlay, (int(x1), int(y1)), (int(x2), int(y2)), color, -1)
                # Count empty spots
                spots_empty += 1

        # Total occupied parking spots
        spots_occupied = spots_total - spots_empty

        # Add overlay to image
        cv2.addWeighted(image_overlay, alpha, image, 1 - alpha, 0, image)

        # Encode image to bytes
        image = cv2.imencode('.jpg', image)[1].tobytes()

        date_image_end = datetime.now()

        print("Total time: " + str(date_image_end - date_image_start))

        # Return image
        yield (b'--frame\r\n' + b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n\r\n')
        yield (b'--frame\r\n' + b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n\r\n')

        # Update spot stats
        stat = Stats.objects.first()
        stat.spots_empty = spots_empty
        stat.spots_total = spots_total
        stat.spots_occupied = spots_occupied
        stat.save()
        # time.sleep(1)


@gzip.gzip_page
def parking_lot_stream(request):
    response = StreamingHttpResponse(parking_lot_jpeg(), content_type='multipart/x-mixed-replace;boundary=frame')
    response['Cache-Control'] = 'no-cache'
    return response


# Get spot stats
def spots_occupied_text():

    while True:
        time.sleep(.001)
        stat = Stats.objects.first()
        yield "data: " + str(stat.spots_occupied) + "\r\n\r\n"


# Stream spots stats
def spots_occupied_stream(request):
    response = StreamingHttpResponse(spots_occupied_text(), content_type='text/event-stream')
    response['Cache-Control'] = 'no-cache'
    return response


# Get spot stats
def spots_empty_text():

    while True:
        time.sleep(.001)
        stat = Stats.objects.first()
        yield "data: " + str(stat.spots_empty) + "\r\n\r\n"


# Stream spots stats
def spots_empty_stream(request):
    response = StreamingHttpResponse(spots_empty_text(), content_type='text/event-stream')
    response['Cache-Control'] = 'no-cache'
    return response


# Render home page template
def home(request):
    return render(request, 'home.html')



