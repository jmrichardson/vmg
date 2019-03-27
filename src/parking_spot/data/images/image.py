import os
import cv2
import argparse

# RTSP stream of IP camera
rtsp = 'rtsp://admin:ez4me2no@192.168.1.10:554/11'

# Parse arguments
parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("--test", help="Save image to test directory", action="store_true")
group.add_argument("--train", help="Save image to train directory", action="store_true")
group.add_argument("--labelimg", help="Save image to labelimg directory", action="store_true")
args = parser.parse_args()
if args.labelimg:
    subdir = 'labelimg'
elif args.test:
    subdir = 'test'
else:
    subdir = 'train'

# Set working directory to script location
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
print(dname)

# Set file name of image (increasing name by 1)
cnt = 1
while True:
    file = os.path.join(dname, subdir, '') + 'image' + str(cnt) + '.jpg'
    if os.path.isfile(file):
        cnt = cnt+1
        continue
    else:
        break

# Capture image from camera and save to file
cap = cv2.VideoCapture(rtsp)
ret, image = cap.read()
print(ret)
print(file)
cv2.imwrite(file, image)
del(cap)

