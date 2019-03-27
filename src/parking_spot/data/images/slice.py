import os
import xml.etree.ElementTree
import cv2
import pickle
import configparser
import argparse

# config = configparser.ConfigParser()
# config.read("../../config.ini")
# homeDir = config.get("vars", "homeDir")
# os.chdir(os.path.join(homeDir, 'data/images'))

abspath = os.path.abspath(__file__)
scriptDir = os.path.dirname(abspath)
os.chdir(scriptDir)

# Parse arguments
parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("--test", help="Save image to test directory", action="store_true")
group.add_argument("--train", help="Save image to train directory", action="store_true")
args = parser.parse_args()
if args.test:
    path = os.path.join(scriptDir, '../test', '')
    dir = "test"
else:
    path = os.path.join(scriptDir, '../train', '')
    dir = "train"

# Parse xml of labelImg
root = xml.etree.ElementTree.parse('labelimg/image1.xml').getroot()

# Get annotation name and coordinates
spots = {}
for object in root.findall('./object'):
    name = object.find('name').text
    xmin = int(object.find('./bndbox/xmin').text)
    ymin = int(object.find('./bndbox/ymin').text)
    xmax = int(object.find('./bndbox/xmax').text)
    ymax = int(object.find('./bndbox/ymax').text)

    print(xmin, ymin, xmax, ymax)
    spots[name] = [xmin, ymin, xmax, ymax]

    for file in os.listdir(dir):
        filename = os.fsdecode(file)
        if filename.endswith(".jpg"):
            img = cv2.imread(dir + '/' + filename)
            spot_img = img[ymin:ymax, xmin:xmax]
            cv2.imwrite(path + name + '_' + filename, spot_img)
            print(path + name + '_' + filename)

with open('../../spots.pickle', 'wb') as handle:
    pickle.dump(spots, handle, protocol=pickle.HIGHEST_PROTOCOL)

