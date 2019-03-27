import cv2
vidcap = cv2.VideoCapture('test.mp4')
success,image = vidcap.read()
count = 0
while success:
  cv2.imwrite("parking_lot.jpg", image)
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  count += 1