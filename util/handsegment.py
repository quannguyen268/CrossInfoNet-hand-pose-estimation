import numpy as np
import cv2
boundaries = [
    ([0, 120, 0], [140, 255, 100]),
    ([25, 0, 75], [180, 38, 255])
]



#Segment hand from original image
def handsegment(frame):
    lower, upper = boundaries[0]
    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")
    mask1 = cv2.inRange(frame, lower, upper) #color 1

    lower, upper = boundaries[1]
    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")
    mask2 = cv2.inRange(frame, lower, upper) # color 2


    mask = cv2.bitwise_or(mask1, mask2) #take both color
    output = cv2.bitwise_and(frame, frame, mask=mask)

    return output



