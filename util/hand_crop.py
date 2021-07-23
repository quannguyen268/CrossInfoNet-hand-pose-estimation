import os
import csv
import cv2
import numpy as np
import tqdm
from imutils import contours,perspective, grab_contours
from scipy.spatial import distance as dist


# Crop hand out of original image
def one_hand(image_gray):
    thresh = cv2.threshold(image_gray, 0, 255,cv2.THRESH_OTSU)[1]
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=lambda z: cv2.contourArea(z), reverse=True)[0]
    (x, y, w, h) = cv2.boundingRect(cnts)
    roi = image_gray[y :y + h , x :x + w ]
    roi = cv2.resize(roi, (224,224), interpolation=cv2.INTER_AREA)
    D1=0
    return roi,D1

def image_to_hand(path_to_images, path_to_distance):
    f1 = open(path_to_images, 'r')
    f2 = open(path_to_distance, 'a')
    lists = f1.readlines()
    label_2hand = [29,31,32,34,35,36,43,44,45,48,49,50,51,53,54,55,56,57,58,60,61,63]
    for path_image in tqdm.tqdm(lists):
        path_image = path_image.split('\n')[0]
        imag = cv2.imread(path_image)
        gray = cv2.cvtColor(imag, cv2.COLOR_RGB2GRAY)
        image = path_image.split('/')[-1]
        label = int(image.split('_')[0])
        if label in label_2hand:
            img, distance = two_hand(gray)
            cv2.imwrite(path_image, img)
            f2.write(str(distance) + os.linesep)
        else:
            img, distance = one_hand(gray)
            cv2.imwrite(path_image, img)
            f2.write(str(distance) + os.linesep)



    f1.close()
    f2.close()



def frame_to_csv(path_to_frame, path_to_csv):
    # lists = os.listdir(path_to_frame)
    # print(lists)
    f = open(path_to_frame, 'r')
    lists = f.readlines()
    lists = lists[1:]
    with open(path_to_csv, 'a', newline='') as f:
        writer = csv.writer(f)
        for image in tqdm.tqdm(lists):
            image = image.split('\n')[0]
            img = cv2.imread(image)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            image = image.split('/')[-1]
            label = int(image.split('_')[0])
            value = np.asarray(img, dtype='uint8').reshape((img.shape[1], img.shape[0]))
            value = value.flatten()
            data = np.append(int(label), value)
            writer.writerow(data)


def two_hand(gray):
    def midpoint(ptA, ptB):
        return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


    blur = cv2.GaussianBlur(gray, (5, 5), 1)
    (T, thresh) = cv2.threshold(blur, 1, 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=lambda z: cv2.contourArea(z), reverse=True)[:2]
    distance = 0

    ref = np.zeros((224,224), dtype='uint8')
    coor = [[0, 24], [112, 24]]
    refObj = None
    for (c, p) in zip(cnts, coor):
        # compute bouding box for the contour then extract the digit
        (x, y, w, h) = cv2.boundingRect(c)
        px, py = p
        roi = gray[y:y + h, x:x + w]
        roi = cv2.resize(roi, (112,112), interpolation=cv2.INTER_AREA)
        ref[px:px + 112, py:py + 112] = roi
        box = cv2.minAreaRect(c)
        box = cv2.boxPoints(box)

        box = np.array(box, dtype="int")
        box = perspective.order_points(box)
        cX = np.average(box[:, 0])
        cY = np.average(box[:, 1])
        if refObj is None:
            # unpack the ordered bounding box, then compute the
            # midpoint between the top-left and top-right points,
            # followed by the midpoint between the top-right and
            # bottom-right
            (tl, tr, br, bl) = box

            (tlblX, tlblY) = midpoint(tl, bl)
            (trbrX, trbrY) = midpoint(tr, br)
            # compute the Euclidean distance between the midpoints,
            # then construct the reference object
            D = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

            refObj = (box, (cX, cY), D / 0.955)

            continue
        distance = dist.euclidean((cX, cY), (refObj[1][0], refObj[1][1])) / refObj[2]


    return ref,distance










path_train = '/home/quan/PycharmProjects/sign-language-gesture-recognition/utils/filename_train.txt'
path_test = '/home/quan/PycharmProjects/sign-language-gesture-recognition/utils/filename_test.txt'
Csv = '/home/quan/PycharmProjects/sign-language-gesture-recognition/data/train_sign.csv'










