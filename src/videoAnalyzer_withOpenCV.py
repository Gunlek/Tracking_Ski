import Image
import cv2
import argparse
import imutils
import matplotlib.pyplot as plt
import numpy as np

def cutImage(srcImage, inf, sup):
    img = Image.open(srcImage)
    imArray = np.array(img)
    columns = range(inf, sup+1)
    tempArray = imArray[:, columns]
    finalImg = Image.fromarray(tempArray)
    finalImg.save("temp.bmp")

def detect(c):
    shape = "unidentified"
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.04*peri, True)

    # if the shape is a triangle, it will have 3 vertices
    if len(approx) == 3:
        shape = "triangle"

    # if the shape has 4 vertices, it is either a square or
    # a rectangle
    elif len(approx) == 4:
        # compute the bounding box of the contour and use the
        # bounding box to compute the aspect ratio
        (x, y, w, h) = cv2.boundingRect(approx)
        ar = w / float(h)

        # a square will have an aspect ratio that is approximately
        # equal to one, otherwise, the shape is a rectangle
        shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"

    # if the shape is a pentagon, it will have 5 vertices
    elif len(approx) == 5:
        shape = "pentagon"

    # otherwise, we assume the shape is a circle
    else:
        shape = "circle"

    # return the name of the shape
    return shape

def getCenters(img):
    image = cv2.imread(img)
    resized = imutils.resize(image, width=image.shape[1])
    ratio = image.shape[0] / float(resized.shape[0])

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]

    array = []
    for c in cnts:
        # compute the center of the contour, then detect the name of the
        # shape using only the contour
        M = cv2.moments(c)
        if M["m00"]!=0:
            cX = int((M["m10"] / M["m00"]) * ratio)
            cY = int((M["m01"] / M["m00"]) * ratio)
        else:
            cX = 0
            cY = 0

        array.append([cX, cY])

        shape = detect(c)
        if shape == "circle":

            # multiply the contour (x, y)-coordinates by the resize ratio,
            # then draw the contours and the name of the shape on the image
            c = c.astype("float")
            c *= ratio
            c = c.astype("int")
            cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
            cv2.putText(image, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 2)

            #cv2.imshow("Image", image)
            #cv2.waitKey(0)
    return array

def rearrange_positions(coords):
    coordsColSorted = []
    for i in range(0, len(coords)):
        temp_array = []
        for pos in range(0, len(coords)):
            if coords[pos][1] != 1:
                temp_array.append(coords[pos][1])
        temp_array = sorted(temp_array)
        temp_coords = []
        for k in range(0, len(coords)):
            if coords[k][1]!=1:
                temp_coords.append(coords[k])
        coordsColSorted = []
        for c in range(0, len(temp_array)):
            for j in range(0, len(temp_coords)):
                if temp_coords[j][1] == temp_array[c]:
                    coordsColSorted.append(temp_coords[k])
    return coordsColSorted


X=[]
Y=[]
cutImage("../img/deformationSki_650.bmp", 75, 110)
arr = getCenters("temp.bmp")
print arr

for i in range(1, 2000):
    print i
    cutImage("../img/deformationSki_%d.bmp" % i, 75, 110)
    arr = getCenters("temp.bmp")
    print arr
    X.append(i)
    Y.append(arr[-1][1])

#finalArray = [[], []]
#for j in range(1, len(Y)):
#    if abs(Y[j]-Y[j-1]) < 1:
#        finalArray[0].append(X[j])
#        finalArray[1].append(Y[j])
plt.plot(X, Y)
plt.show()