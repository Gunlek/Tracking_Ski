import Image
import cv2
import imutils
import matplotlib.pyplot as plt
import numpy as np

# 1 image = 0.03 secondes de video

refreshRate = 0.5
secondsToStudy = 60+52

timeConstant = 0.033
#framePerStep = int(timeConstant*secondsToStudy)+1
realRefreshRate = int(refreshRate/timeConstant)
framesToStudy = secondsToStudy/timeConstant-20

def cutImage(srcImage, resultImg, inf, sup):
    img = Image.open(srcImage)
    imArray = np.array(img)
    columns = range(inf, sup+1)
    tempArray = imArray[:, columns]
    finalImg = Image.fromarray(tempArray)
    finalImg.save(resultImg)

def cutImageHeight(srcImage, resultImg, inf, sup):
    img = Image.open(srcImage)
    imArray = np.array(img)
    lines = range(inf, sup+1)
    tempArray = imArray[lines, :]
    finalImg = Image.fromarray(tempArray)
    finalImg.save(resultImg)

def detect(c):
    shape = "unidentified"
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.04*peri, True)

    # Si c'est un triangle ==> 3 sommets
    if len(approx) == 3:
        shape = "triangle"

    # Si c'est un rectangle ==> 4 sommets
    elif len(approx) == 4:
        # Calcul les contour et dimension du rectangle
        (x, y, w, h) = cv2.boundingRect(approx)
        ar = w / float(h)

        # Si le ratio longueur / largeur est environ egale a 1 ==> carre
        shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"

    # Si c'est un pentagone ==> 5 sommets
    elif len(approx) == 5:
        shape = "pentagon"

    # Sinon c'est un cercle
    else:
        shape = "circle"

    # renvoie le nom de la forme
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
        # Calcul le centre du contour obtenu et detect le nom de la forme d'apres son contour
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

            # Multiplie les dimensions x/y par le ratio de dimension d'image
            # Dessign le contour et le nom de la forme sur l'image
            c = c.astype("float")
            c *= ratio
            c = c.astype("int")
            cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
            cv2.putText(image, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 2)

            #cv2.imshow("Image", image)
            #cv2.waitKey(0)
    return array

def doCuts(img):
    cutImage(img, "marqueur_1.bmp", 75, 110)
    cutImage(img, "marqueur_2.bmp", 110, 150)
    cutImage(img, "marqueur_3.bmp", 150, 200)
    cutImage(img, "marqueur_4.bmp", 210, 255)
    cutImage(img, "marqueur_5.bmp", 260, 305)
    cutImage(img, "marqueur_6.bmp", 315, 355)
    cutImage(img, "marqueur_7.bmp", 365, 405)
    cutImage(img, "marqueur_8.bmp", 420, 460)
    cutImage(img, "marqueur_9.bmp", 470, 510)
    cutImage(img, "marqueur_10.bmp", 530, 570)
    cutImage(img, "marqueur_11.bmp", 580, 620)
    cutImageHeight("marqueur_11.bmp", "marqueur_11.bmp", 50, 120) # y=0.76x+50
    cutImage(img, "marqueur_12.bmp", 625, 665)
    cutImageHeight("marqueur_12.bmp", "marqueur_12.bmp", 50, 120) # y=0.76x+50
    cutImage(img, "marqueur_13.bmp", 675, 715)
    cutImageHeight("marqueur_13.bmp", "marqueur_13.bmp", 0, 120) # y=142x/120
    cutImage(img, "marqueur_14.bmp", 735, 775)
    cutImageHeight("marqueur_14.bmp", "marqueur_14.bmp", 40, 110) # y=0.85x+40
    cutImage(img, "marqueur_15.bmp", 785, 825)
    cutImageHeight("marqueur_15.bmp", "marqueur_15.bmp", 0, 120) # y=142x/120

def getVertices(X, Y):
    verticesX = []
    verticesY = []
    oldValue = 0
    topReached = False

    for i in range(20, len(Y)-20):
        if (Y[i] >= Y[i-1] and Y[i] > Y[i+1]) or (Y[i] <= Y[i-1] and Y[i] < Y[i+1]):
            verticesY.append(Y[i])
            verticesX.append(X[Y.index(Y[i])])
    return [verticesX, verticesY]

X=[]
Y=[]

marks = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]
#arr = getCenters("temp.bmp")
#print arr


#Debug only
framesToStudy = 200
Xorigin = 0

for i in range(1, int(framesToStudy)):
    doCuts("../img/deformationSki_%d.bmp" % i)
    for j in range(1, 2):
        data = getCenters("marqueur_%d.bmp" % j)
        if data[0] == [0, 0]:
            data[0] = data[1]
        if j == 11 or j == 12:
            data[0][1]=int(data[0][1]*0.76+50)
        if j == 13 or j == 15:
            data[0][1]=int(data[0][1]*142/120)
        if j == 14:
            data[0][1]=int(data[0][1]*0.85+40)
        if i==1:
            Xorigin = data[0][1]
        data[0][1] = data[0][1]-Xorigin
        marks[j-1].append(data[0])
    X.append(i*timeConstant)
    print i

for k in range(0, 1):
    for j in range(0, int(framesToStudy)-1):
        Y.append(marks[k][j][1])
    #plt.plot(X, Y)
    #Y = []

plt.plot(X, Y)
vertices = getVertices(X, Y)
verticesX = vertices[0]
verticesY = vertices[1]

for i in range(0, len(verticesX)):
    tempX = [verticesX[i], verticesX[i]]
    tempY = [0, verticesY[i]]
    plt.plot(tempX, tempY)

#finalArray = [[], []]
#for j in range(1, len(Y)):
#    if abs(Y[j]-Y[j-1]) < 1:
#        finalArray[0].append(X[j])
#        finalArray[1].append(Y[j])
plt.show()