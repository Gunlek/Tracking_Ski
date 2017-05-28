## Copyrights SimpleDuino 2017 ##
## Developper: Fabien Aubret ##
## Latest Edit: 27/05/2017 - 03:12

import pymedia.muxer as muxer
import pymedia.video.vcodec as vcodec
import pygame
import Image
import numpy as np
import matplotlib.pyplot as plt

## following functin from official pymedia doc ##
## http://pymedia.org/tut/src/dump_video.py.html ##
def dumpVideo(inputFile, outputFile, fmt):
    dm = muxer.Demuxer(inputFile.split('.')[-1])
    i = 1
    f = open(inputFile, 'rb')
    s = f.read(400000)
    r = dm.parse(s)
    v = filter(lambda x: x['type'] == muxer.CODEC_TYPE_VIDEO, dm.streams)
    if len(v)==0:
        raise 'No video stream in file'

    v_id = v[0]['index']
    print 'Video stream at %d index' % v_id
    c = vcodec.Decoder(dm.streams[v_id])
    while len(s)>0:
        for fr in r:
            if fr[0] == v_id:
                d = c.decode(fr[1])

                #Save file as RGB .bmp

                if d:
                    dd = d.convert(fmt)
                    img = pygame.image.fromstring(dd.data, dd.size, "RGB")
                    pygame.image.save(img, outputFile % i)
                    i+=1
        s = f.read(400000)
        r = dm.parse(s)
    print 'Saved %d frames' % i

#Convert .avi video to image sequence
#dumpVideo('../video/deformation_ski.avi', '../img/deformationSki_%d.bmp', 2)

#Convert image to black and white image
def toBlackAndWhite(srcImage, resultImg):
    img = Image.open(srcImage)
    imArray = np.array(img)
    for line in range(0, imArray.shape[0]):
        for column in range(0, imArray.shape[1]):
            adjust = 10
            if (imArray[line][column][0] > imArray[line][column][1]+adjust or imArray[line][column][0] < imArray[line][column][1]-adjust) or (imArray[line][column][0] > imArray[line][column][2]+adjust or imArray[line][column][0] < imArray[line][column][2]-adjust) or (imArray[line][column][1] > imArray[line][column][2]+adjust or imArray[line][column][1] < imArray[line][column][2]-adjust):
                imArray[line][column][0] = 0
                imArray[line][column][1] = 0
                imArray[line][column][2] = 0
    for line in range(0, imArray.shape[0]):
        for column in range(0, imArray.shape[1]):
            if imArray[line][column][0] > 100:
                imArray[line][column][0] = 255
                imArray[line][column][1] = 255
                imArray[line][column][2] = 255
            else:
                imArray[line][column][0] = 0
                imArray[line][column][1] = 0
                imArray[line][column][2] = 0
    finalImg = Image.fromarray(imArray[25:])
    finalImg.save(resultImg)

#Localize landmark at [index] from left to write (index=0 => leftmost)
def localizeLandmark(index, srcImage, resultImg):
    img = Image.open(srcImage)
    imArray = np.array(img)
    LatestLine = 0
    LatestColumn = 0
    detected = 0
    for line in range(0, imArray.shape[0]):
        for column in range(0, imArray.shape[1]):
            if imArray[line][column][0] == 0:
                if line > 10 and line < imArray.shape[0]-10 and column > 10 and column < imArray.shape[1]-10:
                    isThereLeft = False
                    isThereRight = False
                    isThereTop = False
                    isThereBottom = False
                    for temp_column in range(column-10, column):
                        if imArray[line][temp_column][0] == 255:
                            isThereLeft = True
                    for temp_line in range(line - 10, line):
                        if imArray[temp_line][column][0] == 255:
                            isThereTop = True
                    for temp_column in range(column, column+10):
                        if imArray[line][temp_column][0] == 255:
                            isThereRight = True
                    for temp_line in range(line, line + 10):
                        if imArray[temp_line][column][0] == 255:
                            isThereBottom = True
                    #and abs(line-LatestChanged[0][0])>100 and abs(column-LatestChanged[0][1])>100
                    if isThereLeft and isThereRight and isThereTop and isThereBottom:
                        imArray[line][column][0] = 241
                        imArray[line][column][1] = 196
                        imArray[line][column][2] = 15
    for line in range(0, imArray.shape[0]):
        for column in range(0, imArray.shape[1]):
            if imArray[line][column][0] == 255:
                imArray[line][column][0] = 0
                imArray[line][column][1] = 0
                imArray[line][column][2] = 0
    for line in range(0, imArray.shape[0]):
        for column in range(0, imArray.shape[1]):
            if imArray[line][column][0] > 0:
                if detected > 0 and abs(LatestColumn-column) > 10:
                    imArray[line][column][0] = 255
                    imArray[line][column][1] = 255
                    imArray[line][column][2] = 255
                    LatestColumn = column
                    LatestLine = line
                    detected+=1
                if detected==0 and abs(LatestColumn-column) > 10 and abs(LatestLine-line) > 25:
                    imArray[line][column][0] = 255
                    imArray[line][column][1] = 255
                    imArray[line][column][2] = 255
                    LatestColumn = column
                    LatestLine = line
                    detected += 1
    for line in range(0, imArray.shape[0]):
        for column in range(0, imArray.shape[1]):
            if imArray[line][column][0] > 0 and imArray[line][column][0]<255:
                imArray[line][column][0] = 0
                imArray[line][column][1] = 0
                imArray[line][column][2] = 0
    for column in range(0, imArray.shape[1]):
        for line in range(0, imArray.shape[0]):
            if imArray[line][column][0] == 255:
                stoLine = line
                stoCol = column
                for temp_line in range(line-10, line+10):
                    for temp_column in range(column-10, column+10):
                        imArray[temp_line][temp_column][0] = 0
                        imArray[temp_line][temp_column][1] = 0
                        imArray[temp_line][temp_column][2] = 0
                imArray[stoLine][stoCol][0] = 192
                imArray[stoLine][stoCol][1] = 57
                imArray[stoLine][stoCol][2] = 43
    coords = []
    for column in range (0, imArray.shape[1]):
        for line in range(0, imArray.shape[0]):
            if imArray[line][column][0] == 192:
                coords.append([line, column])
    finalImg = Image.fromarray(imArray)
    finalImg.save(resultImg)

    return coords[index]

fullArrayX = []
fullArrayY = []
#toBlackAndWhite('../img/deformationSki_30.bmp', '../after_treatment/after.bmp')
#localizeLandmark(0, '../after_treatment/after.bmp', '../after_treatment/after_highligthing.bmp')
for i in range(83, 250):
    toBlackAndWhite('../img/deformationSki_%d.bmp' % i, '../after_treatment/after.bmp')
    tempArray = [i, localizeLandmark(0, '../after_treatment/after.bmp', '../after_treatment/after_highligthing.bmp')]
    fullArrayX.append(tempArray[0])
    fullArrayY.append(tempArray[1][0])
    print i

plt.plot(fullArrayX, fullArrayY)
plt.show()


