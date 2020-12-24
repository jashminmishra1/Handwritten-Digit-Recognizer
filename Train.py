import cv2
import numpy as np
import operator
import os

# module level variables 
MIN_CONTOUR_AREA = 100

RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30


class ContourWithData():

    # member variables ############################################################################
    npaContour = None           # contour
    boundingRect = None         # bounding rect for contour
    intRectX = 0                # bounding rect top left corner x location
    intRectY = 0                # bounding rect top left corner y location
    intRectWidth = 0            # bounding rect width
    intRectHeight = 0           # bounding rect height
    fltArea = 0.0               # area of contour

    def calculate(self):               # calculate bounding rect info
        [intX, intY, intWidth, intHeight] = self.boundingRect
        self.intRectX = intX
        self.intRectY = intY
        self.intRectWidth = intWidth
        self.intRectHeight = intHeight

     # this is oversimplified, for a production grade program
    def ValidContour(self): 
         # much better validity checking would be necessary                          
        if self.fltArea < MIN_CONTOUR_AREA: return False       
        return True

def recognise_image(data):
    allContoursWithData = []                # declare empty lists,
    validContoursWithData = []              # we will fill these shortly

    try:
         # read in training classifications
        npaClassifications = np.loadtxt("Digitclassifications.txt", np.float32)
    except:
        print("error, unable to open classifications.txt, exiting program\n")
        os.system("pause")
        return
    # end try

    try:
        # read in training images
        npaFlattenedImages = np.loadtxt("Digitflattenedimages.txt", np.float32)
    except:
        print("error, unable to open flattened_images.txt, exiting program\n")
        os.system("pause")
        return
    # end try

     # reshape numpy array to 1d, necessary to pass to call to train
    npaClassifications = npaClassifications.reshape(
        (npaClassifications.size, 1))
    # instantiate KNN object
    kNearest = cv2.ml.KNearest_create()

    kNearest.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)

    # read in testing numbers image
    imgTestingNumbers = cv2.imread(data)
    os.remove(data)

    if imgTestingNumbers is None:                           # if image was not read successfully
        # print error message to std out
        print("error: image not read from file \n\n")
        # pause so user can see error message
        os.system("pause")
        # and exit function (which exits program)
        return
    # end if

    # get grayscale image
    Grayimg = cv2.cvtColor(imgTestingNumbers, cv2.COLOR_BGR2GRAY)
    Blurimg = cv2.GaussianBlur(Grayimg, (5, 5), 0)                    # blur

    # filter image from grayscale to black and white
    # input image
    # make pixels that pass the threshold full white
    # invert so foreground will be white, background will be black
    # use gaussian rather than mean, seems to give better resultsv
    # size of a pixel neighborhood used to calculate threshold value
    # constant subtracted from the mean or weighted mean
    imgThresh = cv2.adaptiveThreshold(
        Blurimg, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # make a copy of the thresh image, this in necessary b/c findContours modifies the image
    imgThreshCopy = imgThresh.copy()

    # input image, make sure to use a copy since the function will modify this image in the course of finding contours
    # retrieve the outermost contours only
    # compress horizontal, vertical, and diagonal segments and leave only their end points
    # imgContours, npaContours, npaHierarchy = cv2.findContours(imgThreshCopy,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    npaContours, npaHierarchy = cv2.findContours(
        imgThreshCopy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for npaContour in npaContours:                             # for each contour
        # instantiate a contour with data object
        contourWithData = ContourWithData()
        # assign contour to contour with data
        contourWithData.npaContour = npaContour
        contourWithData.boundingRect = cv2.boundingRect(
            contourWithData.npaContour)     # get the bounding rect
        contourWithData.calculate()                    # get bounding rect info
        contourWithData.fltArea = cv2.contourArea(
            contourWithData.npaContour)           # calculate the contour area
        # add contour with data object to list of all contours with data
        allContoursWithData.append(contourWithData)
    # end for

    for contourWithData in allContoursWithData:                 # for all contours
        if contourWithData.ValidContour():             # check if valid
            # if so, append to valid contour list
            validContoursWithData.append(contourWithData)
        # end if
    # end for

    # sort contours from left to right
    validContoursWithData.sort(key=operator.attrgetter("intRectX"))

    # declare final string, this will have the final number sequence by the end of the program
    strFinalString = ""

    for contourWithData in validContoursWithData:            # for each contour
        # draw a green rect around the current char

        # draw rectangle on original testing image
        # upper left corner
        # lower right corner
         # green
         # thickness
        cv2.rectangle(imgTestingNumbers, (contourWithData.intRectX, contourWithData.intRectY), (contourWithData.intRectX +
                                                                                                contourWithData.intRectWidth, contourWithData.intRectY + contourWithData.intRectHeight), (0, 255, 0), 2)

        # crop char out of threshold image
        imgROI = imgThresh[contourWithData.intRectY: contourWithData.intRectY + contourWithData.intRectHeight,
                           contourWithData.intRectX: contourWithData.intRectX + contourWithData.intRectWidth]

        # resize image, this will be more consistent for recognition and storage
        imgROIResized = cv2.resize(
            imgROI, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))

        # flatten image into 1d numpy array
        npaROIResized = imgROIResized.reshape(
            (1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))

        # convert from 1d numpy array of ints to 1d numpy array of floats
        npaROIResized = np.float32(npaROIResized)

        retval, npaResults, neigh_resp, dists = kNearest.findNearest(
            npaROIResized, k=1)     # call KNN function find_nearest

        # get character from results
        strCurrentChar = str(chr(int(npaResults[0][0])))

        # append current char to full string
        strFinalString = strFinalString + strCurrentChar
    # end for

    print("\n" + strFinalString + "\n")                  # show the full string
    return strFinalString

    # show input image with green boxes drawn around found digits
    cv2.imshow("imgTestingNumbers", imgTestingNumbers)
    # wait for user key press
    cv2.waitKey(0)

    cv2.destroyAllWindows()             # remove windows from memory

    return

def main():
    allContoursWithData = []                # declare empty lists,
    validContoursWithData = []              # we will fill these shortly

    try:
         # read in training classifications
        npaClassifications = np.loadtxt("Digitclassifications.txt", np.float32)                 
    except:
        print ("error, unable to open classifications.txt, exiting program\n")
        os.system("pause")
        return
    # end try

    try:
        # read in training images
        npaFlattenedImages = np.loadtxt("Digitflattenedimages.txt", np.float32)                 
    except:
        print ("error, unable to open flattened_images.txt, exiting program\n")
        os.system("pause")
        return
    # end try

     # reshape numpy array to 1d, necessary to pass to call to train
    npaClassifications = npaClassifications.reshape((npaClassifications.size, 1))      
    # instantiate KNN object
    kNearest = cv2.ml.KNearest_create()                   


    kNearest.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)

    imgTestingNumbers = cv2.imread(data)          # read in testing numbers image

    if imgTestingNumbers is None:                           # if image was not read successfully
        print ("error: image not read from file \n\n")        # print error message to std out
        os.system("pause")                                  # pause so user can see error message
        return                                              # and exit function (which exits program)
    # end if

    Grayimg = cv2.cvtColor(imgTestingNumbers, cv2.COLOR_BGR2GRAY)       # get grayscale image
    Blurimg = cv2.GaussianBlur(Grayimg, (5,5), 0)                    # blur

    # filter image from grayscale to black and white                                                   
    # input image   
    # make pixels that pass the threshold full white    
    # use gaussian rather than mean, seems to give better resultsv
    # invert so foreground will be white, background will be black      
    # size of a pixel neighborhood used to calculate threshold value
    # constant subtracted from the mean or weighted mean
    imgThresh = cv2.adaptiveThreshold(Blurimg,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)

    imgThreshCopy = imgThresh.copy()        # make a copy of the thresh image, this in necessary b/c findContours modifies the image

    # input image, make sure to use a copy since the function will modify this image in the course of finding contours
    # retrieve the outermost contours only
    # compress horizontal, vertical, and diagonal segments and leave only their end points
    # imgContours, npaContours, npaHierarchy = cv2.findContours(imgThreshCopy,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) 
    npaContours, npaHierarchy = cv2.findContours(imgThreshCopy,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) 

    for npaContour in npaContours:                             # for each contour
        contourWithData = ContourWithData()                                             # instantiate a contour with data object
        contourWithData.npaContour = npaContour                                         # assign contour to contour with data
        contourWithData.boundingRect = cv2.boundingRect(contourWithData.npaContour)     # get the bounding rect
        contourWithData.calculate()                    # get bounding rect info
        contourWithData.fltArea = cv2.contourArea(contourWithData.npaContour)           # calculate the contour area
        allContoursWithData.append(contourWithData)                                     # add contour with data object to list of all contours with data
    # end for

    for contourWithData in allContoursWithData:                 # for all contours
        if contourWithData.ValidContour():             # check if valid
            validContoursWithData.append(contourWithData)       # if so, append to valid contour list
        # end if
    # end for

    # sort contours from left to right
    validContoursWithData.sort(key = operator.attrgetter("intRectX"))         

     # declare final string, this will have the final number sequence by the end of the program
    strFinalString = ""        

    for contourWithData in validContoursWithData:            # for each contour
        # draw a green rect around the current char

        # draw rectangle on original testing image
        # upper left corner
        # lower right corner
         # green
         # thickness
        cv2.rectangle(imgTestingNumbers,(contourWithData.intRectX, contourWithData.intRectY),(contourWithData.intRectX + contourWithData.intRectWidth, contourWithData.intRectY + contourWithData.intRectHeight),(0, 255, 0),2)                        

        # crop char out of threshold image
        imgROI = imgThresh[contourWithData.intRectY : contourWithData.intRectY + contourWithData.intRectHeight,contourWithData.intRectX : contourWithData.intRectX + contourWithData.intRectWidth]

        imgROIResized = cv2.resize(imgROI, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))             # resize image, this will be more consistent for recognition and storage

        npaROIResized = imgROIResized.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))      # flatten image into 1d numpy array

        npaROIResized = np.float32(npaROIResized)       # convert from 1d numpy array of ints to 1d numpy array of floats

        retval, npaResults, neigh_resp, dists = kNearest.findNearest(npaROIResized, k = 1)     # call KNN function find_nearest

        strCurrentChar = str(chr(int(npaResults[0][0])))                                             # get character from results

        strFinalString = strFinalString + strCurrentChar            # append current char to full string
    # end for

    print ("\n" + strFinalString + "\n")                  # show the full string

    cv2.imshow("imgTestingNumbers", imgTestingNumbers)      # show input image with green boxes drawn around found digits
    cv2.waitKey(0)                                          # wait for user key press

    cv2.destroyAllWindows()             # remove windows from memory

    return

###################################################################################################
if __name__ == "__main__":
    main()
# end if
