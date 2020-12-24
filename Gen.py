# GenData.py

import sys
import numpy as np
import cv2
import os

# module level variables 
MIN_CONTOUR_AREA = 100

RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30

def main():
    imgTrainingNumbers = cv2.imread("digit.png")   # read in training numbers image

    if imgTrainingNumbers is None:                          # if image was not read successfully
        print "error: image not read from file \n\n"        # print error message to std out
        os.system("pause")                                  # pause so user can see error message
        return                                              # and exit function (which exits program)

    Grayimg = cv2.cvtColor(imgTrainingNumbers, cv2.COLOR_BGR2GRAY)    # get grayscale image
    Blurimg = cv2.GaussianBlur(Grayimg, (5,5), 0)                  # get blur images

    # filter image from grayscale to black and white 
    # input image   
    # make pixels that pass the threshold full white    
    # use gaussian rather than mean, seems to give better resultsv
    # invert so foreground will be white, background will be black      
    # size of a pixel neighborhood used to calculate threshold value
    # constant subtracted from the mean or weighted mean
    imgThresh = cv2.adaptiveThreshold(Blurimg,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)                                  
                
    # show threshold image for reference
    cv2.imshow("Threshimg", imgThresh)   

    imgThreshCopy = imgThresh.copy()  
    # make a copy of the thresh image,in necessary modifies the image

    # input image, make sure to use a copy since the function will modify this image in the course of finding contours
    # retrieve the outermost contours only
    # compress horizontal, vertical, and diagonal segments and leave only their end points
    imgContours, npaContours, npaHierarchy = cv2.findContours(imgThreshCopy,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)           

    # declare empty numpy array, we will use this to write to file later
    # zero rows, enough cols to hold all image data
    npaFlattenedImages =  np.empty((0, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))
    # declare empty classifications list, this will be our list of how we are classifying our chars from user input, we will write to file at the end
    intClassifications = []        

    # possible digits we are interested in are digits 0 through 9, put these in list intValidChars
    intValiddigits = [ord('0'), ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), ord('6'), ord('7'), ord('8'), ord('9')]

    for npaContour in npaContours:                          # for each contour
        if cv2.contourArea(npaContour) > MIN_CONTOUR_AREA:          # if contour is big enough to consider
            [intX, intY, intW, intH] = cv2.boundingRect(npaContour)         # get and break out bounding rect

        # draw rectangle around each contour as we ask user for input
        # draw rectangle on original training image
        # upper left corner
        # lower right corner
        # red

            cv2.rectangle(imgTrainingNumbers,(intX, intY),(intX+intW,intY+intH),(0, 0, 255),2)

            # show training numbers image, this will now have red rectangles drawn on it
            # get key press

            imgROI = imgThresh[intY:intY+intH, intX:intX+intW]                                  # crop char out of threshold image
            imgROIResized = cv2.resize(imgROI, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))     # resize image, this will be more consistent for recognition and storage



            cv2.imshow("imgROI", imgROI)                    # show cropped out char for reference
            cv2.imshow("imgROIResized", imgROIResized)      # show resized image for reference
            cv2.imshow("digit.png", imgTrainingNumbers)      # show training numbers image, this will now have red rectangles drawn on it

            intDigit = cv2.waitKey(0)                     

            # if esc key was pressed
            if intDigit == 27:                  
                sys.exit()                      # exit program
            elif intDigit in intValiddigits:      # else if the char is in the list of chars we are looking for . . .

                # append classification char to integer list of chars (we will convert to float later before writing to file)
                intClassifications.append(intDigit)    

                # flatten image to 1d numpy array so we can write to file later
                npaFlattenedImage = imgROIResized.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT)) 

                # add current flattened impage numpy array to list of flattened image numpy arrays 
                npaFlattenedImages = np.append(npaFlattenedImages, npaFlattenedImage, 0)                    
            # end if
        # end if
    # end for

    # convert classifications list of ints to numpy array of floats
    fltClassifications = np.array(intClassifications, np.float32)  

    # flatten numpy array of floats to 1d so we can write to file later                 
    npaClassifications = fltClassifications.reshape((fltClassifications.size, 1))  

    print "\n\ntraining complete !!\n"

    np.savetxt("Digitclassifications.txt", npaClassifications)          # write flattened images to file
    np.savetxt("Digitflattenedimages.txt", npaFlattenedImages)          #

    cv2.destroyAllWindows()             # remove windows from memory

    return

if __name__ == "__main__":
    main()
# end if



