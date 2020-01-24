"""
Submission for assignment 2
"""
import cv2

import numpy as np

import math
from math import sin, cos, radians

def blur_image_gaussian(image, kernel=(5,5)):  # Can increase kernel to blur image more
    """returns the image that has been blurred using a gaussian filter
    you should determing appropriate values for the filter to be used
    See here:
    https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_filtering/py_filtering.html

    Input:  an image

    Output:  numpy.array:  the blurred image
    """
    return cv2.GaussianBlur(image, kernel, 0)

def shifted_difference(image, left_shift):
    """returns the image that has been shifted left and then subtracted from itself.
    The image should be converted to grayscale first
    This was basically done in assignment 1

    Input: an image

    Output: numpy.array:  the result of shifting and subtracting the image
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    left = np.pad(gray, left_shift, mode='edge')
    left = left[left_shift:left_shift+image.shape[0], -image.shape[1]:]
    subtracted = gray.astype(float) - left.astype(float)
    min = np.amin(subtracted)
    return (subtracted - min) / (np.amax(subtracted) - min)

def sobel_image(image, hor=0, vert=1):  # vert=0 for horizontal edges, hor=0 for vertical, both=1 for diagonal
    """returns the image that has had a Sobel filter applied to it.  Look here:
    https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html?#sobel
    for more information about how to use a sobel filter in opencv
    You should make sure to convert the image to grayscale first, and you may want to blur it as well
    You should also mess around with the different arguments to see what the effects are

    Input: an image

    Output:  numpy.array:  an image
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(float)
    #blurred = blur_image_gaussian(gray, (7, 7))
    return cv2.Sobel(gray, -1, vert, hor) / 255

def canny_image(image):
    """use the canny edge operator to highlight the edges of an image.  Look here:
    https://docs.opencv.org/3.1.0/da/d22/tutorial_py_canny.html
    for some information about how to use the canny edge detector
    You should make sure to convert the image to grayscale first, and you may want to blur it as well

    Input: an image

    Output:  numpy.array:  an edge image
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = blur_image_gaussian(gray)
    return cv2.Canny(blurred, 50, 100)
    #raise NotImplementedError

def custom_line_detector(image, slope_step=1):
    """create your own Hough Line accumulator that will tell you all of the lines on a given image
    to start you will want to setup the image by using Canny to create an edge image and maybe blur as well (notice a pattern??).
    Then you will  need to look at all the edges and have them "vote" for lines that they belong to.  Choose the most
    relevant lines and return them.  You can look here:
    https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_houghlines/py_houghlines.html
    for more ideas

    Input:  an image

    Output:  Vector of lines in the form (rho, theta)
    """
    canny = canny_image(image)
    # canny = blur_image_gaussian(canny, (3, 3))  # Can blur to reduce noise
    # This is all possible lines: Rows are theta values, columns are possible r values
    lines = np.zeros((180, round(2 * (canny.shape[0]**2 + canny.shape[1]**2)**(.5))))
    edges = np.nonzero(canny)
    for pixel in range(len(edges[0])):  # Iterates through all edge pixels
        for theta in range(0, 180, slope_step):  # High slope step reduces run time
            rad = radians(theta)  # Math trig functions take radians, not degrees
            d = int(edges[1][pixel]*cos(rad) + edges[0][pixel]*sin(rad)) # d = xcos(theta) + ysin(theta)
            lines[theta, d] += 1
    max = np.amax(lines)
    where = np.where(lines >= .8*max)
    print(np.array([[where[1][i], where[0][i]] for i in range(len(where))]).shape)
    return [[where[1][i], where[0][i]] for i in range(len(where))]
    #return np.where(lines >= .8*max)


def draw_lines_on_image(lines, image):
    """draws the given lines on the image.  Note that the input lines are the same values you
    return in custom_line_detector.  See how to draw lines here:
    https://docs.opencv.org/2.4/modules/core/doc/drawing_functions.html#line

    Input:  an image and a list of lines

    Output:  numpy.array:  an image with lines drawn on it
    """
    print('lines shape:', np.array(lines).shape)
    # for i in range(len(lines[0])):
    #     print('\nr', lines[1][i], '\ttheta', lines[0][1], '\n')
    #     r = lines[1][i]
    #     theta = radians(lines[0][i])
    print(lines[0])
    for theta, r in lines:
        theta = radians(theta)
        slope = -cos(theta)/sin(theta)
        #slope = cos(theta)/sin(theta)
        # point1 = (int(r*sin(theta) + 1000*slope), int(r*cos(theta) + 1000))
        # point2 = (int(r*sin(theta) - 1000*slope), int(r*cos(theta) - 1000))
        point1 = (int(r*cos(theta) + 1000), int(r*sin(theta) + 1000*slope))
        point2 = (int(r*cos(theta) - 1000), int(r*sin(theta) - 1000*slope))
        # point1 = ( int(r*cos(theta) - 1000), int(r*sin(theta) - 1000 * slope) )
        # point2 = ( int(r*cos(theta) + 1000), int(r*sin(theta) + 1000 * slope) )
        image = cv2.line(image, point1, point2, (255, 0, 0), 2)
    return image

def hough_line_detector(image):
    """now you will use the Hough line detector that is available in open cv to redo
    what you did in the custom_line_detector method.  See here:
    https://docs.opencv.org/2.4/modules/imgproc/doc/feature_detection.html?#houghlines

    Input:  an image

    Output:  Vector of lines in the form (rho, theta)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    res = cv2.HoughLines(gray, 3, radians(5), 150)
    #print('\norig res', res.shape)
    res = [i[0] for i in res]
    # print(np.array(res).shape, '\n')
    return res

def hough_circle_detector(image):
    """now use the Hough Circle detector to find circles in a given image.  See here:
    https://docs.opencv.org/2.4/modules/imgproc/doc/feature_detection.html?#houghcircles

    input:  an image to find circles in

    output:  Vector of circles in the form (x, y, radius)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 2, 5)
    return circles[0]
    #raise NotImplementedError

def draw_circles_on_image(circles, image):
    """draws the given circles on the image.  Note that the input circles are the same values you
    return in hough_circle_detector.  See here:
    https://docs.opencv.org/2.4/modules/core/doc/drawing_functions.html#circle

    Input:  an image and a list of circles

    Output:  numpy.array:  an image with circles drawn on it
    """
    for circle in circles:
        image = cv2.circle(image, (circle[0], circle[1]), circle[2], (255,0,0), 5)
    return image
