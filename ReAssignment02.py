"""
Resubmission for assignment 2
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
    My additional input: kernel size (increase to blur more)

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
    My additional inputs:
    hor and vert are orders of derivatives in x and y directions.
    vert=0 for horizontal edges, hor=0 for vertical, both=1 for diagonal

    Output:  numpy.array:  an image
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(float)
    #blurred = blur_image_gaussian(gray, (7, 7))  # Can blur to reduce noise
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

def custom_line_detector(image, slope_step=1, cutoff=.7):  # Increase slope step to reduce amount of lines checked
    """create your own Hough Line accumulator that will tell you all of the lines on a given image
    to start you will want to setup the image by using Canny to create an edge image and maybe blur as well (notice a pattern??).
    Then you will  need to look at all the edges and have them "vote" for lines that they belong to.  Choose the most
    relevant lines and return them.  You can look here:
    https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_houghlines/py_houghlines.html
    for more ideas

    Input:  an image
    My additional inputs:
    slope_step is the resolution of theta in degrees. Increase to speed up function but get less lines.
    cutoff is the percentage of max votes needed to choose a line. Increase to get less lines.

    Output:  Vector of lines in the form (rho, theta)
    """
    canny = canny_image(image)
    hypotenuse = round((canny.shape[0]**2 + canny.shape[1]**2)**(.5))  # Hypotenuse of image

    # This is all possible lines: Rows are rho values, columns are theta values
    lines = np.zeros((hypotenuse, 180))
    edges = np.nonzero(canny)
    edges = zip(edges[0], edges[1])
    for row, col in edges:
        for theta in range(0, 180, slope_step):  # High slope step reduces run time
            rad = radians(theta)  # Math trig functions take radians, not degrees
            d = int(col*cos(rad) + row*sin(rad))  # d = xcos(theta) + ysin(theta)
            try:
                lines[d, theta] += 1
            except Exception as e:  # Shouldn't happen but left in from testing, just in case
                print(e)
                continue

    max = np.amax(lines)
    where = np.where(lines >= cutoff*max)
    return zip(where[0], where[1])


def draw_lines_on_image(lines, image):
    """draws the given lines on the image.  Note that the input lines are the same values you
    return in custom_line_detector.  See how to draw lines here:
    https://docs.opencv.org/2.4/modules/core/doc/drawing_functions.html#line

    Input:  an image and a list of lines

    Output:  numpy.array:  an image with lines drawn on it
    """
    for rho, theta in lines:
        # This part checks to see if theta is in degrees, and if so, converts it
        if int(theta) == theta:
            theta = radians(theta)
        x = cos(theta) * rho
        y = sin(theta) * rho
        # Gets 2 points on the line, both 1000 away from the actual point, for the line
        point1 = (int(x + 1000*(-sin(theta))), int(y + 1000*(cos(theta))))
        point2 = (int(x - 1000*(-sin(theta))), int(y - 1000*(cos(theta))))
        cv2.line(image, point1, point2, (255, 0, 0), 2)  # Draws line on image
    return image

def hough_line_detector(image):
    """now you will use the Hough line detector that is available in open cv to redo
    what you did in the custom_line_detector method.  See here:
    https://docs.opencv.org/2.4/modules/imgproc/doc/feature_detection.html?#houghlines

    Input:  an image

    Output:  Vector of lines in the form (rho, theta)
    """
    canny = canny_image(image)
    res = cv2.HoughLines(canny, 1, np.pi/180, 200)
    # Reshapes array to get rid of extra dimension that's there for some reason
    res = np.reshape(res.flatten(), (res.shape[0], res.shape[2]))
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

def draw_circles_on_image(circles, image):
    """draws the given circles on the image.  Note that the input circles are the same values you
    return in hough_circle_detector.  See here:
    https://docs.opencv.org/2.4/modules/core/doc/drawing_functions.html#circle

    Input:  an image and a list of circles

    Output:  numpy.array:  an image with circles drawn on it
    """
    for circle in circles:
        cv2.circle(image, (circle[0], circle[1]), circle[2], (255,0,0), 5)
    return image
