"""
Tests for problem set 2
"""

import cv2
import numpy as np

# from Assignment02 import *
from ReAssignment02 import *


def part_1():
    # 1a
    img1 = cv2.imread("inputs/img1.jpg")

    cv2.imshow("Original Image", img1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    gb = blur_image_gaussian(img1)

    cv2.imshow("Gaussian Blurred Image", gb)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def part_2():
    # 2a
    img = cv2.imread("inputs/img2.jpg")

    cv2.imshow("Original Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #difference of image and shifted image
    shifted_img = shifted_difference(img, 50)
    cv2.imshow("Shifted Image", shifted_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #sobel filters
    sobel_img = sobel_image(img)
    cv2.imshow("Sobeled Image", sobel_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #canny
    canny_img = canny_image(img)
    cv2.imshow("Cannyed Image", canny_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def part_3():
    #using canny to get edges, create your own hough line accumulator
    img = cv2.imread("inputs/img3.jpg")

    cv2.imshow("Original Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    lines = custom_line_detector(img)
    lined_image = draw_lines_on_image(lines, img)

    cv2.imshow("Lined Image", lined_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def part_4():
    #use the opencv hough line accumulator to compare
    img = cv2.imread("inputs/img3.jpg")

    cv2.imshow("Original Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    lines = hough_line_detector(img)
    lined_image = draw_lines_on_image(lines, img)
    # lined_image = canny_image(img)

    cv2.imshow("Hough Lined Image", lined_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def part_5():
    #use opencv hough circle to try and detect some circles in an image
    img = cv2.imread("inputs/img4.jpg")

    cv2.imshow("Original Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    circles = hough_circle_detector(img)
    cirled_image = draw_circles_on_image(circles, img)

    cv2.imshow("Hough Circled Image", cirled_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    #comment out parts you don't want to run

    part_1()
    part_2()
    part_3()
    part_4()
    part_5()
