# Generating Arbitrary Aruco Markers
import sys
import cv2 as cv
import argparse
import numpy as np
import os

# User Arguments
parser = argparse.ArgumentParser(
    description="Generating Arucomarkers"
)
parser.add_argument(
    "--output", help="Path to output where the aruco marker should be saved"
)
parser.add_argument(
    "--id", required=True, type=int, help="ID of Aruco tag to generate"
)
parser.add_argument(
    "--type", type=str, help="Type of Aruco tag to generate", default="DICT_ARUCO_ORIGINAL"
)
args = parser.parse_args()
args = vars(args)

ARUCO_DICT = {
    "DICT_4X4_50": cv.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv.aruco.DICT_ARUCO_ORIGINAL,
    "DICT_APRILTAG_16h5": cv.aruco.DICT_APRILTAG_16h5,
    "DICT_APRILTAG_25h9": cv.aruco.DICT_APRILTAG_25h9,
    "DICT_APRILTAG_36h10": cv.aruco.DICT_APRILTAG_36h10,
    "DICT_APRILTAG_36h11": cv.aruco.DICT_APRILTAG_36h11,
    "DICT_ARUCO_MIP_36h12": cv.aruco.DICT_ARUCO_MIP_36h12
}

if ARUCO_DICT.get(args["type"], None) is None:
    end_msg = "This type of Aruco tag is not supported"
    sys.exit(end_msg)

arucoDict = cv.aruco.getPredefinedDictionary(ARUCO_DICT[args["type"]])
tag = np.zeros((300, 300, 1), dtype="uint8")
cv.aruco.generateImageMarker(arucoDict, args["id"], 300, tag, 1)
cv.imwrite(os.path.join(os.getcwd(), 'img.png'), tag)
cv.imshow("ArUCo Tag", tag)

cv.waitKey(0)
cv.destroyWindow()
