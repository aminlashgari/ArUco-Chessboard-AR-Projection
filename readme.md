# Augmenting an arbitrary image on ArUco Markers and Chessboard Pattern
In this project, an image is warped onto a video using homography, based on either four ArUco markers or a chessboard pattern.
Using the **[Lucas-Kanade](https://en.wikipedia.org/wiki/Lucas%E2%80%93Kanade_method)** method, key points within the chessboard and ArUco markers
can be tracked across consecutive frames of the video. Finally, with the help of homography, the image is seamlessly augmented into the video feed.

## How to run
1. Clone the repo
```sh
git clone https://github.com/aminlashgari/ArUco-Chessboard-AR-Projection.git
```
2. Check out the `requirements.txt` file and install the dependencies
```sh
pip install -r requirements.txt
```
3. Run the `main_AR.py` file

Note: The `marker_generation.py` file can be used to generate ArUco markers. Alternatively, there are several online tools available for
generating ArUco markers with arbitrary IDs and sizes, such as [Oleg Kalachev](https://chev.me/arucogen/) Website.

## Output Videos (Samples)
There are two examples: the first shows the output of a video with a chessboard pattern, while the second features non-identical ArUco markers.
Near the end of the second video, the ArUco markers are being displaced; however, the key points are still successfully tracked,
and the warped image remains stable and undistorted.
### Chessboard AR Projection


https://github.com/user-attachments/assets/935517d7-0362-4e05-9706-2ab99750bceb


### ArUco AR Projection


https://github.com/user-attachments/assets/a07d4808-695d-4829-bbd0-86cb3f3cbae1

## Acknowledgment
1. The [video](http://webpages.iust.ac.ir/mehralian/files/courses/vision95/1483540623.avi) with the chessboard pattern was taken from IUST Computer Vision course.
2. Some parts within the project were developed with help from [PyImageSearch](https://pyimagesearch.com/category/aruco-markers/) blogs on ArUco markers.
