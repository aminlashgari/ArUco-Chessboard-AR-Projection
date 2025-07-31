# Import necessary libraries
import cv2 as cv
import numpy as np
import sys
import argparse
import os


def main(cap, source, out) -> None:

    # Paramaters:
    ARUCO_DICT = {
        "DICT_5X5_50": cv.aruco.DICT_5X5_50,
        "DICT_5X5_100": cv.aruco.DICT_5X5_100,
        "DICT_5X5_250": cv.aruco.DICT_5X5_250,
        "DICT_6X6_50": cv.aruco.DICT_6X6_50,
        "DICT_6X6_100": cv.aruco.DICT_6X6_100,
        "DICT_6X6_250": cv.aruco.DICT_6X6_250,
        "DICT_7X7_50": cv.aruco.DICT_7X7_50,
        "DICT_7X7_100": cv.aruco.DICT_7X7_100,
        "DICT_7X7_250": cv.aruco.DICT_7X7_250
    }
    all_pts = {}
    aruco_pts_pre = np.empty((4, 2), dtype=np.float32)
    isAruco = False
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.01)
    size_src = source.shape
    lk_params = dict(winSize=(31, 31),
                     maxLevel=4,
                     criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
    pts_src = np.array([
        [0, 0],
        [size_src[1] - 1, 0],
        [size_src[1] - 1, size_src[0] - 1],
        [0, size_src[0] - 1]
    ], dtype=float)

    ret, frame_old = cap.read()

    if ret:
        frame_old = cv.resize(frame_old, None, fx=1/2,
                              fy=1/2, interpolation=cv.INTER_AREA)
        frame_old_gray = cv.cvtColor(frame_old, cv.COLOR_BGR2GRAY)
        st, pts_pre = cv.findChessboardCorners(
            frame_old_gray, (11, 7), None)

        if st and len(pts_pre) == 77:
            pts_pre_refined = cv.cornerSubPix(
                frame_old_gray, pts_pre, (11, 11), (-1, -1), criteria)

        else:
            # ArucoMarker Detection:
            def aruco_detection() -> tuple:

                aruco_len = list()
                ArucoParams = cv.aruco.DetectorParameters()

                for ArucoName, ArucoDict in ARUCO_DICT.items():
                    ArucoDict = cv.aruco.getPredefinedDictionary(
                        ArucoDict)
                    (marker_corner, marker_id, _) = cv.aruco.detectMarkers(
                        frame_old_gray, ArucoDict, parameters=ArucoParams)

                    if marker_id is not None and len(marker_id) > 0:
                        marker_corners = list()
                        marker_id = np.array(marker_id).reshape(-1)
                        unique_ids, cnt = np.unique(
                            marker_id, return_counts=True)
                        repeated_ids = unique_ids[cnt > 1]

                        if repeated_ids.size > 0:
                            msg = f"Num of Points: {len(marker_id)}\nFound identical aruco markers '(identical ids: {ArucoName}: {cnt[-1]}x{repeated_ids})'"
                            sys.exit(msg)

                        marker_id.tolist()
                        aruco_size = int(
                            (ArucoName.split("_")[1]).split("X")[0])

                        for corner in marker_corner:
                            marker_corner = np.array(corner)
                            marker_corner = np.squeeze(marker_corner, axis=0)
                            marker_corners.append(marker_corner)

                        if aruco_size not in all_pts:
                            all_pts[aruco_size] = {}

                        for id in range(len(marker_id)):
                            key = marker_id[id]
                            value = marker_corners[id]

                            if key not in aruco_len:
                                aruco_len.append(key)

                            if key not in all_pts[aruco_size]:
                                all_pts[aruco_size][key] = value

                return (all_pts, len(aruco_len))

            def aruco_processing():

                nonlocal aruco_pts_pre
                four_corners = list()
                all_pts, aruco_len = aruco_detection()

                if aruco_len == 4:
                    isAruco = True

                    for marker_size, marker_dict in all_pts.items():
                        for marker_id, marker_corner in marker_dict.items():
                            four_corners.append(marker_corner)

                    valid_arrs = [four_corners[i]
                                  for i in range(aruco_len)]
                    valid_arrs = np.array(valid_arrs)

                    for i in range(aruco_len):
                        width_pts = (
                            (valid_arrs[i][0, 0] + valid_arrs[i][2, 0]) / 2)
                        height_pts = (
                            (valid_arrs[i][0, 1] + valid_arrs[i][2, 1]) / 2)
                        aruco_pts_pre[i] = [width_pts, height_pts]

                    aruco_array = aruco_pts_pre.copy()
                    minimum = np.min(aruco_array[:, 0])
                    min_idx = np.where(aruco_array[:, 0] == minimum)[0]
                    tmp_array = np.delete(aruco_array, min_idx, axis=0)
                    min_width_2 = np.min(tmp_array[:, 0])
                    min_idx_2 = np.where(aruco_array[:, 0] == min_width_2)[0]

                    first_arr = np.squeeze(aruco_array[min_idx], axis=None)
                    fourth_arr = np.squeeze(aruco_array[min_idx_2], axis=None)

                    if first_arr[1] < fourth_arr[1]:
                        first_pts = first_arr
                        fourth_pts = fourth_arr
                    else:
                        first_pts = fourth_arr
                        fourth_pts = first_arr

                    tmp_array = np.delete(
                        aruco_array, [min_idx, min_idx_2], axis=0)
                    second_arr = tmp_array[0]
                    third_arr = tmp_array[1]

                    if second_arr[1] < third_arr[1]:
                        second_pts = second_arr
                        third_pts = third_arr
                    else:
                        second_pts = third_arr
                        third_pts = second_arr

                    aruco_pts_pre = np.array([first_pts,
                                              second_pts,
                                              third_pts,
                                              fourth_pts])

                    return aruco_pts_pre, isAruco

                msg = f"Found more/less than 4 points: {aruco_len}"
                sys.exit(msg)

            aruco_pts_pre, isAruco = aruco_processing()

    if out:
        # Output video parameters
        fourcc = cv.VideoWriter_fourcc(*'XVID')
        RESOLUTION = (640, 480)
        out_path = os.path.join(os.getcwd(), 'out', 'output.avi')
        out_save = cv.VideoWriter(out_path, fourcc, 20.0, RESOLUTION)
        if not out_save.isOpened():
            exit()

    while ret:

        key = cv.waitKey(25) & 0xFF
        isTrue, frame = cap.read()

        if isTrue:
            frame = cv.resize(frame, (frame_old.shape[1], frame_old.shape[0]))
            frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            if not isAruco:
                new_pts, st, err = cv.calcOpticalFlowPyrLK(
                    frame_old_gray, frame_gray, pts_pre_refined, None, **lk_params)

                if not np.all(st == 1) or len(new_pts) != 77:
                    sys.exit(1)
                good_new = np.array([new_pts[0], new_pts[10], new_pts[76],
                                     new_pts[66]], dtype=float)
            else:
                # aruco markers
                new_pts, st, err = cv.calcOpticalFlowPyrLK(
                    frame_old_gray, frame_gray, aruco_pts_pre, None, **lk_params)
                good_new = new_pts

            # Homography
            H, status = cv.findHomography(pts_src, good_new, cv.RANSAC, 5)

            # Mapping an image into the plane
            img_out = cv.warpPerspective(
                source, H, (frame.shape[1], frame.shape[0]))
            cv.fillConvexPoly(frame, good_new.astype(int), 0, 16)
            frame = frame + img_out

            # Updating the frames
            frame_old_gray = frame_gray.copy()

            if not isAruco:
                pts_pre_refined = new_pts.reshape(-1, 1, 2)
                cv.putText(frame, f"Chessboard AR Projection", (5, 20),
                           cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1, cv.LINE_AA)

            else:
                aruco_pts_pre = new_pts.reshape(-1, 1, 2)
                cv.putText(frame, f"Aruco AR Projection", (5, 20),
                           cv.FONT_HERSHEY_SIMPLEX, 0.7, (150, 255, 0), 1, cv.LINE_AA)
                cv.putText(frame, f"Number of Points:4", (5, 45),
                           cv.FONT_HERSHEY_SIMPLEX, 0.7, (150, 255, 0), 1, cv.LINE_AA)

            # Displaying each frame
            cv.imshow("Final Image", frame)

            # Writing frames into a file
            if out:
                frame_modified = cv.resize(frame, RESOLUTION)
                out_save.write(frame_modified)

            # Key operations
            # Quit:
            if key == ord('q'):
                break
            # Pause:
            if key == ord("p"):
                cv.waitKey()

        else:
            print("Feed is not available or finished")
            break


if __name__ == "__main__":

    # Parsing User Arguments
    parser = argparse.ArgumentParser(
        description="Initialize the Program by determining the inputs")
    parser.add_argument(
        "--img_src", help="Path to source image", required=True)
    parser.add_argument(
        "--feed", help="Path to a video file", required=True)
    parser.add_argument(
        "--save", help="Optional argument to save the output as a video file", default=False
    )
    args = parser.parse_args()
    img_src, feed_path, out = args.img_src, args.feed, args.save

    try:
        source = cv.imread(img_src, cv.IMREAD_UNCHANGED)
        cap = cv.VideoCapture(feed_path)

    except Exception as err:
        err_msg = f"Raised Error: {err}"
        sys.exit(err_msg)

    else:
        main(cap, source, out)
        cap.release()
        cv.destroyAllWindows()
