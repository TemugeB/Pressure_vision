import cv2 as cv
import numpy as np

ARUCO_PARAMETERS = cv.aruco.DetectorParameters()
ARUCO_DICT = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_ARUCO_ORIGINAL)

#load camera instrinsic parameters from file
def read_camera_parameters(filepath = 'intrinsic.dat'):

    inf = open(filepath, 'r')

    cmtx = []
    dist = []

    #ignore first line
    line = inf.readline()
    for _ in range(3):
        line = inf.readline().split()
        line = [float(en) for en in line]
        cmtx.append(line)

    #ignore line that says "distortion"
    line = inf.readline()
    line = inf.readline().split()
    line = [float(en) for en in line]
    dist.append(line)

    #cmtx = camera matrix, dist = distortion parameters
    return np.array(cmtx), np.array(dist)


def get_RT(points, cmtx, dist):

    #Selected coordinate points for each corner of QR code.
    qr_edges = np.array([[0,0,0],
                        [0,1,0],
                        [1,1,0],
                        [1,0,0]], dtype = 'float32').reshape((4,1,3))

    #determine the orientation of QR code coordinate system with respect to camera coorindate system.
    ret, rvec, tvec = cv.solvePnP(qr_edges, points, cmtx, dist)

    #Define unit xyz axes. These are then projected to camera view using the rotation matrix and translation vector.
    unitv_points = np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1]], dtype = 'float32').reshape((4,1,3))
    if ret:
        points, jac = cv.projectPoints(unitv_points, rvec, tvec, cmtx, dist)
        return points, rvec, tvec

    #return empty arrays if rotation and translation values not found
    else: return [], [], []  

def draw_axes(frame, points):
    colors = ((0,0,255), (0,255,0), (255,0,0))
    origin = points[0]
    for axis, col in zip(points[1:], colors):
        cv.line(frame, tuple(origin), tuple(axis), col, 2)

    return frame

def detect_marker(frame, cmtx, dist):

        #detect the markers
        corners, ids, rejectedImgPoints = cv.aruco.detectMarkers(frame, ARUCO_DICT, parameters=ARUCO_PARAMETERS)

        points, rvec, tvec = [], [], []

        if ids is not None:
            frame = cv.aruco.drawDetectedMarkers(frame, corners, borderColor = (255,0,0))

            #get rvec and tvec for the marker
            points, rvec, tvec = get_RT(corners[0], cmtx, dist)

            if len(points) != 0:
                points = np.array(points).reshape((4,2)).astype(np.int32)
                frame = draw_axes(frame, points)

        return frame, rvec, tvec

def get_calibration(calibframe_path, intrinsics_path):

    cmtx, dist = read_camera_parameters(intrinsics_path)
    frame = cv.imread(calibframe_path, 1)
    print(frame.shape)

    frame, rvec, tvec = detect_marker(frame, cmtx, dist)
    cv.imshow('frame', frame)
    cv.waitKey(0)

    np.savetxt('R.dat', rvec)
    np.savetxt('T.dat', tvec)

def main():

    calibframe_path = 'calibframe.png'
    intrinsics_path = 'intrinsic.dat'
    get_calibration(calibframe_path, intrinsics_path)

main()