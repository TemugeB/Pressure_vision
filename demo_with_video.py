import numpy as np 
import cv2 as cv
import sys
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import pickle
import open3d as o3d
from scipy.linalg import pinv

#load camera preset values from file
def get_settings(filename = 'preset_settings.dat'):

    settings = {}
    with open(filename) as f:
        while True:
            line = f.readline()
            if line == '': break

            if line[0] == '#': continue 
            line = line.split(',')
            
            settings[line[0]] = [int(l) for l in line[1:]]
    
    return settings


class VisMesh():

    def __init__(self):

        #define an initial mesh
        xp, yp = 30, 20
        self.yv, self.xv = np.meshgrid(np.linspace(0, 2.5, xp), np.linspace(0, 2., yp))
        self.mesh_p = np.stack([self.xv, self.yv], -1).reshape((-1, 2))
        self.mesh_p = np.concatenate([self.mesh_p, np.zeros((self.mesh_p.shape[0], 1))], axis = -1) #add zero coords
        self.simpl = [] #mesh vertex indices. Is there a better way to do this?
        for j in range(yp - 1):
            for k in range(xp - 1):
                self.simpl.append([xp * j + k, xp * (j + 1) + k, xp * (j + 1) + k + 1])
                self.simpl.append([xp *j + k, xp * (j+1) + k +1, xp * j + k + 1])

        #read the projection matrxi
        self.P, R, T = self.get_projection_matrix()
        self.P_inv = pinv(self.P)
        self.origin_to_cam = -R.T @ T.reshape((3, 1)) #position of camera from the defined plane coordinates


        #create visualizer
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(visible=True, width = 720, height = 1280, left = 0, top = 0)     

        #add the mesh
        self.mesh = o3d.geometry.TriangleMesh()
        self.mesh.vertices = o3d.utility.Vector3dVector(self.mesh_p)
        self.mesh.triangles = o3d.utility.Vector3iVector(self.simpl)
        self.vis.add_geometry(self.mesh)

        #self.vis.run()

    def get_projection_matrix(self):

        #load camera instrinsic parameters from file
        def read_camera_parameters(filepath = 'camera_parameters/intrinsic.dat'):

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

        def _make_homogeneous_rep_matrix(R, t):
            P = np.zeros((4,4))
            P[:3,:3] = R
            P[:3, 3] = t.reshape(3)
            P[3,3] = 1
            return P

        cmtx, dist = read_camera_parameters()
        R = np.loadtxt('camera_parameters/R.dat'); R, _  = cv.Rodrigues(R)
        T = np.loadtxt('camera_parameters/T.dat')

        P = cmtx @ _make_homogeneous_rep_matrix(R, T)[:3,:]

        return P, R, T

    def get_3d_point(self, Pinv, origin_to_cam, pixel, z = 0.05):
        #this function calculates the 3d point from the projection ray at certain z value.
        #As default, we assume z = 0, which is the plane defined by the calibration pattern

        p1 = Pinv @ np.array([pixel[0], pixel[1], 1]).reshape((3,1)) #this returns a point in homogeneous coordinates
        p1 = p1/p1[-1] # divide by homogeneous coorindate to get cartesian coorindates

        ray = origin_to_cam - p1[:3] #t * ray is the ray of points that map to the same pixel coorindates at different values of t.

        #need to solve for t: p3 = p1 + t*ray
        t = (z - p1[2])/ray[2]
        p3 = p1[:3] + t * ray
        return p3

    def update_mesh(self, press_point, deltas):

        p = self.get_3d_point(self.P_inv, self.origin_to_cam, np.array(press_point), z = 0.5)

        zs = p[2] * np.exp((- (self.mesh_p[:,0] - p[0])**2 -(self.mesh_p[:,1] - p[1])**2 )/.02)
        self.mesh_p[:, 2] = zs
        
        self.mesh.vertices = o3d.utility.Vector3dVector(self.mesh_p)
        self.vis.update_geometry(self.mesh)

    def cycle(self):
        self.vis.poll_events()
        self.vis.update_renderer()
        

class PressureSensor:

    def __init__(self, vid_path):

        #user settings
        self.roi = ((650,300), (1350, 800))
        self.roi_width = self.roi[1][0] - self.roi[0][0]
        self.roi_height = self.roi[1][1] - self.roi[0][1]
        self.max_cont_size = 500
        self.min_cont_size = 5
        self.tracking_color = (0,0,0)
        self.threshold = 135

        #camera presets
        self.preset_settings = get_settings()

        #open capture device
        self.vidpath = vid_path
        self.cap = cv.VideoCapture(vid_path)
        # self.cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        # self.cap.set(3, 1920) #set resolution
        # self.cap.set(4, 1080)
        #self.apply_settings('preset') #This function is used to freeze the camera stream settings. But not neccsary for recorded video

        #tracking flag
        self.tracking = False
        self.start_markers = None
        self._tracking_markers = None
        self._alpha = 0.85 #for exponential smoothing

        #for training
        self.pressed_points = []
        self.deltas = []
        self.press_point = None

        #for predicting pressed point
        self.pp = None
        with open('model.pkl', 'rb') as f:
            self.model = pickle.load(f)

        #for mesh visualization
        self.mesher = VisMesh()

    def detect_markers(self, frame):
        
        #get the roi
        roi_im = frame[self.roi[0][1]:self.roi[1][1], self.roi[0][0]:self.roi[1][0]]
        
        #simplest of color detection
        mask = np.sqrt(np.sum(np.square(roi_im - self.tracking_color), axis = -1)) < self.threshold
        mask = mask.astype(np.uint8) #boolean -> uint8

        #find the contours that could be markers
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        #shortlist contours by area
        cont_areas = [cv.contourArea(cont) for cont in contours]
        contours = [contours[i] for i in range(len(contours)) if self.min_cont_size < cont_areas[i] < self.max_cont_size]

        if len(contours) == 0:
            return [], []

        #determine the center of the contours
        cont_centers = np.array([np.mean(cont.reshape(-1, 2), axis = 0) for cont in contours])

        return cont_centers + self.roi[0], contours #back to full image coorindates

    #poor man's point correspondence finder
    def track_point_correspondences(self, markers):
        shifted_markers = (np.mean(markers, axis = 0) - np.mean(self.start_markers, axis = 0)) + self.start_markers

        dist_mat = cdist(shifted_markers, markers)
        _ms = self.start_markers.copy()
        used_indices = []
        orig_indices = []
        for i in range(len(_ms)):
            _m = np.argsort(dist_mat[i])
            for j in _m:
                if j not in used_indices:
                    orig_indices.append(i)
                    used_indices.append(j) #this can cause issues
                    break
        
        _ms[orig_indices] = markers[used_indices]
        return _ms

    #we can automatically change the tracking marker color for better frame to frame tracking
    def update_tracking_color(self, frame, contours):

        roi_im = frame[self.roi[0][1]:self.roi[1][1], self.roi[0][0]:self.roi[1][0]]
        mask = np.zeros(roi_im.shape[:2], dtype= np.float32)
        cv.drawContours(mask, contours, -1, 1, -1)
        
        #mask using the detected contours
        masked = roi_im.reshape((-1, 3)) * mask.reshape((-1, 1))
        self.tracking_color = np.sum(masked, axis = 0)/np.sum(mask).astype(np.int32)
        
        #self.tracking_color = np.median(masked[masked > 1].reshape((-1, 3)), axis = 0) #median of non-zero pixels
        #reduce the tracking theshold
        self.threshold = np.clip(self.threshold - 105, 0, 255)


    def setup_tracking(self, frame):
        
        #dont update more than once
        if self.tracking: return
        self.tracking = True

        markers, contours = self.detect_markers(frame)
        self.update_tracking_color(frame, contours)
        markers,contours = self.detect_markers(frame) # get the markers again with new tracking color
        self.start_markers = markers
        self._tracking_markers = markers
    
    def stream(self,):

        while True:
            ret, frame = self.cap.read()
            if not ret: 
                print('No more frames. :-( ')
                break
            frame_orig = frame.copy()
            #frame_empty = np.zeros(frame.shape, dtype=np.uint8)
            
            #draw detected markers
            markers, _ = self.detect_markers(frame)
            if self.start_markers is None:
                [cv.circle(frame, (int(c[0]), int(c[1])), 3, (0,255,0), 1) for c in markers]
            else: #tracking markers set. So draw deltas
                markers = self.track_point_correspondences(markers)
                self._tracking_markers = self._alpha * self._tracking_markers + (1 - self._alpha) * markers

                #draw markers that indicate pressure
                for m1, m2 in zip(self.start_markers, self._tracking_markers):
                    r = 4* np.sqrt(np.sum(np.square(m1 - m2))).astype(np.int32)
                    cv.line(frame, (int(m1[0]), int(m1[1])), (int(m2[0]), int(m2[1])), (0,0,255), 2)
                    cv.circle(frame, (int(m2[0]), int(m2[1])), r, (0,0,255), 2)

                #detect pressed point
                pp = self.get_pressed_point()
                if len(pp) != 0:
                    self.pp = self._alpha * self.pp + (1 - self._alpha) * pp
                    self.mesher.update_mesh(self.pp, None)

            if self.press_point is not None:
                cv.circle(frame, tuple(self.press_point), 5, (0,255,0), -1)
            
            #draw roi
            cv.rectangle(frame, self.roi[0], self.roi[1], (0,0,255), 1)

            #draw frame
            cv.imshow('frame', frame)

            k = cv.waitKey(10)
            if k == 27 or k == ord('q'):
                break
            
            if not self.tracking:
                self.setup_tracking(frame_orig)

            #update meshed
            self.mesher.cycle()


    def get_pressed_point(self):
        if self._tracking_markers is None: return []

        deltas = self.start_markers - self._tracking_markers
        total_delta = np.sum(np.abs(deltas))

        if total_delta < 100: return []

        try:
            y = self.model.predict([deltas.reshape((-1))])[0]
        except Exception as e:
            print('Unable to predict: ', str(e))
            return []


        #convert to frame scale
        y = np.array([y[0] * self.roi_width, y[1] * self.roi_height]) + self.roi[0]
        y = y.astype(np.int32)
        #set tracking point 
        if self.pp is None:
            self.pp = y

        return y

    #applies preset file settings to camera
    def apply_settings(self, setting_type = 'default'):

        if setting_type == 'preset':
            _t = 2
        else:
            _t = 1 #default values

        for key, value in self.preset_settings.items():
            if value[1] == -1 and _t == 1: continue
            self.cap.set(value[0], value[_t])


def main():

    if len(sys.argv) != 2:
        print('Call the function with videopath. Exiting')
        quit()

    vidpath = sys.argv[1]
    ps = PressureSensor(vidpath)
    ps.stream()

if __name__ == '__main__':
    main()