import cv2
import dlib
import numpy as np
from imutils import face_utils


class FaceDetector():
    
    def __init__(self):
        face_landmark_path = './shape_predictor_68_face_landmarks.dat'
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(face_landmark_path)

    def draw_ROI(self, eye_landmarks, frame):
        x1, y1 = np.min(eye_landmarks, axis=0)
        x2, y2 = np.max(eye_landmarks, axis=0)
        x1, y1 = x1-9, y1-9
        x2, y2 = x2+9, y2+9
        x1,y1,x2,y2 = self.__cramp(x1,y1,x2,y2,frame)
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 1)
        return [x1,y1,x2,y2]

    def __cramp(self, x1, y1, x2, y2, frame):
        if x1 < 0:
            x1 = 0
        if y1 < 0:
            y1 = 0
        if x2 > frame.shape[0]:
            x2 = frame.shape[0]-1
        if y2 > frame.shape[1]:
            y2 = frame.shape[1]-1
        return x1, y1, x2, y2

    def detect_features(self, frame):
        face_rects = self.detector(frame, 0)
        if len(face_rects) > 0:
            shape = self.predictor(frame, face_rects[0])
            shape = face_utils.shape_to_np(shape)
            l,r = self.__detect_eyes(shape, frame)
            f = self.__detect_facial_points(shape, frame)
            return f, l, r
        return [],[],[]


    def __detect_eyes(self, shape, frame):
        left_eye  = shape[36:42]
        right_eye = shape[42:48]
        lbox = self.draw_ROI(left_eye, frame)
        rbox = self.draw_ROI(right_eye, frame)
        return lbox,rbox


    def __detect_facial_points(self, shape, frame):
        p_ends = [[14,2], [27,8]]
        p_med = 30
        dists = self.__get_dists(p_ends, p_med, shape)
        face_points = np.vstack((shape[0:36], shape[48:]))
        for (x,y) in face_points:
            cv2.circle(frame, (x,y), 2, (255,0,0), -1)
        return dists

    def __get_dists(self, p_ends, p_med, shape):
        dists = []
        for (p1,p2) in p_ends:
            d1 = np.linalg.norm(shape[p_med]-shape[p1])
            d2 = np.linalg.norm(shape[p_med]-shape[p2])
            dt = d1 + d2
            dists.append(d1/dt)
        return dists



