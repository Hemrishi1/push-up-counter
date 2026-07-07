import cv2
import mediapipe as mp
import math
import numpy as np
import time

class poseDetector():
    def __init__(self, mode=False, smooth=True, detectionCon=0.8, trackingCon=0.8):
        self.mode = mode
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackingCon = trackingCon
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(
            static_image_mode=self.mode,
            smooth_landmarks=self.smooth,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackingCon,
            model_complexity=2,
            enable_segmentation=False
        )
        self.mpDraw = mp.solutions.drawing_utils

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if draw and self.results.pose_landmarks:
            self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img
    
    def findPosition(self, img, draw=True):
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                self.lmList.append([id, cx, cy])
                if draw: cv2.circle(img, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
        return self.lmList
    
    def findAngle(self, img, p1, p2, p3, draw=True):
        x1, y1 = self.lmList[p1][1], self.lmList[p1][2]
        x2, y2 = self.lmList[p2][1], self.lmList[p2][2]
        x3, y3 = self.lmList[p3][1], self.lmList[p3][2]
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
        if angle > 180: angle = 360 - angle
        elif angle < 0: angle = -angle
        if draw:
            cv2.circle(img, (x1, y1), 10, (64, 127, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 10, (64, 127, 255), cv2.FILLED)
            cv2.circle(img, (x3, y3), 10, (64, 127, 255), cv2.FILLED)
            cv2.circle(img, (x1, y1), 15, (64, 127, 255))
            cv2.circle(img, (x2, y2), 15, (64, 127, 255))
            cv2.circle(img, (x3, y3), 15, (64, 127, 255))
            cv2.line(img, (x1, y1), (x2, y2), (255, 127, 64), 3)
            cv2.line(img, (x2, y2), (x3, y3), (255, 127, 64), 3)
        return angle
