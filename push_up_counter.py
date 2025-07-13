import cv2
import mediapipe as mp
import time
import math
import numpy as np
import pygame  

class poseDetector():
    
    def __init__(self, mode=False, smooth=True, detectionCon = 0.8, trackingCon=0.8):
        self.mode = mode
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackingCon = trackingCon
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(
            static_image_mode = self.mode,
            smooth_landmarks = self.smooth,
            min_detection_confidence = self.detectionCon,
            min_tracking_confidence = self.trackingCon,
            model_complexity = 2,  # Higher complexity for better accuracy
            enable_segmentation = False  # Disable segmentation for better performance
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
        if  self.results.pose_landmarks:
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



cap = cv2.VideoCapture(0)
# Set camera FPS to higher value
cap.set(cv2.CAP_PROP_FPS, 60)  # Reduced from 120 for stability
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Reduced resolution for better processing
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Initialize pygame mixer for sound effects
pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)

# Create simple beep sounds programmatically
def create_beep_sound(frequency, duration, sample_rate=22050):
    frames = int(duration * sample_rate)
    arr = np.zeros((frames, 2))
    
    for i in range(frames):
        wave = 4096 * np.sin(2 * np.pi * frequency * i / sample_rate)
        arr[i][0] = wave  # Left channel
        arr[i][1] = wave  # Right channel
    
    return pygame.sndarray.make_sound(arr.astype(np.int16))

# Create different sounds for feedback
try:
    rep_complete_sound = create_beep_sound(800, 0.2)  # Higher pitch for rep completion
    position_sound = create_beep_sound(400, 0.1)     # Lower pitch for position feedback
except:
    rep_complete_sound = None
    position_sound = None
    print("Sound initialization failed - continuing without audio")

pTime = 0
detector = poseDetector(detectionCon=0.9, trackingCon=0.9)  # Increased confidence thresholds
count, dir, bar, per = 0, 0, 0, 0
last_rep_count = 0  # Track last rep count for sound triggering

# Add smoothing variables for better accuracy
angle_history = []
angle_smoothing_window = 5  # Number of frames to average

while True:
    success, img = cap.read()
    img = cv2.resize(img, (1366, 780))
    # Apply flip consistently at the beginning for mirror effect
    img = cv2.flip(img, 1)
    img = detector.findPose(img, draw=False)
    lmList = detector.findPosition(img, draw=False)

    if (len(lmList)):
        # More robust position detection - check if person is in push-up position
        if (lmList[31][2] + 30 > lmList[29][2] and lmList[32][2] + 30 > lmList[30][2]):
            # Calculate left arm angle (shoulder-elbow-wrist)
            left_angle = detector.findAngle(img, 11, 13, 15)
            # Calculate right arm angle for comparison - NOW WITH VISUAL LINES
            right_angle = detector.findAngle(img, 12, 14, 16, draw=True)
            
            # Additional validation: Check if both hands are visible and in reasonable positions
            left_shoulder = lmList[11]
            right_shoulder = lmList[12]
            left_wrist = lmList[15]
            right_wrist = lmList[16]
            
            # Validate push-up position - hands should be roughly shoulder-width apart
            hand_distance = abs(left_wrist[1] - right_wrist[1])
            shoulder_distance = abs(left_shoulder[1] - right_shoulder[1])
            
            # Only count if hands are in proper push-up position
            if hand_distance > shoulder_distance * 0.5 and hand_distance < shoulder_distance * 2.5:
                # Use average of both arms for more accuracy
                angle = (left_angle + right_angle) / 2
                
                # Add angle smoothing
                angle_history.append(angle)
                if len(angle_history) > angle_smoothing_window:
                    angle_history.pop(0)
                smoothed_angle = sum(angle_history) / len(angle_history)
                
                # More precise percentage calculation
                per = np.interp(smoothed_angle, [70, 160], [100, 0])  # Adjusted angle range
                per = max(0, min(100, per))  # Clamp between 0-100
                bar = np.interp(per, (0, 100), (650, 100))
                
                # More strict thresholds for counting with stability check
                if per >= 90 and len(angle_history) >= angle_smoothing_window:  # Top position (arms extended)
                    if dir == 0:
                        count += 0.5
                        dir = 1
                        # Play sound for reaching top position
                        if position_sound and pygame.mixer.get_init():
                            try:
                                position_sound.play()
                            except:
                                pass
                elif per <= 15 and len(angle_history) >= angle_smoothing_window:  # Bottom position (arms bent)
                    if dir == 1:
                        count += 0.5
                        dir = 0
                        # Play sound for reaching bottom position
                        if position_sound and pygame.mixer.get_init():
                            try:
                                position_sound.play()
                            except:
                                pass
            else:
                # Invalid position - reset angle history to prevent false counts
                angle_history = []
                per = 50  # Neutral position
                bar = np.interp(per, (0, 100), (650, 100))

            # Display angle information for debugging (removed flip here)
            if 'smoothed_angle' in locals():
                cv2.putText(img, f'Angle: {int(smoothed_angle)}', (1000, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(img, f'L:{int(left_angle)} R:{int(right_angle)}', (1000, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Better visual feedback with color coding
            if (per >= 90 or per <= 15):
                cv2.putText(img, f'{int(per)}%', (1200, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            else:
                cv2.putText(img, f'{int(per)}%', (1200, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 3)  # Orange for mid-range
            
            # Dynamic color bar based on percentage
            # Calculate color: Red (0%) -> Yellow (50%) -> Green (100%)
            if per <= 50:
                # Red to Yellow transition (0% to 50%)
                red = 255
                green = int(255 * (per / 50))
                blue = 0
            else:
                # Yellow to Green transition (50% to 100%)
                red = int(255 * (1 - (per - 50) / 50))
                green = 255
                blue = 0
            
            bar_color = (blue, green, red)  # BGR format for OpenCV
            
            # Draw the progress bar outline
            cv2.rectangle(img, (1200, 100), (1275, 650), (255, 255, 255), 3)  # White outline
            # Draw the filled progress bar with dynamic color
            cv2.rectangle(img, (1200, int(bar)), (1275, 650), bar_color, cv2.FILLED)
    else:
        # Reset angle history when person is not detected
        angle_history = []
        # Removed flip here as it's now done at the beginning
        cv2.rectangle(img, (430, 740), (1335, 620), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, 'Take your position.', (440, 710), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5)
        
    # Changed background color from green to blue for count display
    cv2.rectangle(img, (12, 6), (425, 100), (255, 140, 0), cv2.FILLED)  # Orange background
    cv2.putText(img, f'count: {int(count)}', (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 5)  # White text
    
    # Check for completed reps and play completion sound
    if int(count) > last_rep_count:
        last_rep_count = int(count)
        if rep_complete_sound and pygame.mixer.get_init():
            try:
                rep_complete_sound.play()
            except:
                pass
            
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (20, 730), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3)
    
    cv2.imshow("Image", img)
    key = cv2.waitKey(1) & 0xFF  # Reduced delay for higher FPS
    if key == ord('q') or cv2.getWindowProperty("Image", cv2.WND_PROP_VISIBLE) < 1:
        break
cap.release()
cv2.destroyAllWindows()
# Clean up pygame mixer
if pygame.mixer.get_init():
    pygame.mixer.quit()
