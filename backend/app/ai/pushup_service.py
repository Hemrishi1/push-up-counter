import cv2
import numpy as np
import base64
import time
from fastapi import WebSocket
from typing import List, Dict
from app.ai.pushup_counter import poseDetector

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

# Global detector to avoid re-initializing
detector = poseDetector(detectionCon=0.9, trackingCon=0.9)

def process_frame(base64_img: str, session_state: Dict):
    """
    Decodes base64 image, processes it with pushup_counter logic,
    and returns the annotated image as base64 along with stats.
    """
    # Remove header if present
    if "base64," in base64_img:
        base64_img = base64_img.split("base64,")[1]
        
    # Decode image
    img_data = base64.b64decode(base64_img)
    np_arr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    
    if img is None:
        return {"error": "Invalid image"}
        
    img = cv2.resize(img, (1366, 780))
    # Flip for mirror effect
    img = cv2.flip(img, 1)
    
    img = detector.findPose(img, draw=False)
    lmList = detector.findPosition(img, draw=False)

    count = session_state.get("count", 0)
    dir = session_state.get("dir", 0)
    angle_history = session_state.get("angle_history", [])
    angle_smoothing_window = 5
    per = 0
    bar = 650
    rep_completed = False
    
    if len(lmList) > 0:
        if (lmList[31][2] + 30 > lmList[29][2] and lmList[32][2] + 30 > lmList[30][2]):
            left_angle = detector.findAngle(img, 11, 13, 15)
            right_angle = detector.findAngle(img, 12, 14, 16, draw=True)
            
            left_shoulder = lmList[11]
            right_shoulder = lmList[12]
            left_wrist = lmList[15]
            right_wrist = lmList[16]
            
            hand_distance = abs(left_wrist[1] - right_wrist[1])
            shoulder_distance = abs(left_shoulder[1] - right_shoulder[1])
            
            if hand_distance > shoulder_distance * 0.5 and hand_distance < shoulder_distance * 2.5:
                angle = (left_angle + right_angle) / 2
                angle_history.append(angle)
                if len(angle_history) > angle_smoothing_window:
                    angle_history.pop(0)
                smoothed_angle = sum(angle_history) / len(angle_history)
                
                per = np.interp(smoothed_angle, [70, 160], [100, 0])
                per = max(0, min(100, per))
                bar = np.interp(per, (0, 100), (650, 100))
                
                if per >= 90 and len(angle_history) >= angle_smoothing_window:
                    if dir == 0:
                        count += 0.5
                        dir = 1
                elif per <= 15 and len(angle_history) >= angle_smoothing_window:
                    if dir == 1:
                        count += 0.5
                        dir = 0
                        rep_completed = True # Rep completed at bottom or top, depending on preference
                        # For pushups, going down and then up is a rep. So top is completion. 
                        # But wait, original code: 0.5 for down, 0.5 for up.
                        if int(count) > session_state.get("last_rep_count", 0):
                            rep_completed = True
            else:
                angle_history = []
                per = 50
                bar = np.interp(per, (0, 100), (650, 100))

            cv2.putText(img, f'{int(per)}%', (1200, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if (per >= 90 or per <= 15) else (0, 165, 255), 3)
            
            if per <= 50:
                red = 255
                green = int(255 * (per / 50))
                blue = 0
            else:
                red = int(255 * (1 - (per - 50) / 50))
                green = 255
                blue = 0
            
            bar_color = (blue, green, red)
            cv2.rectangle(img, (1200, 100), (1275, 650), (255, 255, 255), 3)
            cv2.rectangle(img, (1200, int(bar)), (1275, 650), bar_color, cv2.FILLED)
    else:
        angle_history = []
        cv2.rectangle(img, (430, 740), (1335, 620), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, 'Take your position.', (440, 710), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5)
        
    cv2.rectangle(img, (12, 6), (425, 100), (255, 140, 0), cv2.FILLED)
    cv2.putText(img, f'count: {int(count)}', (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 5)
    
    # Calculate FPS
    cTime = time.time()
    pTime = session_state.get("pTime", 0)
    fps = 1 / (cTime - pTime) if pTime != 0 else 0
    
    cv2.putText(img, f'FPS: {int(fps)}', (20, 730), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3)
    
    # Encode processed image back to base64
    _, buffer = cv2.imencode('.jpg', img)
    encoded_img = base64.b64encode(buffer).decode('utf-8')
    
    # Update session state
    session_state["count"] = count
    session_state["dir"] = dir
    session_state["angle_history"] = angle_history
    session_state["pTime"] = cTime
    
    if int(count) > session_state.get("last_rep_count", 0):
        rep_completed = True
        session_state["last_rep_count"] = int(count)
    else:
        rep_completed = False

    return {
        "type": "result",
        "data": "data:image/jpeg;base64," + encoded_img,
        "count": int(count),
        "fps": int(fps),
        "percent": int(per),
        "event": "rep_complete" if rep_completed else None
    }
