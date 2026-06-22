import cv2 
import mediapipe as mp
import time

class poseDetector():
    def __init__(self, mode=False, smooth=True,
                detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(
            static_image_mode=self.mode,
            smooth_landmarks=self.smooth,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
        
    def findPose(self, img, draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        #print(results.pose_landmarks)
        if self.results.pose_landmarks :
            if draw :
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS, self.mpDraw.DrawingSpec(color=(0, 0, 255), thickness=2), self.mpDraw.DrawingSpec(color=(0, 255, 0), thickness=2))
        return img

    def findPosition(self, img, draw= True):
        lmList = []
        if self.results.pose_landmarks :
            
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                # print(id, lm)
                cx, cy = int(lm.x*w), int(lm.y*h)
                lmList.append([id,cx,cy])
                if draw:
                    cv2.circle(img, (cx,cy), 5, (255,0,255), cv2.FILLED)
        return lmList

def main():
    cap = cv2.VideoCapture('Pose Estimation/PoseVideos/4.mp4')
    pTime = 0
    detector = poseDetector()

    while True :
        success, img = cap.read()
        if not success:
            print("finished")
            break
    
        img = cv2.resize(img, (640, 480))
        
        img = detector.findPose(img)
        lmList = detector.findPosition(img, draw=False)
        if lmList != []:
            print(lmList[14])
            cv2.circle(img, (lmList[14][1],lmList[14][2]), 5, (255,0,255), cv2.FILLED)
            
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
        cv2.imshow('Image', img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()