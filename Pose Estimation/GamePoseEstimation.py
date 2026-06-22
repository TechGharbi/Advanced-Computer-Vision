import cv2 
import mediapipe as mp
import time
import PoseEstimationModule as pm

cap = cv2.VideoCapture('Pose Estimation/PoseVideos/6.mp4')
pTime = 0
detector = pm.poseDetector()

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