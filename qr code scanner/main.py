import cv2
import numpy as np 
from pyzbar.pyzbar import decode


cap = cv2.VideoCapture(1)
cap.set(3,640)
cap.set(4,480)

while True:
    sucess,img = cap.read()
    code = decode(img)
    for qrCode in code:
        mytext = qrCode.data.decode('utf-8')
        pts = np.array([qrCode.polygon],np.int32)
        rP = qrCode.rect
        cv2.polylines(img,[pts],True,(0,255,0),3)
        cv2.putText(img,mytext,(rp[0],rp[1]),cv2.FONT_HERSHEY_PLAIN,0.9,(0,0,255),1)
        print(mytext)
    cv2.imshow('Result image',img)
    cv2.waitKey(2)

