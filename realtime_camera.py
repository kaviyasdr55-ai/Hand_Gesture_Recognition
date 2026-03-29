import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("model.h5")

labels = ["Palm","L","Fist","Fist Move","Thumb","Index","OK","Palm Move","C","Down"]

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    ret, frame = cap.read()
    if not ret:
       print("Camera not detected")
       break

    roi = frame[100:300,100:300]
    img = cv2.resize(roi,(64,64))
    img = img/255.0
    img = np.reshape(img,(1,64,64,3))

    prediction = model.predict(img, verbose=0)
    classID = np.argmax(prediction)

    cv2.putText(frame, labels[classID], (100,90),
                cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

    cv2.rectangle(frame,(100,100),(300,300),(0,255,0),2)

    cv2.imshow("Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()