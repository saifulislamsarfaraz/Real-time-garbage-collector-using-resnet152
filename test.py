from controller import wasteOpening
from controller import wasClosing
import cv2
import numpy as np
from gtts import gTTS 
from playsound import playsound
import warnings
warnings.filterwarnings('ignore')
import numpy as np 
import cv2
from keras.models import load_model
import keras.utils as image
def speech(text):
    language = "en"
    output = gTTS(text=text,lang=language, slow = False)
    output.save("./sounds/output.mp3")
    playsound("D:\waste\sounds\output.mp3")
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import cv2
import tensorflow
import keras 
import keras.utils as image
from keras.models import load_model
model = load_model('model.h5')
from keras.preprocessing import image
import numpy as np
import cvlib as cv
from cvlib.object_detection import draw_bbox
labels = classes = ['facemask','paper']
# Initialize detection count
#facedetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
threshold=0.90
cap=cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
font=cv2.FONT_HERSHEY_COMPLEX
total_detection_count = 0
def preprocessing(img):
    #img = image.load_img(img, target_size=(224, 224))
    img = image.img_to_array(img, dtype=np.uint8)
    return img

# Assuming you're using a real-time video feed from a camera (replace with your source)
cap = cv2.VideoCapture(0)  # You'll need to import cv2 for this to work
font_scale = 2
font = cv2.FONT_HERSHEY_PLAIN
count = 0
while True:
    # Read a frame from the video feed   
    ret, frame = cap.read()
    frame_resized = cv2.resize(frame, (224, 224))  # Resize the frame
    #bbox,label,conf = cv.detect_common_objects(frame)
    # image = draw_bbox(frame,bbox,labels,conf)
    
    img = preprocessing(frame_resized)
    p = model.predict(img[np.newaxis,...])
    #print(np.argmax(p))
    text = "Real-time Camera Feed"
    probabilityValue=np.amax(p)
    print(probabilityValue)
    if(probabilityValue>=0.9987):
        #print(classes[np.argmax(p)])
        pred = labels[np.argmax(p[0],axis=-1)]
        print(pred) 
        cv2.putText(frame, "Class: " + pred + ", Probalility: " + str(probabilityValue), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        count+=1
        if count>=10:
            if pred == 'facemask':
                speech("Found a " + pred + " The Trash box is opening")
                wasteOpening(1)
                count = 0
                if count == 0:
                    speech("The Trash box is closing now")
                    wasClosing(0)
            elif pred =='paper':
                speech("Found a " + pred + " The Trash box is opening")
                wasteOpening(1)
                count = 0
                if count == 0:
                    speech("The Trash box is closing now")
                    wasClosing(0)
        elif count>=10:
            count = 0
    cv2.imshow("Real-time Object Detection", frame)
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        #wasteDetect(1)
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()

#print(f"Total object detections: {total_detection_count}")

