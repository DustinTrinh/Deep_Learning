import cv2

#Load cascade
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")
smile_cascade = cv2.CascadeClassifier("smile.xml")
#Define function to detect faces

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') # We load the cascade for the face.
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml') # We load the cascade for the eyes.

def detect(gray, frame): # We create a function that takes as input the image in black and white (gray) and the original image (frame), and that will return the same image with the detector rectangles. 
    faces = face_cascade.detectMultiScale(gray, 1.3, 5) # We apply the detectMultiScale method from the face cascade to locate one or several faces in the image.
    for (x, y, w, h) in faces: # For each detected face:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2) # We paint a rectangle around the face.
        roi_gray = gray[y:y+h, x:x+w] # We get the region of interest in the black and white image.
        roi_color = frame[y:y+h, x:x+w] # We get the region of interest in the colored image.
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1,22) # We apply the detectMultiScale method to locate one or several eyes in the image.
        for (ex, ey, ew, eh) in eyes: # For each detected eye:
            cv2.rectangle(roi_color,(ex, ey),(ex+ew, ey+eh), (0, 255, 0), 2) # We paint a rectangle around the eyes, but inside the referential of the face.
        
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.7, 22) # We apply the detectMultiScale method to locate one or several smiles in the image.
        for (sx, sy, sw, sh) in smiles: # For each detected smiles:
            cv2.rectangle(roi_color,(sx, sy),(sx+sw, sy+sh), (0, 0, 255), 2)
    
    return frame # We return the image with the detector rectangles.

#Face detection with webcam

#turn on Webcam
#0 if webcam of computer. 1 is External webcam
video_capture = cv2.VideoCapture(0)

while True:
    #get last frame
    _, frame = video_capture.read()
    #Color transformation
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #Get output of the detect function
    canvas = detect(gray, frame)
    #Display the output
    cv2.imshow('Video', canvas)
    #Exit key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#Turn off webcam
video_capture.release()
# We destroy all the windows inside which the images were displayed.
cv2.destroyAllWindows()   
    
    