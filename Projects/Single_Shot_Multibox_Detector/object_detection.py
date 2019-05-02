# Object Detection

# Importing the libraries
import torch
from torch.autograd import Variable
import cv2
from data import BaseTransform, VOC_CLASSES as labelmap
from ssd import build_ssd
import imageio

# Defining a function that will do the detections
#Frame -> the image / Net -> SSD Neural networks / Transform -> Transform image into correct format 
def detect(frame, net, transform):
    #Get height and width of the image (frame)
    #Shape returns 3 params but we need only first 2 -> :2
    height, width = frame.shape[:2]
    
    #Transform the frame -> Add the rectangular in
    frame_t = transform(frame)[0]
    
    #Convert numpy array to Torch
    x = torch.from_numpy(frame_t).permute(2, 0, 1)
    
    #Add a fake dimension corresponding to the batch.
    x = Variable(x.unsqueeze(0))
    
    #Feed the neural network ssd with the image and we get the output y.
    y = net(x)
    
    #Create the detections tensor contained in the output y.
    detections = y.data

    #Create a tensor object of dimensions [width, height, width, height].
    scale = torch.Tensor([width, height, width, height])
    
    # detections = [batch, number of classes, number of occurences, [score, x0, Y0, x1, y1]]
    for i in range(detections.size(1)):
        j = 0 
        
        #Take into account all the occurrences j of the class i that have a matching score larger than 0.6.
        while detections[0, i, j, 0] >= 0.6:
            #Get the coordinates of the points at the upper left and the lower right of the detector rectangle.
            pt = (detections[0, i, j, 1:] * scale).numpy()
            #Draw a rectangle around the detected object.
            cv2.rectangle(frame, (int(pt[0]), int(pt[1])), (int(pt[2]), int(pt[3])), (255, 0, 0), 2)
            #Put the label of the class right above the rectangle.
            cv2.putText(frame, labelmap[i - 1], (int(pt[0]), int(pt[1])), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA) 
            j += 1
    return frame

#Create SSD Neural Network
#Create an object that is our neural network ssd.
net = build_ssd('test')
net.load_state_dict(torch.load('pretrained_file.pth', map_location = lambda storage, loc: storage))

#Create the transformation
#Create an object of the BaseTransform class, a class that will do the required transformations so that the image can be the input of the neural network.
transform = BaseTransform(net.size, (104/256.0, 117/256.0, 123/256.0)) 

#Implement the object detection on Video

#Open the video.
reader = imageio.get_reader('funny_dog.mp4') 

#Get the fps frequence (frames per second).
fps = reader.get_meta_data()['fps'] 

#Create an output video with this same fps frequence.
writer = imageio.get_writer('output.mp4', fps = fps) 

for i, frame in enumerate(reader):
    
    #Call our detect function (defined above) to detect the object on the frame.
    frame = detect(frame, net.eval(), transform) 
    
    #Add the next frame in the output video.
    writer.append_data(frame) 
    print(i) #Print the number of the processed frame.
#Close the process that handles the creation of the output video.
writer.close() 


