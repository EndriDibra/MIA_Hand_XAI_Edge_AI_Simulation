# Importing the required libraries 
import os 
from ultralytics import YOLO 


# Ensuring the correct path
os.chdir("C:/Users/User/Documents/AI_Robotics Projects/AAU/Summer/Dataset_YOLO")

# Loading base model to train it on the datasets
# and be able to recognise licence plates
model = YOLO("yolo26n.pt")

# Starting training process
model.train(data="C:\\Users\\User\\Documents\\AI_Robotics Projects\\AAU\\Summer\\YOLOs\\data.yaml",
             epochs=20,
             imgsz=416,
             seed=42)