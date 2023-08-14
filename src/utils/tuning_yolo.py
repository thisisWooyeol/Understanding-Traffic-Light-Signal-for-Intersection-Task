# tuning yolov8s model with Traffic Light and Crosswalk dataset

from ultralytics import YOLO

model = YOLO('yolov8n.pt')
model.to('cuda')

# Tranining the model on TL and CW dataset
results = model.train(data='data.yaml',
                      imgsz=512, 
                      epochs=300,
                      batch = 24, 
                      name = 'yolov8s_TL_CW'
                    )

# Test the fine-tuned model on the test dataset
data_test = 'data_test.yaml'
metrics = model.val(data=data_test)
