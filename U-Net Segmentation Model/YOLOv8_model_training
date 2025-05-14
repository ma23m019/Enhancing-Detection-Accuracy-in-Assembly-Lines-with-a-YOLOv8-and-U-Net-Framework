# Download the annotated dataset from Roboflow
pip install roboflow
from roboflow import Roboflow
rf = Roboflow(api_key="Dsj05EpTCoqBvTI3RTTb")
project = rf.workspace("yolo-bafci").project("new-model-wcgn7")
version = project.version(22)
dataset = version.download("yolov8")             

# Train the model
!yolo task=detect \
mode=train \
model=yolov8s.pt \
data="/home/priyanka/Training Notebooks/New-Model-9/data.yaml" \
epochs=30 \
imgsz=640
