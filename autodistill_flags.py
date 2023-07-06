# use roboflows autodistill to build a efficient model from a large one
#pip install -q autodistill autodistill-grounded-sam autodistill-yolov8 supervision
from autodistill_grounded_sam import GroundedSAM
from autodistill.detection import CaptionOntology
from autodistill_yolov8 import YOLOv8

# define an ontology to map class names to our GroundingDINO prompt
# the ontology dictionary has the format {caption: class}
# where caption is the prompt sent to the base model, and class is the label that will
# be saved for that caption in the generated annotations
base_model = GroundedSAM(ontology=CaptionOntology({"a flag": "Fleg", "a flag with a red cross on a white background and a red hand":"Ulster Fleg"}))
# label all images in a folder called `samples` and save the annotations to `dataset`
base_model.label(
  input_folder="./sample",
  output_folder="./dataset")

target_model = YOLOv8("yolov8n.pt")
target_model.train("./dataset/data.yaml", epochs=200)

# run inference on the new model
#pred = target_model.predict("threeflags.jpg", confidence=0.5)
# print(pred)

# optional: upload your model to Roboflow for deployment
#from roboflow import Roboflow

#rf = Roboflow(api_key="API_KEY")
#project = rf.workspace().project("PROJECT_ID")
#project.version(DATASET_VERSION).deploy(model_type="yolov8", model_path=f"./runs/detect/train/")