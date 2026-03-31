\# YOLOv8 Helmet Detection



Real-time helmet detection using YOLOv8 for industrial safety applications. This project provides an end-to-end pipeline from dataset collection and annotation to training, inference, REST API deployment, and Docker containerization.



---



\##  Project Overview



\- \*\*Problem Statement:\*\* Ensure safety compliance by detecting helmets in industrial environments.

\- \*\*Goal:\*\* Train a YOLOv8 model for accurate real-time helmet detection.

\- \*\*Pipeline:\*\*

&nbsp; 1. Dataset collection \& annotation

&nbsp; 2. Model training via Jupyter Notebook and modular scripts

&nbsp; 3. Inference on images/videos

&nbsp; 4. REST API integration using FastAPI

&nbsp; 5. Docker deployment for easy portability

&nbsp; 6. Live testing and evaluation



---



\##  Project Structure



helmet-detection/

├── data/

│ ├── images/

│ │ ├── train/

│ │ └── val/

│ └── labels/

├── train/

├── val/

├── notebooks/

│ └── helmet\_detection\_yolov8.ipynb

├── model/

│ └── yolov8/

├── src/

│ ├── train.py

│ ├── detect.py

│ └── utils.py

├── app/

│ └── app.py

├── README.md

├── requirements.txt

├── Dockerfile

├── .gitignore

└── .env



---



\##  Installation



1\. \*\*Clone the repository:\*\*



```bash

git clone <repository\_url>

cd helmet-detection



2\. \*\*Install dependencies:\*\*

pip install -r requirements.txt

Ensure ultralytics YOLOv8 is installed for model training and inference.



3\. \*\*Dataset Preparation:\*\*



-Collect helmet images from Roboflow, Kaggle, CCTV footage, or custom sources.



-Annotate images using LabelImg or Roboflow in YOLO .txt format.



-Organize the dataset:

data/

├── images/

│   ├── train/

│   └── val/

└── labels/

&nbsp;   ├── train/

&nbsp;   └── val/

-Create and configure data.yaml specifying classes and dataset paths.



4\. \*\*Model Training:\*\*



a. Via Notebook:



-Open notebooks/helmet\_detection\_yolov8.ipynb



-Configure data.yaml paths



-Train the YOLOv8 model using the notebook cells



b. Via Modular Script:

&nbsp;python src/train.py --data data/data.yaml --epochs 50 --img-size 640 --batch-size 16



5\. \*\*Inference:\*\*

from ultralytics import YOLO



model = YOLO("model/yolov8/best.pt")

results = model.predict(source="data/images/val/sample1.jpg", conf=0.5, save=True, save\_txt=True)

results.show()



\- Outputs saved in runs/detect/exp/ folder



6\. \*\*REST API with FastAPI:\*\*



\- Start the API:

uvicorn app.app:app --reload



-Endpoint: /predict

Accepts image uploads and returns JSON containing:



* labels
* confidence
* bounding boxes



-Test via Postman or curl:

curl -X POST -F "file=@sample.jpg" http://localhost:8000/predict



7\. \*\*Docker Deployment:\*\*



a. Build the Docker image:



&nbsp; docker build -t helmet-detector .





b. Run the container:



&nbsp; docker run -p 8000:8000 helmet-detector





c. Access API at http://localhost:8000/predict



Live Testing \& Improvements



Webcam feed detection



Batch inference on multiple images



Analyze accuracy, false positives, and edge cases





8\. \*\*Future enhancements:\*\*



* Helmet-type classification
* Sound alerts or visual dashboard
* Multi-camera integration







