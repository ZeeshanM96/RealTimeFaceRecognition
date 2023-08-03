# Real-Time Face Recognition
This project performs real-time face detection and recognition on video captured from a webcam. It uses the Google Cloud Vision API for face detection and the `face_recognition` library for face recognition.

# Face recognition

Technologies Used
`Python`: The project is implemented in Python.
`OpenCV`: Used to capture video from the webcam, draw bounding boxes and labels around detected faces, and display the video.
`Google Cloud Vision API`: Used for face detection. Given an image, it returns bounding polygons for all detected faces in the image.
`face_recognition`: Used for face recognition. It computes face encodings (numeric representations of face images) and compares them to recognize faces.

# Setup Instructions
Clone the repository:

`git clone https://github.com/ZeeshanM96/RealTimeFaceRecognition.git`

`cd RealTimeFaceRecognition`

# Install dependencies:

`pip install opencv-python google-cloud-vision face_recognition`

# Google Cloud Vision API setup:
Follow the Google Cloud Vision API Quickstart guide to create a Google Cloud project, enable the Vision API, and create an API key.
Download the JSON file with your API key and save it in the project directory.
Set the `GOOGLE_APPLICATION_CREDENTIALS` environment variable to the path of your API key JSON file, either in your environment or directly in the Python script.

# Prepare training data:
Collect several images of each person you want to recognize. These images should show the person's face from different angles and in different lighting conditions.
Save these images in the known_faces directory. Each image file should be named as the person's name followed by a number, like Zeeshan-1.png, Zeeshan-2.png, etc.

Run the script:

`python facetracker.py`

# Notes
This system is designed to recognize faces of known individuals and label them by name in real-time video. Unknown faces will be labeled as "Unknown".
The user is required to provide their own Google Cloud Vision API key and include images of known individuals in the `known_faces` directory for the system to function properly.
The accuracy of face recognition depends on the quality and variety of the training images and the conditions in which the video is captured. It may not work well with low-quality images, poor lighting, extreme face angles, or partial face occlusions.
The face_recognition library is based on the `dlib` library, which uses a method called Histogram of Oriented Gradients (HOG) for face detection and a neural network for face recognition. Different models or methods may give better results.
