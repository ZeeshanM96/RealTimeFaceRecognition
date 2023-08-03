# Import necessary libraries
import cv2
import face_recognition
from google.cloud import vision
import os
from google.cloud.vision_v1 import Image

# Set the path to your Google Cloud Vision credentials
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "C:\\Users\\zeesh\\Desktop\\Python\\Projects\\FaceTracker\\quiet-odyssey-394722-ba2fca0d219c.json"

# Load the known face images and get the face encodings
known_face_encodings = []
known_face_names = []
# Loop over all files in the known_faces directory
for filepath in os.listdir('known_faces'):
    # Load an image file
    image = face_recognition.load_image_file('known_faces/' + filepath)
    # Compute face encodings for the image
    encodings = face_recognition.face_encodings(image)
    # Only proceed if at least one face was detected
    if encodings:
        # Add the first face encoding and the corresponding name to the known faces
        encoding = encodings[0]
        known_face_encodings.append(encoding)
        # Extract the name from the filename (assumes the name is before the dash)
        known_face_names.append(filepath.split('-')[0])

# Initialize the Google Cloud Vision client
client = vision.ImageAnnotatorClient()

# Initialize the video capture object for the default camera (camera index 0)
cap = cv2.VideoCapture(0)

# Start the main loop
while True:
    # Capture a frame from the camera
    ret, frame = cap.read()

    # Convert the frame from BGR (OpenCV's default color format) to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Encode the image as a PNG and convert it to bytes for the Google Cloud Vision API
    success, encoded_image = cv2.imencode('.png', image)
    content = encoded_image.tobytes()

    # Create a vision_v1.Image object for the Google Cloud Vision API
    vision_image = Image(content=content)

    # Perform face detection on the image
    response = client.face_detection(image=vision_image)
    # Get the face annotations from the response
    faces = response.face_annotations

    # Loop over all detected faces
    for face in faces:
        # Get the vertices of the bounding polygon for the face
        vertices = [(vertex.x, vertex.y) for vertex in face.bounding_poly.vertices]

        # Draw a green rectangle around the face
        cv2.rectangle(image, vertices[0], vertices[2], (0, 255, 0), 2)

        # Compute face encoding for the face
        face_location = (vertices[0][1], vertices[2][0], vertices[2][1], vertices[0][0])  # top, right, bottom, left
        face_encoding = face_recognition.face_encodings(image, [face_location])[0]

        # Compare the face encoding with the known face encodings
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)

        # If there's a match, get the name of the matching known face
        if True in matches:
            match_index = matches.index(True)
            name = known_face_names[match_index]
        else:
            # If there's no match, label the face as "Unknown"
            name = "Unknown"

        # Draw the name of the person (or "Unknown") on the image
        cv2.putText(image, name, vertices[0], cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Display the image
    cv2.imshow('Frame', image)

    # If the 'q' key is pressed, break the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
