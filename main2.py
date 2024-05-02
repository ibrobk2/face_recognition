import cv2
import dlib
import face_recognition
# import playsound
# import time
import os

# Load a pre-trained face detection model from dlib
detector = dlib.get_frontal_face_detector()

# Path to the directory containing images of known faces
known_faces_dir = "known_faces"

# Initialize the camera
cap = cv2.VideoCapture(0)

# Initialize the sound alert
# alert_sound = "alarm.wav"  # Replace with the path to your sound file

# Set the duration for the alert interval (in seconds)
# alert_interval = 3

# Load known faces and their names
known_faces = ["ibrahim.jpg", "biden.jpg"]
known_names = ["Ibrahim Bakori", "Joe Biden"]

for filename in os.listdir(known_faces_dir):
    name = os.path.splitext(os.path.basename(filename))[0]
    image = face_recognition.load_image_file(os.path.join(known_faces_dir, filename))
    encoding = face_recognition.face_encodings(image)[0]
    known_faces.append(encoding)
    known_names.append(name)

# Initialize variables to track time
# start_time = time.time()

while True:
    # Capture a frame from the camera
    ret, frame = cap.read()

    # Convert the frame to RGB for face recognition
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces using dlib
    faces = detector(rgb_frame)

    for face in faces:
        # Get face locations and encodings
        face_locations = [(face.top(), face.right(), face.bottom(), face.left())]
        face_encodings = face_recognition.face_encodings(rgb_frame, [face])

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Check if the face matches any known face
            matches = face_recognition.compare_faces(known_faces, face_encoding)
            name = "Unknown"

            if True in matches:
                first_match_index = matches.index(True)
                name = known_names[first_match_index]

                # Draw a rectangle around the face in green for known faces
                color = (0, 255, 0)
            else:
                # Draw a rectangle around the face in red for unknown faces
                color = (0, 0, 255)

            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.rectangle(frame, (left, bottom -35), (right, bottom), color, cv2.FILLED)

            # Display the name
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, color, 1)

            # Check if it's time to trigger the alert
            # if name == "Unknown" and time.time() - start_time > alert_interval:
            #     # playsound.playsound(alert_sound)
            #     start_time = time.time()

    # Display the frame
    cv2.imshow('Face Detection System', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()
