import cv2
import face_recognition

# Initialize the camera
cap = cv2.VideoCapture(0)

# Capture a few frames to get some known face encodings
known_face_encodings = []
known_face_names = []

print("Hold still and look at the camera to capture your face...")
for i in range(10):
    ret, frame = cap.read()
    face_locations = face_recognition.face_locations(frame)
    if len(face_locations) > 0:
        face_encoding = face_recognition.face_encodings(frame, face_locations)[0]
        known_face_encodings.append(face_encoding)
        known_face_names.append("User")

# Main loop
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    # Loop through each face in this frame of video
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # If a match was found in known_face_encodings, just use the first one.
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Face Recognition', frame)

    # Press 'q' to exit the video stream
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()