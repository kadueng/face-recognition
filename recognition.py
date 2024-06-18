import face_recognition
import os
import sys
import cv2
import numpy as np
import math
from datetime import datetime
import time

# Helper function
def face_confidence(face_distance, face_match_threshold=0.6):
    range = (1.0 - face_match_threshold)
    linear_val = (1.0 - face_distance) / (range * 2.0)

    if face_distance > face_match_threshold:
        return str(round(linear_val * 100, 2)) + '%'
    else:
        value = (linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))) * 100
        return str(round(value, 2)) + '%'

class FaceRecognition:
    face_locations = []
    face_encodings = []
    face_names = []
    known_face_encodings = []
    known_face_names = []
    process_current_frame = True
    known_face_images = {}
    unknown_face_images = []
    recognized_people = set()
    frame_count = 0
    start_time = time.time()

    def __init__(self):
        self.encode_faces()

    def encode_faces(self):
        for image in os.listdir('faces'):
            face_image = face_recognition.load_image_file(f"faces/{image}")
            face_encoding = face_recognition.face_encodings(face_image)
            if face_encoding:
                self.known_face_encodings.append(face_encoding[0])
                self.known_face_names.append(os.path.splitext(image)[0])
                self.known_face_images[os.path.splitext(image)[0]] = cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR)
        print(f"Known faces: {self.known_face_names}")

    def process_frame(self, frame):
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        self.face_locations = face_recognition.face_locations(rgb_small_frame)
        self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)

        print(f"Detected {len(self.face_locations)} faces")

        self.face_names = []
        self.unknown_face_images.clear()  # Clear the list of unknown faces for each frame
        for (face_encoding, face_location) in zip(self.face_encodings, self.face_locations):
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"
            confidence = '???'

            # Calculate the shortest distance to face
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)

            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]
                confidence = face_confidence(face_distances[best_match_index])
                self.recognized_people.add(name.split(' ')[0])  # Add recognized person to the set
                print(f"Recognized {name} with confidence {confidence}")
            else:
                # Save the image of the unknown face
                top, right, bottom, left = face_location
                top *= 2
                right *= 2
                bottom *= 2
                left *= 2
                unknown_face_image = frame[top:bottom, left:right]
                self.unknown_face_images.append(unknown_face_image)
                print(f"Detected unknown face")

            if name not in self.face_names:
                self.face_names.append(f'{name} ({confidence})')

    def run_recognition(self):
        video_path = "rtmp://localhost/live/drone"
        video_capture = cv2.VideoCapture(video_path)

        if not video_capture.isOpened():
            sys.exit('Video source not found...')

        while True:
            ret, frame = video_capture.read()
            if not ret:
                break  # Exit loop if video is over
            self.frame_count += 1

            # Calculate FPS
            end_time = time.time()
            fps = self.frame_count / (end_time - self.start_time)

            # Add background color
            frame = cv2.addWeighted(frame, 0.7, np.zeros(frame.shape, frame.dtype), 0.3, 0)

            # Add title
            cv2.putText(frame, 'Real-Time Face Recognition', (10, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)

            # Add date information
            current_date = datetime.now().strftime('%Y-%m-%d')
            cv2.putText(frame, current_date, (10, 60), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

            # Add total recognized people
            cv2.putText(frame, f'Total Recognized People: {len(self.recognized_people)}', (10, 120), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

            # Process every 5th frame
            if self.frame_count % 5 == 0:
                self.process_frame(frame)

            # Display the results
            unique_faces = set()
            for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
                if name in unique_faces:
                    continue
                unique_faces.add(name)

                # Scale back up face locations since the frame we detected in was scaled to 1/2 size
                top *= 2
                right *= 2
                bottom *= 2
                left *= 2

                # Draw the red rectangle and add the person's name
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                # Draw the black rectangle below the red one
                cv2.rectangle(frame, (left, bottom), (right, bottom + 35), (0, 0, 0), cv2.FILLED)
                # Add text inside the black rectangle
                cv2.putText(frame, name, (left + 6, bottom + 25), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

            # Display recognized and unknown faces at the bottom of the frame
            y_offset = frame.shape[0] - 180  # Increase the offset for larger images
            x_offset = 10
            unique_faces.clear()
            for name in self.face_names:
                person_name = name.split(' ')[0]
                if person_name in unique_faces:
                    continue
                unique_faces.add(person_name)
                if person_name in self.known_face_images:
                    face_image = self.known_face_images[person_name]
                    face_image = cv2.resize(face_image, (160, 160))  # Larger size for better visibility
                    frame[y_offset:y_offset + 160, x_offset:x_offset + 160] = face_image

                    # Add black background for name
                    cv2.rectangle(frame, (x_offset, y_offset + 160), (x_offset + 160, y_offset + 180), (0, 0, 0), cv2.FILLED)

                    # Add name below the image
                    cv2.putText(frame, person_name, (x_offset + 5, y_offset + 175), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

                    x_offset += 170  # Increase the offset for the next image

            # Display unknown faces
            for unknown_face_image in self.unknown_face_images:
                if x_offset + 160 > frame.shape[1]:
                    break  # Stop if there is no space for more images
                unknown_face_image = cv2.resize(unknown_face_image, (160, 160))  # Resize to fit
                frame[y_offset:y_offset + 160, x_offset:x_offset + 160] = unknown_face_image

                # Add black background for label
                cv2.rectangle(frame, (x_offset, y_offset + 160), (x_offset + 160, y_offset + 180), (0, 0, 0), cv2.FILLED)

                # Add "Unknown" label below the image
                cv2.putText(frame, "Unknown", (x_offset + 5, y_offset + 175), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

                x_offset += 170  # Increase the offset for the next image

            # Display the resulting image
            cv2.imshow('Face Recognition', frame)

            # Hit 'q' on the keyboard to quit!
            if cv2.waitKey(1) == ord('q'):
                break

        # Release handle to the video file
        video_capture.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    fr = FaceRecognition()
    fr.run_recognition()
