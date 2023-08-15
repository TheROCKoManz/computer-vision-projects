import mysql.connector
from Database_Connect.utils import DB_Disconnect, DB_Connect
import cv2
import face_recognition
import numpy as np

def fetch_encodings_from_db():
    connection = DB_Connect()
    cursor = connection.cursor()
    try:
        # Fetch user information and face encodings
        fetch_query = """
            SELECT UserID, FirstName, LastName, face_encodings FROM Users;
        """
        cursor.execute(fetch_query)
        results = cursor.fetchall()
        user_ids = []
        first_names = []
        last_names = []
        face_encodings = []
        for row in results:
            user_ids.append(row[0])
            first_names.append(row[1])
            last_names.append(row[2])
            face_encoding = np.fromstring(row[3].strip('[]'), sep=',')
            face_encodings.append(face_encoding)
        cursor.close()
        DB_Disconnect(connection)
        return [user_ids, first_names, last_names, face_encodings]

    except mysql.connector.Error as err:
        print(f"Error: {err}")
        cursor.close()
        DB_Disconnect(connection)
        return

def Frame_Face_Recognition(frame, user_ids, first_names, last_names, face_encodings):
    # Find face locations and encodings in the current frame
    similarity_threshold = 0.99
    face_locations = face_recognition.face_locations(frame)
    face_encodings_frame = face_recognition.face_encodings(frame, face_locations)

    for (top, right, bottom, left), encoding in zip(face_locations, face_encodings_frame):
        distances = face_recognition.face_distance(face_encodings, encoding)
        min_distance_idx = np.argmin(distances)

        if distances[min_distance_idx] <= similarity_threshold:
            user_id = user_ids[min_distance_idx]
            first_name = first_names[min_distance_idx]
            last_name = last_names[min_distance_idx]

            # Display user information on the frame
            label = f"{first_name} {last_name} (ID: {user_id})"
            print(label)
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

def RealTime_FacialRecognition():
    user_ids, first_names, last_names, face_encodings = fetch_encodings_from_db()
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = Frame_Face_Recognition(frame, user_ids, first_names, last_names, face_encodings)

        cv2.imshow('Face Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    # Close the database connection
    print('done')


if __name__ == '__main__':
    RealTime_FacialRecognition()
