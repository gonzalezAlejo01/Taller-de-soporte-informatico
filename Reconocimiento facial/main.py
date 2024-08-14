import cv2
import face_recognition
from deepface import DeepFace

def detect_faces(image_path):
    try:
        # Load the image
        image = face_recognition.load_image_file(image_path)

        # Find all face locations in the image
        face_locations = face_recognition.face_locations(image)

        # Return face locations
        return face_locations
    except Exception as e:
        print(f"Error in detect_faces: {e}")
        return []

def analyze_face(image_path, face_location):
    try:
        # Load the image
        image = cv2.imread(image_path)

        # Extract face coordinates
        top, right, bottom, left = face_location

        # Crop the face from the image
        face_image = image[top:bottom, left:right]

        # Perform age, gender, and emotion analysis
        analysis = DeepFace.analyze(face_image, actions=['age', 'gender', 'emotion'], enforce_detection=False)

        return analysis
    except Exception as e:
        print(f"Error in analyze_face: {e}")
        return {}

def main():
    # Path to the image
    image_path = '1.jpg'

    # Detect faces in the image
    face_locations = detect_faces(image_path)

    if not face_locations:
        print("No faces detected.")
        return

    # Load the image for drawing rectangles and displaying results
    image = cv2.imread(image_path)
    cont = 0
    # Loop through each detected face
    for i, face_location in enumerate(face_locations):
        # Draw rectangle around the face
        top, right, bottom, left = face_location
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)

        # Analyze face for age, gender, and emotion
        analysis = analyze_face(image_path, face_location)

        if not analysis:
            print(f"Analysis failed for Face {i + 1}.")
            continue

        # Display the analysis results near each face
        if isinstance(analysis, list):
            cont += 1
            for j, face_analysis in enumerate(analysis):
                age_text = f"Age: {face_analysis.get('age', 'Not available')}"
                gender_text = f"Gender: {face_analysis.get('dominant_gender', 'Not available').capitalize()}"
                emotion_text = f"Emotion: {face_analysis.get('dominant_emotion', 'Not available')}"

                cv2.putText(image, age_text, (left, top - 110 + 60 * (j + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image, gender_text, (left, top - 90 + 60 * (j + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image, emotion_text, (left, top - 70 + 60 * (j + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                cont += 1
        elif isinstance(analysis, dict):
            age_text = f"Age: {analysis.get('age', 'Not available')}"
            gender_text = f"Gender: {analysis.get('dominant_gender', 'Not available').capitalize()}"
            emotion_text = f"Emotion: {analysis.get('dominant_emotion', 'Not available')}"

            cv2.putText(image, age_text, (left, top - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, gender_text, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, emotion_text, (left, top + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

    # Display the image with rectangles and analysis results
    cv2.imshow('Face Recognition', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
