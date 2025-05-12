import mediapipe as mp


class SimpleFaceDetector:
    def __init__(self):
        self.face_detection = mp.solutions.face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5
        )

    def detect_face(self, image):
        # Process the image and detect faces.
        results = self.face_detection.process(image)

        if not results.detections:  # Face not detected
            return False

        if len(results.detections) != 1:
            return False
        return True

    def close(self):
        self.face_detection.close()
