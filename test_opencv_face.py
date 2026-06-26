import cv2
print("OpenCV Version:", cv2.__version__)
print("FaceDetectorYN available:", hasattr(cv2, "FaceDetectorYN"))
print("FaceRecognizerSF available:", hasattr(cv2, "FaceRecognizerSF"))
