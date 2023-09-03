import dlib
import cv2
import face_recognition
import numpy as np
from scipy.spatial import distance
import logging


class CompareImages:
    def __init__(self):
        self.face_detector = dlib.get_frontal_face_detector()
        shape_predictor_path = "./model/shape_predictor_68_face_landmarks.dat"
        self.shape_predictor = dlib.shape_predictor(shape_predictor_path)
        logging.basicConfig(level=logging.INFO)

    def comapre(self, path_to_image1, path_to_image2):
        image1 = cv2.imread(path_to_image1)
        image2 = cv2.imread(path_to_image2)

        gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        faces1 = self.face_detector(gray_image1)
        faces2 = self.face_detector(gray_image2)

        if len(faces1) == 1 and len(faces2) == 1:
            face1_shape = self.shape_predictor(gray_image1, faces1[0])
            face2_shape = self.shape_predictor(gray_image2, faces2[0])

            face1_landmarks = face_recognition.face_landmarks(image1, [
                (faces1[0].top(), faces1[0].right(), faces1[0].bottom(), faces1[0].left())])[0]
            face2_landmarks = face_recognition.face_landmarks(image2, [
                (faces2[0].top(), faces2[0].right(), faces2[0].bottom(), faces2[0].left())])[0]

            face1_shape_np = [list(point) for point in face1_landmarks.values()]
            face2_shape_np = [list(point) for point in face2_landmarks.values()]

            face_descriptor1 = face_recognition.face_encodings(image1, [
                (faces1[0].top(), faces1[0].right(), faces1[0].bottom(), faces1[0].left())])[0]
            face_descriptor2 = face_recognition.face_encodings(image2, [
                (faces2[0].top(), faces2[0].right(), faces2[0].bottom(), faces2[0].left())])[0]

            euclidean_distance = distance.euclidean(face_descriptor1, face_descriptor2)

            if euclidean_distance < 0.6:
                return True
            else:
                return False
        else:
            logging.info(f'No faces was detect {"faces1 - not found" if faces1 == 0 else "faces2 - not found"}')


CompImages = CompareImages()
