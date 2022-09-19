import mediapipe as mp
import cv2
import numpy as np
from tqdm import tqdm

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

def detect_hand(img, padding=0.3, debug=False):
    with mp_hands.Hands(
        static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5
    ) as hands:
        bbox = []
        image = cv2.flip(img, 1)
        # Convert the BGR image to RGB before processing.
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        image_height, image_width, _ = image.shape

        img_sqr = [0, 0, image_width, image_height]
        if image_width < image_height:
            img_sqr[1] = image_width/4
            img_sqr[3] = image_width
        else:
            img_sqr[0] = image_height/4
            img_sqr[2] = image_height

        if not results.multi_handedness:
            tqdm.write("-- WARNING: Couldn't detect hand, returning image dimensions")
            return img_sqr

        if not results.multi_hand_landmarks:
            tqdm.write("-- WARNING: Couldn't detect hand, returning image dimensions")
            return img_sqr

        annotated_image = image.copy()

        landmarks = results.multi_hand_landmarks[0].landmark
        for landmark in landmarks:
            bbox.append([landmark.x * image_width, landmark.y * image_height])

        # Calculate bounding box
        bbox = np.array(bbox)
        bbox_min = bbox.min(0)
        bbox_max = bbox.max(0)
        bbox_size = bbox_max - bbox_min

        # Pad hand bounding box
        bbox_min -= bbox_size * padding
        bbox_max += bbox_size * padding
        bbox_size = bbox_max - bbox_min

        # Convert bbox to square of length equal
        # to longer edge
        diff = bbox_size[0] - bbox_size[1]
        if diff > 0:
            bbox_min[1] -= diff / 2
            bbox_max[1] += diff / 2
            bbox_size[1] = bbox_size[0]
        else:
            bbox_min[0] -= -diff / 2
            bbox_max[0] += -diff / 2
            bbox_size[0] = bbox_size[1]

        # Flip
        tmp = bbox_min[0]
        bbox_min[0] = image_width - bbox_max[0]
        bbox_max[0] = image_width - tmp
        image = cv2.flip(image, 1)

        if debug:
            mp_drawing.draw_landmarks(
                annotated_image,
                results.multi_hand_landmarks[0],
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
            annotated_image = cv2.flip(annotated_image, 1)
            cv2.rectangle(annotated_image, bbox_min.astype(int), bbox_max.astype(int), color=[0, 0, 255], thickness=5)
            cv2.imshow("MediaPipe Prediction", annotated_image)
            cv2.waitKey(0)

        return (*bbox_min, *bbox_size)

