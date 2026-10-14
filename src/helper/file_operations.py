import cv2
import os
import numpy as np

def load_images(root_dir):
    data = []
    labels = []
    for label in sorted(os.listdir(root_dir)):
        folder_path = os.path.join(root_dir, label)
        if os.path.isdir(folder_path):
            for filename in os.listdir(folder_path):
                img_path = os.path.join(folder_path, filename)
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                if img is not None:
                    data.append(img.flatten())
                    labels.append(int(label))
    return np.array(data), np.array(labels)