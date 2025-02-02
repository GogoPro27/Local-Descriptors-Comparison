import cv2
import numpy as np

def extract_features(image, method="SIFT"):
    """
    Ги детектира клучните точки и пресметува дескриптори за дадена слика користејќи ја специфицираната метода.
    :param image: Влезна (сива) слика претставена како NumPy низа
    :param method: Текстуален аргумент (string) кој го специфицира екстракторот на карактеристики: SIFT, ORB, BRIEF, BRISK, AKAZE, KAZE или RootSIFT
    :return: (клучни точки, дескриптори)
    """
    if method.upper() == "SIFT":
        extractor = cv2.SIFT_create()

    elif method.upper() == "ORB":
        extractor = cv2.ORB_create()

    elif method.upper() == "BRISK":
        extractor = cv2.BRISK_create()

    elif method.upper() == "AKAZE":
        extractor = cv2.AKAZE_create()

    elif method.upper() == "KAZE":
        extractor = cv2.KAZE_create()

    elif method.upper() == "BRIEF":
        star = cv2.xfeatures2d.StarDetector_create()  # детектор
        brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()  # дескриптор
        keypoints = star.detect(image, None)
        keypoints, descriptors = brief.compute(image, keypoints)
        return keypoints, descriptors

    elif method.upper() == "ROOTSIFT":
        keypoints, descriptors = extract_features(image, method="SIFT")
        if descriptors is None:
            return keypoints, None

        eps = 1e-7
        descriptors /= (descriptors.sum(axis=1, keepdims=True) + eps)
        descriptors = np.sqrt(descriptors)
        return keypoints, descriptors

    else:
        raise ValueError(f"Непознат метод: {method}")

    keypoints, descriptors = extractor.detectAndCompute(image, None)
    return keypoints, descriptors