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
        # Requires opencv-contrib-python
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
        # BRIEF во OpenCV е всушност само дескриптор; му е потребен детектор за клучни точки.
        # Типично, на пример, може да се користи FAST детектор, а потоа да се проследи до BRIEF екстрактор.
        # Ние ќе направиме едноставна комбинација тука:
        star = cv2.xfeatures2d.StarDetector_create()  # detector
        brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()  # descriptor
        keypoints = star.detect(image, None)
        keypoints, descriptors = brief.compute(image, keypoints)
        return keypoints, descriptors

    elif method.upper() == "ROOTSIFT":
        # RootSIFT не е вграден во OpenCV. Можеме да го пресметаме SIFT, а потоа да направиме постпроцесирање.
        keypoints, descriptors = extract_features(image, method="SIFT")  # get normal SIFT
        if descriptors is None:
            return keypoints, None
        #   Примени RootSIFT постпроцесирање:
        #	    1.	L1-нормирај го секој дескриптор
        #	    2.	Земи го квадратниот корен
        eps = 1e-7
        descriptors /= (descriptors.sum(axis=1, keepdims=True) + eps)
        descriptors = np.sqrt(descriptors)
        return keypoints, descriptors

    else:
        raise ValueError(f"Unknown method: {method}")

    # За методите кои се и детектори и дескриптори во едно
    keypoints, descriptors = extractor.detectAndCompute(image, None)
    return keypoints, descriptors