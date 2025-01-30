import cv2
import numpy as np

def match_features(desc1, desc2, method="bf", ratio_test=True):
    """
    Врши совпаѓање на дескрипторите desc1 и desc2.
    :param desc1: Дескриптори од слика 1
    :param desc2: Дескриптори од слика 2
    :param method: “bf” или “flann”
    :param ratio_test: Дали да се примени Lowe-овиот ratio test
    :return: листа од добри совпаѓања
    """
    if desc1 is None or desc2 is None:
        return []

    # Претвори во float32 ако користиш FLANN за небинарни дескриптори
    if method == "flann" and desc1.dtype != np.float32:
        desc1 = desc1.astype(np.float32)
        desc2 = desc2.astype(np.float32)

    if method == "bf":
        # За бинарни дескриптори како ORB, BRIEF, BRISK, користи NORM_HAMMING
        # За SIFT, RootSIFT, KAZE, AKAZE, користи NORM_L2
        norm_type = cv2.NORM_HAMMING
        # Хевристика: ако димензијата > 32, веројатно не е бинарен дескриптор
        if desc1.shape[1] > 32:
            norm_type = cv2.NORM_L2
        bf = cv2.BFMatcher(norm_type, crossCheck=False)
        # k=2 значи дека секоја карактеристика во desc1 ќе се совпаѓа со двете најдобри во desc2
        matches = bf.knnMatch(desc1, desc2, k=2)

    elif method == "flann":
        # За бинарни дескриптори можеш да користиш FLANN со LSH индекс,
        # но за SIFT можеме да користиме KDTree (или HierarchicalClustering).
        # Ќе претпоставиме SIFT/RootSIFT => KDTree

        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(desc1, desc2, k=2)
    else:
        raise ValueError(f"Unknown matching method: {method}")

    good_matches = []
    if ratio_test:
        ratio_thresh = 0.75
        for m, n in matches:
            if m.distance < ratio_thresh * n.distance:
                good_matches.append(m)
    else:
        # Ако не се користи ratio test, само спој ги сите совпаѓања
        for m, n in matches:
            good_matches.append(m)

    return good_matches

def compute_match_metrics(keypoints1, keypoints2, matches):
    """
    Пример функција за пресметување на некои метрики поврзани со совпаѓањето.
    Тука само го враќаме бројот на совпаѓања.
    """
    num_keypoints1 = len(keypoints1)
    num_keypoints2 = len(keypoints2)
    num_matches = len(matches)

    metrics = {
        "num_keypoints_image1": num_keypoints1,
        "num_keypoints_image2": num_keypoints2,
        "num_good_matches": num_matches
    }
    return metrics