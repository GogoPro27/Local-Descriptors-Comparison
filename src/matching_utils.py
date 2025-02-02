import cv2

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

    if method == "bf":
        norm_type = cv2.NORM_HAMMING
        if desc1.shape[1] > 32:
            norm_type = cv2.NORM_L2
        bf = cv2.BFMatcher(norm_type, crossCheck=False)
        matches = bf.knnMatch(desc1, desc2, k=2)
    else:
        raise ValueError(f"Непознат метод: {method}")

    good_matches = []
    if ratio_test:
        ratio_thresh = 0.75
        for m, n in matches:
            if m.distance < ratio_thresh * n.distance:
                good_matches.append(m)
    else:
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