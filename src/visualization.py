import cv2
import matplotlib.pyplot as plt
import numpy as np


def draw_matches(img1, kp1, img2, kp2, matches, max_matches=75):
    """
    Нацртај до ‘max_matches’ најдобри совпаѓања помеѓу img1 и img2.
    :param img1: прва слика (BGR или сива скала)
    :param kp1: клучни точки во првата слика
    :param img2: втора слика
    :param kp2: клучни точки во втората слика
    :param matches: листа од совпаѓања
    :param max_matches: број на најдобри совпаѓања за прикажување
    :return: None (прикажува matplotlib фигура)
    """
    if len(matches) > max_matches:
        matches = matches[:max_matches]

    if len(img1.shape) == 2:
        img1_color = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    else:
        img1_color = img1

    if len(img2.shape) == 2:
        img2_color = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    else:
        img2_color = img2

    matched_img = cv2.drawMatches(
        img1_color, kp1,
        img2_color, kp2,
        matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        matchesThickness=5
    )

    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(matched_img, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()