import cv2
import matplotlib.pyplot as plt
import numpy as np


def draw_matches(img1, kp1, img2, kp2, matches, max_matches=50):
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
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(matched_img, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()


def draw_thick_matches(img1, kp1, img2, kp2, matches, max_matches=50, line_thickness=2):
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

    combined_width = img1_color.shape[1] + img2_color.shape[1]
    combined_height = max(img1_color.shape[0], img2_color.shape[0])
    combined_img = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)
    combined_img[:img1_color.shape[0], :img1_color.shape[1], :] = img1_color
    combined_img[:img2_color.shape[0], img1_color.shape[1]:, :] = img2_color

    for match in matches:
        pt1 = tuple(map(int, kp1[match.queryIdx].pt))
        pt2 = tuple(map(int, kp2[match.trainIdx].pt))
        pt2 = (pt2[0] + img1_color.shape[1], pt2[1])
        cv2.line(combined_img, pt1, pt2, color=(0, 255, 0), thickness=line_thickness)

    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(combined_img, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()



def draw_colored_matches(img1, kp1, img2, kp2, matches, max_matches=50, line_thickness=2):
    if len(matches) > max_matches:
        matches = matches[:max_matches]

    # Convert grayscale to color
    if len(img1.shape) == 2:
        img1_color = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    else:
        img1_color = img1

    if len(img2.shape) == 2:
        img2_color = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    else:
        img2_color = img2

    combined_width = img1_color.shape[1] + img2_color.shape[1]
    combined_height = max(img1_color.shape[0], img2_color.shape[0])
    combined_img = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)
    combined_img[:img1_color.shape[0], :img1_color.shape[1], :] = img1_color
    combined_img[:img2_color.shape[0], img1_color.shape[1]:, :] = img2_color

    color_list = [
        (255, 0, 0),   # Blue
        (0, 255, 0),   # Green
        (0, 0, 255),   # Red
        (255, 255, 0), # Cyan
        (255, 0, 255), # Magenta
        (0, 255, 255), # Yellow
    ]

    max_distance = max([m.distance for m in matches]) if matches else 1
    for i, match in enumerate(matches):
        pt1 = tuple(map(int, kp1[match.queryIdx].pt))
        pt2 = tuple(map(int, kp2[match.trainIdx].pt))
        pt2 = (pt2[0] + img1_color.shape[1], pt2[1])


        normalized_distance = match.distance / max_distance
        gradient_color = (
            int(255 * (1 - normalized_distance)),
            int(255 * normalized_distance),
            0
        )


        cv2.line(combined_img, pt1, pt2, color=gradient_color, thickness=line_thickness)

    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(combined_img, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()