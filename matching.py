from ultralytics import YOLO
import supervision as sv
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

def matchSIFT(image, template):
    template = cv2.flip(template, 1)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    tmp_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    detector = cv2.SIFT_create()
    kp1, desc1 = detector.detectAndCompute(image_gray, None)
    kp2, desc2 = detector.detectAndCompute(tmp_gray, None)

    
    matcher = cv2.BFMatcher(cv2.NORM_L2SQR, crossCheck=True)

    matches = matcher.match(desc1,desc2)
    matches = sorted(matches, key = lambda x:x.distance)

    good_matches = matches[:3]
    kp1_good = [kp1[good_matches[i].queryIdx] for i in range(len(good_matches))]
    
    img = cv2.drawKeypoints(image, kp1_good, image, None, cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    left_top_x = round(sorted(kp1_good, key=lambda x:x.pt[0])[0].pt[0])
    left_top_y = round(sorted(kp1_good, key=lambda x:x.pt[1])[0].pt[1])
    right_bottom_x = round(sorted(kp1_good, key=lambda x:x.pt[0])[-1].pt[0])
    right_bottom_y = round(sorted(kp1_good, key=lambda x:x.pt[1])[-1].pt[1])
    img = cv2.rectangle(img, (left_top_x, left_top_y), (right_bottom_x, right_bottom_y), (0, 0, 255), 3)
    return img



def HSV_similarity(template, image):
    template_hsv = cv2.cvtColor(template, cv2.COLOR_BGR2HSV)
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    method = cv2.HISTCMP_CORREL

    similarity = cv2.compareHist(cv2.calcHist([template_hsv], [0, 1], None, [180, 256], [0, 180, 0, 256]),
                                 cv2.calcHist([image_hsv], [0, 1], None, [180, 256], [0, 180, 0, 256]),
                                 method)
    
    return similarity

def SSIM_similarity(imageA, imageB):
    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
    grayB = cv2.resize(grayB, grayA.shape[::-1])

    (score, diff) = ssim(grayA, grayB, full=True)

    print(f"Similarity: {score:.5f}")
    return score

def detectYOLO(model, image, template, confThreshold, HSVThreshold):
    results = model.predict(image, conf=confThreshold)[0]
    most_similar_box_pos = None
    most_similar_score = 0
    for box in results.boxes:
        cx, cy, w, h = map(float, box.xywh[0])
        box_crop = results.orig_img[round(cy-h/2):round(cy+h/2), round(cx-w/2):round(cx+w/2)]
        similarity = SSIM_similarity(template, box_crop)
        if similarity > most_similar_score:
            most_similar_score = similarity
            most_similar_box_pos = ((round(cx-w/2), round(cy-h/2)), (round(cx+w/2), round(cy+h/2)))
    # detections = sv.Detections.from_ultralytics(results)
    # box_annotator = sv.BoxAnnotator(thickness=4, text_thickness=4, text_scale=2)
    # labels=[f"#{model.names[class_id]} {most_similar_score}" for _, _, conf, class_id, _  in detections] 
    # box_annotator.annotate(scene=image, detections=detections, labels=labels)
    if most_similar_box_pos:
        if  most_similar_score > HSVThreshold:
            box_color = (0, 255, 0)
        else:
            box_color = (0, 0, 255)
        cv2.rectangle(image, most_similar_box_pos[0], most_similar_box_pos[1], box_color, 3)
        cv2.putText(image, str(most_similar_score), (50,50), 1, 3, box_color, 1)
    return image


def match(image, template, method, threshold=None):
    template = cv2.resize(template, (15, 170))
    img_draw = image.copy()
    th, tw = template.shape[:2]
    match_result = cv2.matchTemplate(image, template, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(match_result)

    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
        match_val = min_val
    elif method in [cv2.TM_CCOEFF_NORMED]:
        top_left = max_loc
        match_val = max_val

    bottom_right = (top_left[0] + tw, top_left[1] + th)
    if threshold:
        if threshold[0] < match_val < threshold[1]:
            cv2.rectangle(img_draw, top_left, bottom_right, (0,0,255),10)
            cv2.putText(img_draw, str(match_val), top_left, \
                            cv2.FONT_HERSHEY_PLAIN, 2,(0,255,0), 1, cv2.LINE_AA)
    else:
        cv2.rectangle(img_draw, top_left, bottom_right, (0,0,255),10)
        cv2.putText(img_draw, str(match_val), top_left, \
                    cv2.FONT_HERSHEY_PLAIN, 2,(0,255,0), 1, cv2.LINE_AA)
    return img_draw