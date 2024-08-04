
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 18:36:11 2024

@author: rehan
"""


import tqdm
from google.cloud import vision
import cv2
import re
import os
import numpy as np

credientials_path = 'credential.json'
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credientials_path


def is_point_inside_box(point, box):
    x, y = point
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


def extract_text(image, client=vision.ImageAnnotatorClient()):
    try:
        # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kernel = np.ones((1, 3), np.uint8)

        img = cv2.erode(image, kernel, iterations=2)

        content = cv2.imencode('.jpg', img)[1].tobytes()
        image = vision.Image(content=content)
        responses = client.text_detection(image=image)
        return responses
    except:
        return ''


def create_groups(sorted_list, max_distance=10):
    groups = []
    current_group = []

    for i in range(len(sorted_list)):
        if not current_group:
            current_group.append(sorted_list[i])
        else:
            if sorted_list[i] - current_group[-1] <= max_distance:
                current_group.append(sorted_list[i])
            else:
                groups.append(current_group)
                current_group = [sorted_list[i]]

    if current_group:
        groups.append(current_group)

    return groups


def detect_large_boxes(image, min_box_area=0):

    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    thresh = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 4)

    kernel = np.ones((2, 2), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    preprocessed_image = cv2.medianBlur(sure_bg, 5)
    edges = cv2.Canny(preprocessed_image, 10, 255, apertureSize=3)
    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    large_boxes = []
    largest_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)

        if area > min_box_area:
            x, y, w, h = cv2.boundingRect(contour)
            if w*h >= largest_area:
                large_boxes = [x, y, x+w, y+h]
                largest_area = w*h
    
    print('large_boxes',large_boxes)
    return image[large_boxes[1]:large_boxes[3], large_boxes[0]:large_boxes[2]]


def detect_small_id_box(image,
                        min_box_area=2):

    edges = cv2.Canny(image, 0, 180, apertureSize=3)

    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    centers = []
    dimensions = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_box_area:

            x, y, w, h = cv2.boundingRect(contour)
            # print(box_center)
            if abs(w-h) <= 10:
                box_center = (x+(w//2), y+(h//2))
                dimensions.append(np.mean([w, h]))
                centers.append(box_center)
                boxes.append((x, y, w, h))

                # cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                # cv2.circle(image, box_center, 5, (0,0,255), -1)

    avg_dim = int(np.mean(sorted(dimensions)))-10
    centers = np.array(centers).reshape(-1, 2)

    columns = create_groups(sorted(centers[:, 0]))
    rows = create_groups(sorted(centers[:, 1]))

    columns = [int(np.mean(h)) for h in columns if len(h) >= 2]
    rows = [int(np.mean(h)) for h in rows if len(h) >= 2]

    # print(len(columns),
    # len(rows))

    boxes = []
    id_detected = ''
    all_values = []
    for col_pixel in columns[1:-1]:
        each_col_filled = []
        for row_pixel in rows:
            # cv2.circle(image, (col_pixel,row_pixel), 5, (0,0,255), -1)

            if not len(boxes):
                boxes.append([col_pixel-avg_dim, row_pixel-avg_dim,
                             col_pixel+avg_dim, row_pixel+avg_dim])
                
                x1, y1, x2, y2 = boxes[-1]
                box_check = image[y1:y2, x1:x2]
                count = np.sum(box_check <= 10)
                height, width = box_check.shape
                filled_percentage = count/(height*width)
                if not filled_percentage>=0.4:
                    filled_percentage = 0.0
                each_col_filled.append(filled_percentage)

                # cv2.rectangle(image, (col_pixel-avg_dim,row_pixel-avg_dim),
                #               (col_pixel+avg_dim,row_pixel+avg_dim), (0, 255, 0), 2)
                # each_col_num += 1
                # continue/
            found = False
            for box in reversed(boxes):
                if is_point_inside_box((col_pixel, row_pixel), box):
                    found = True
                    break

            if not found:

                boxes.append([col_pixel-avg_dim, row_pixel-avg_dim,
                             col_pixel+avg_dim, row_pixel+avg_dim])

                # cv2.rectangle(image, (col_pixel-avg_dim,row_pixel-avg_dim),
                #               (col_pixel+avg_dim,row_pixel+avg_dim), (0, 255, 0), 2)

                x1, y1, x2, y2 = boxes[-1]
                box_check = image[y1:y2, x1:x2]
                count = np.sum(box_check <= 10)
                height, width = box_check.shape
                filled_percentage = count/(height*width)
                if not filled_percentage>=0.4:
                    filled_percentage = 0.0
                each_col_filled.append(filled_percentage)
        all_values.append(each_col_filled)

    id_detected = np.argmax(np.array(all_values),axis = 1)
    # cv2.imwrite('id_detecting.jpg',image)
    return id_detected


def detect_small_ans_box(image,
                         min_box_area=2):

    edges = cv2.Canny(image, 0, 180, apertureSize=3)

    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    centers = []
    dimensions = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_box_area:

            x, y, w, h = cv2.boundingRect(contour)
            # print(box_center)
            if abs(w-h) <= 10:
                box_center = (x+(w//2), y+(h//2))
                dimensions.append(np.mean([w, h]))
                centers.append(box_center)
                boxes.append((x, y, w, h))

                # Draw the bounding box on the original image (optional)
                # cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                # cv2.circle(image, box_center, 5, (0,0,255), -1)

    avg_dim = int(np.mean(sorted(dimensions)))-10
    centers = np.array(centers).reshape(-1, 2)

    columns = create_groups(sorted(centers[:, 1]))
    rows = create_groups(sorted(centers[:, 0]))

    columns = [int(np.mean(i)) for i in columns if len(i) > 2]
    rows = [int(np.mean(i)) for i in rows if len(i) > 2]
    # print(columns)
    # print(len(columns))
    # print(len(rows))

    boxes = []
    id_detected = ''
    all_values = []
    for row_pixel in columns:
        # each_col_num = 1
        each_col_filled = []
        for col_pixel in rows[1:-2]:
            # cv2.circle(image, (col_pixel,row_pixel), 5, (0,0,255), -1)

            if not len(boxes):
                boxes.append([col_pixel-avg_dim, row_pixel-avg_dim,
                             col_pixel+avg_dim, row_pixel+avg_dim])
                
                x1, y1, x2, y2 = boxes[-1]
                box_check = image[y1:y2, x1:x2]
                count = np.sum(box_check <= 10)
                height, width = box_check.shape
                filled_percentage = count/(height*width)
                if not filled_percentage>=0.4:
                    filled_percentage = 0.0
                each_col_filled.append(filled_percentage)

                # cv2.rectangle(image, (col_pixel-avg_dim,row_pixel-avg_dim),
                #               (col_pixel+avg_dim,row_pixel+avg_dim), (0, 255, 0), 2)
                # each_col_num += 1
                # continue/
            found = False
            for box in reversed(boxes):
                if is_point_inside_box((col_pixel, row_pixel), box):
                    found = True
                    break

            if not found:

                boxes.append([col_pixel-avg_dim, row_pixel-avg_dim,
                             col_pixel+avg_dim, row_pixel+avg_dim])

                # cv2.rectangle(image, (col_pixel-avg_dim,row_pixel-avg_dim),
                #               (col_pixel+avg_dim,row_pixel+avg_dim), (0, 255, 0), 2)

                x1, y1, x2, y2 = boxes[-1]
                box_check = image[y1:y2, x1:x2]
                count = np.sum(box_check <= 10)
                height, width = box_check.shape
                filled_percentage = count/(height*width)
                if not filled_percentage>=0.4:
                    filled_percentage = 0.0
                each_col_filled.append(filled_percentage)
        all_values.append(each_col_filled)

    id_detected = np.argmax(np.array(all_values),axis = 1)
    # cv2.imwrite('ans_detecting.jpg',image)
    return id_detected+1

def main(image_path = '', true_ans = []):
    image_ori = cv2.imread(image_path)
    
    height, width, _ = image_ori.shape
    image = cv2.cvtColor(image_ori, cv2.COLOR_BGR2GRAY)
    responses = extract_text(image)
    if responses:
        
        all_text = responses.text_annotations[0].description
        total_M = re.sub(r'[A-Za-z]','',re.findall(r'M[0-9]+',all_text)[0])
        total_Q = re.sub(r'[A-Za-z]','',re.findall(r'Q[0-9]+',all_text)[0])
        total_S = re.sub(r'[A-Za-z]','',re.findall(r'S[0-9]+',all_text)[0])
        total_H = re.sub(r'[A-Za-z]','',re.findall(r'H[0-9]+',all_text)[0])
    
        all_text = all_text.split('\n')
        all_details = {'text': [],
                       'bboxes': []}
        # prev_i = 0
        i = 1
        ans_box = []
        item_boxes = []
        for text in all_text:
            joined_text = text.replace(' ', '')
    
            line_data = []
            # print('text_line:',text)
            # print('splitter_text:',splitter_text)
            detected_Word = ''
            for word in responses.text_annotations[i:]:
                i += 1
    
                detected_Word += str(word.description)
                line_data.append(word)
                if len(detected_Word) == len(joined_text):
                    break
                if len(detected_Word) > len(joined_text):
                    i = i-1
                    break
    
            boxes = []
            text = ''
            for word_data in line_data:
    
                text += word_data.description+' '
                for point in word_data.bounding_poly.vertices:
                    boxes.append([point.x, point.y])
    
                # print(word_data.description)
                # if is_alphanumeric(word_data.description):
                #     boxes_ = np.array(boxes).reshape(-1,2)
                #     x1,y1,x2,y2 = np.array([np.min(boxes_[-4:,0]),np.min(boxes_[-4:,1]), np.max(boxes_[-4:,0]),np.max(boxes_[-4:,1])])
                #     cv2.rectangle(image, (x1,y1),(x2,y2), (255,255,255),-1)
    
            text = text.lower().strip()
    
            boxes = np.array(boxes).reshape(-1, 2)
            boxes = np.array([np.min(boxes[:, 0]), np.min(
                boxes[:, 1]), np.max(boxes[:, 0]), np.max(boxes[:, 1])])
    
            all_details['text'].append(text)
            all_details['bboxes'].append(boxes)
            # prev_i = i
    
            if 'identification' in text and not 'exemple' in text:
                id_num_box = [boxes[0]-50, boxes[1], width//3, height]
    
            if 'des reponses' in text:
                ans_box = [boxes[0]-(width//10), boxes[1], width, height]
    
        for index, text in enumerate(all_details['text']):
            boxes = all_details['bboxes'][index]
            if 'item' in text and len(ans_box):
                if is_point_inside_box((boxes[0], boxes[-1]), ans_box):
                    item_boxes.append(boxes)
    
        item_boxes = np.array(item_boxes).reshape(-1, 4)
        item_box = [np.min(item_boxes[:, 0]), np.max(
            item_boxes[:, -1]), width, height]
        
        print('item_boxes',item_box)
        print('id_num_box',id_num_box)
        
        id_detecting_box = detect_large_boxes(
            image[id_num_box[1]:id_num_box[3], id_num_box[0]:id_num_box[2]])
        id_detected = detect_small_id_box(id_detecting_box)
        
        print('total_M',total_M)
        print('total_Q',total_Q)
        print('total_S',total_S)
        print('total_H',total_H)
        
        print('id_detected:', id_detected)
    
        ans_detecting_box = detect_large_boxes(
            image[item_box[1]:item_box[3], item_box[0]:item_box[2]])
        ans = detect_small_ans_box(ans_detecting_box)
        ans = ans[:len(true_ans)]
        print('ans:', ans)
        
        obtain_marks = list(np.array(ans) == np.array(true_ans)).count(True)
        print('obtain marks:',obtain_marks)
        
true_ans = [3,4,5,3,1,2,5,3,2,1]
main(image_path = 'test.tif', true_ans = true_ans)
    
