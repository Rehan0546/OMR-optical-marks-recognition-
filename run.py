
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 18:36:11 2024

@author: rehan
"""

import pytesseract
import cv2
import tqdm
import argparse
import cv2
import re
import os
import numpy as np
os.environ['TESSDATA_PREFIX'] = 'tesseract/tessdata' 
pytesseract.pytesseract.tesseract_cmd = r'tesseract\tesseract.exe'

def is_point_inside_box(point, box):
    x, y = point
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


def extract_text(img):
    try:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
        img = cv2.erode(img, kernel, iterations=3)
        text = pytesseract.image_to_string(img)
        return text
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

def get_box_loc(box, height, width):
    x1, y1, x2, y2  = box
    return np.array([x1*width, y1*height, x2*width, y2*height],dtype=int)

def create_text_img(img,height, width,boxes):
    new_img = np.ones_like(img)*255
    croping = np.array([boxes[0][0]*width, boxes[0][1]*height, boxes[-1][2]*width, boxes[-1][3]*height],dtype=int)

    for box in boxes:
        box = get_box_loc(box, height, width)
        new_img[box[1]:box[3], box[0]:box[2]] = img[box[1]:box[3], box[0]:box[2]]
    return new_img[croping[1]:croping[3], croping[0]:croping[2]]

def main(image_path = '', true_ans = [],
         mark_per_question = 1):
    true_ans = np.array(true_ans,dtype = int)
    image_ori = cv2.imread(image_path)
    
    height, width, _ = image_ori.shape
    image = cv2.cvtColor(image_ori, cv2.COLOR_BGR2GRAY)
    
    boxes = [[0.15991787899613277, 0.5423675131326653, 0.17106815632027683, 0.5647777034502482],
            [0.17272743301618643, 0.5423786629374416, 0.1838558810271179, 0.5649608819829441],
            [0.1855977250636115, 0.5425155248710508, 0.19696809585738934, 0.5645502961821167],
            [0.20897257506424446, 0.5427628375706881, 0.22002996401547706, 0.5645841713276714],
            [0.22208715265756687, 0.5422779190427552, 0.23314454160879944, 0.5645841713276714],
            [0.23545887883115044, 0.542641607938705, 0.24643055158896268, 0.5644629416956881],
            [0.25860225105466056, 0.5423991486747385, 0.26948820761905234, 0.5640992527997385],
            [0.27140350519376666, 0.5424544665458304, 0.2817294048363461, 0.5641458010404823],
            [0.2944849279242383, 0.5428839979219621, 0.3040515702401575, 0.5639310353524166],
            [0.3064811936854704, 0.5422397008577645, 0.3161996874667217, 0.5639310353524164],
            [0.31847745944670247, 0.5420249351696986, 0.3351811206332281, 0.5639310353524164]]
    
    for_text_img = create_text_img(image,height, width,boxes)
    for_text_img = cv2.resize(for_text_img,None,fx=3,fy=3)
    # print(for_text_img.shape)
    text = extract_text(for_text_img)
    exam_num = ''
    
    if len(text):
        splitted_text = text.split()
        if len(splitted_text)<5:
            splitted_text = splitted_text + ['X']*5
            splitted_text = splitted_text[:5]
        total_M = 'M'+re.sub(r'[A-Za-z]','',re.findall(r'[0-9]+',splitted_text[0])[0])
        total_Q = 'Q'+re.sub(r'[A-Za-z]','',re.findall(r'[0-9]+',splitted_text[1])[0])
        total_S = 'S'+re.sub(r'[A-Za-z]','',re.findall(r'[0-9]+',splitted_text[2])[0])
        total_H = 'H'+re.sub(r'[A-Za-z]','',re.findall(r'[0-9]+',splitted_text[3])[0])
        total_last_num = re.findall(r'[0-9]+',splitted_text[-1])[0]
        
        exam_num = ' '.join([total_M,total_Q,total_S,total_H,total_last_num])
    print(f'Exam Code: {exam_num}')
    
    boxes = [[0.35757842504791637, 0.040395087871174205, 0.3690168696009896, 0.06031261512260457],
        [0.37030830688923966, 0.040569040074243466, 0.38174675144231285, 0.06048656732567383],
        [0.3832226797717415, 0.0403950878711742, 0.39466112432481476, 0.06031261512260456],
        [0.3962398229343045, 0.04034483267107128, 0.4076321867083411, 0.060376442953858474],
        [0.4090397374449016, 0.040407042640893606, 0.4205640591004907, 0.06006539310474688],
        [0.4219716098370512, 0.04015820276160432, 0.43340795957160533, 0.06012760307456919],
        [0.4350794260712709, 0.040220412731426645, 0.4466037477268597, 0.06031423298403616],
        [0.44796731250290295, 0.04034483267107128, 0.45949163415849203, 0.06006539310474687],
        [0.4609871568160876, 0.040220412731426645, 0.4724235065506417, 0.05994097316510223],
        [0.47391902920823675, 0.04028262270124894, 0.4852674070217556, 0.05987876319527992],
        [0.48667495775831676, 0.040095992791782005, 0.4979793496113183, 0.06000318313492455],
        [0.4995628441899483, 0.03990936288231504, 0.5109552079639849, 0.06006539310474686],
        [0.5124431274752946, 0.0399979203890499, 0.5238134982690725, 0.05991133172917469],
        [0.525265034966151, 0.03986105845544082, 0.5367321748730672, 0.059774469795565624],
        [0.5382320961267142, 0.0397241965218317, 0.5496992360336318, 0.05956917689515203],
        [0.5513363532984513, 0.03987744252832378, 0.5625185081671747, 0.059915398228032865],
        [0.5640992316302926, 0.03987744252832378, 0.5754570224393625, 0.05983259675819935],
        [0.577037745902481, 0.03962903811882325, 0.5882784460846532, 0.05950139087886532],
        [0.5897309071495749, 0.03972230633773802, 0.6011012779433527, 0.05943042477744915],
        [0.6025528146404309, 0.03958544440412893, 0.6141167236604858, 0.05956728671105827],
        [0.6154512882716572, 0.039730540820982346, 0.6269668127351428, 0.059575521194302594],
        [0.6284183494322205, 0.03966210985417779, 0.6398371047825676, 0.059507090227498026],
        [0.6415789488190613, 0.03952524792056868, 0.6528041659431313, 0.059507090227498026],
        [0.6543545518855854, 0.03935793033782976, 0.6656889245692758, 0.05930480442072195],
        [0.6674921202234992, 0.03944901195464663, 0.6787620930623959, 0.05912264118708827],
        [0.6803076893374446, 0.03954009357146348, 0.6914488624867537, 0.05930480442072197],
        [0.6929500993796067, 0.03943205431262875, 0.704142792404751, 0.05906925089834365],
        [0.7057721084780313, 0.0393318645341302, 0.7173898404788139, 0.05936982023383928],
        [0.7189424755194078, 0.039448893366782124, 0.729860721373861, 0.058795554523128214],
        [0.7313666863193027, 0.03891641645422216, 0.7427869204889032, 0.05879555452312827],
        [0.7445438795919188, 0.03927140106259549, 0.7555876225251588, 0.05897304682731494],
        [0.7573407419684781, 0.03892809135574714, 0.7687495673127345, 0.05887763959490911],
        [0.7704090328173536, 0.039074779210446865, 0.7817141415675711, 0.05887763959490911],
        [0.7833736070721902, 0.03907477921044687, 0.7944712826343304, 0.05829088817611023],
        [0.7965456145151042, 0.03892809135574714, 0.8075395734832057, 0.05887763959490911],
        [0.8093027555818635, 0.03892809135574714, 0.8207115809261196, 0.05887763959490911]]
    
    for_text_img = create_text_img(image,height, width,boxes)
    for_text_img = cv2.resize(for_text_img,None,fx=1,fy=1)
    # print(for_text_img.shape)
    text = extract_text(for_text_img)
    print(f'Name: {text[:-1]}')
    
    id_num_box = [0.03689516129032258, 0.6589677787282577, 0.3175403225806452, 0.8511548331907614]
    ans_box = [0.8409274193548387, 0.1217564870259481, 0.9850806451612903, 0.9580838323353293]
    
    id_num_box = get_box_loc(id_num_box,height, width)
    ans_box = get_box_loc(ans_box,height, width)
    
    id_detected = detect_small_id_box(image[id_num_box[1]:id_num_box[3], id_num_box[0]:id_num_box[2]])

    print('ID Detected:', "".join(id_detected.astype(str)))

    ans = detect_small_ans_box(image[ans_box[1]:ans_box[3], ans_box[0]:ans_box[2]])
    ans = ans[:len(true_ans)]
    print('True Ans:    ',np.array(true_ans))
    print('ANS Detected:', ans)
    
    correct = list(np.array(ans) == np.array(true_ans)).count(True)
    print('Correct:',correct)
    
    wrong = list(np.array(ans) == np.array(true_ans)).count(False)
    print('Wrong:',wrong)
    
    obtain_marks = correct*mark_per_question
    print('obtain marks:',obtain_marks)

# true_ans = [3,4,5,3,1,2,5,3,2,1]
# main(image_path = 'test.tif', true_ans = true_ans,
#      mark_per_question = 1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--image_path', type=str, default='test.tif', help='Path to the image file')
    parser.add_argument('--true_ans',nargs='+', default=[3,4,5,3,1,2,5,3,2,1], help='True answer')
    parser.add_argument('--mark_per_question', type=int, default=1, help='Mark per question')

    args = parser.parse_args()
    main(args.image_path, args.true_ans, args.mark_per_question)
    