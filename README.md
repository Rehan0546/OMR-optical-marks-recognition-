# OMR-optical-marks-recognition-

This project is a comprehensive optical mark recognition (OMR) solution designed to automate the evaluation of answer sheets. Here’s an overview of its purpose, functionality, and features:

Project Overview
Purpose
The project aims to streamline the process of evaluating answer sheets by automating the detection and extraction of student information and answers. It is capable of handling bulk uploads, extracting relevant data, and uploading results to a cloud-based student database.

Key Features
Automated Answer Detection: The system reads answer sheets, identifies filled answer boxes, and determines the selected answers.
Student Information Extraction: It extracts student ID, name, paper ID, and other relevant details from the answer sheet.
Score Calculation: Based on the provided correct answers, the system calculates the total marks obtained by each student.
Bulk Processing: Capable of processing multiple images in bulk, making it efficient for large-scale assessments.
Cloud Integration: Results, including student information and scores, are uploaded to a cloud-based database for easy access and management.
Detailed Functionality
Setup and Configuration

Google Cloud Vision API: Utilized for accurate text detection from the answer sheets.
OpenCV: Employed for image preprocessing and contour detection.
Image Processing

Grayscale Conversion: Images are converted to grayscale for easier processing.
Thresholding and Morphological Operations: Used to highlight the regions of interest and prepare the image for contour detection.
Text Extraction and Analysis

Text Detection: Google Cloud Vision API is used to detect and extract text from the images, including student details and answer identifiers.
Data Parsing: Extracted text is parsed to identify and separate the student ID, name, paper ID, and other details.
Answer Box Detection

Large Box Detection: Identifies the large bounding boxes that potentially contain the main content (e.g., answers).
Small Box Detection: Within the large boxes, smaller boxes are detected which correspond to individual answer choices.
Data Extraction and Scoring

ID and Answer Detection: Small boxes are analyzed to detect filled answers and student ID boxes.
Score Calculation: Answers detected from the answer boxes are compared against the provided correct answers to calculate the total marks.
Bulk Processing and Cloud Upload

Batch Processing: Multiple images are processed in a loop, with results for each student being extracted and stored.
Cloud Upload: Extracted data, including student information and scores, are uploaded to a cloud database for further processing and record-keeping.
Example Usage
The main function main(image_path, true_ans) processes an individual image and compares detected answers against the provided correct answers. Here's how the function operates:

Read and Preprocess Image: The image is read and converted to grayscale.
Extract Text: Student details are extracted using Google Cloud Vision API.
Detect Answer and ID Boxes: Large and small boxes are detected to identify the locations of answers and student ID.
Score Calculation: Detected answers are compared with the true answers to calculate the score.
Output Results: The results, including student ID, name, and obtained marks, are printed and prepared for upload.
Future Enhancements
Improved Accuracy: Implementing advanced algorithms for more accurate detection of filled answers.
Enhanced Scalability: Optimizing the system for faster processing of even larger batches of images.
User Interface: Developing a user-friendly interface for easier interaction with the system and visualization of results.
This OMR project leverages advanced computer vision techniques and cloud-based services to provide an efficient and automated solution for answer sheet evaluation, making it a valuable tool for educational institutions and examination bodies.
