# to run the code command is
# streamlit run Count_V.py

# to stop command is
# ctrl+c

import cv2
import numpy as np
import streamlit as st
import tempfile

def get_centroid(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    return x + x1, y + y1

def process_vehicle_detection(video_path):
    min_contour_width = 40  
    min_contour_height = 40  
    offset = 10  
    line_height = 550  
    matches = []
    vehicles = 0
    
    cap = cv2.VideoCapture(video_path)
    cap.set(3, 1920)
    cap.set(4, 1080)
    
    ret, frame1 = cap.read()
    ret, frame2 = cap.read()
    
    while ret:
        d = cv2.absdiff(frame1, frame2)
        grey = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(grey, (5, 5), 0)
        ret, th = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(th, np.ones((3, 3)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        closing = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        for c in contours:
            (x, y, w, h) = cv2.boundingRect(c)
            contour_valid = (w >= min_contour_width) and (h >= min_contour_height)
            if not contour_valid:
                continue
            
            cv2.rectangle(frame1, (x-10, y-10), (x+w+10, y+h+10), (255, 0, 0), 2)
            cv2.line(frame1, (0, line_height), (1200, line_height), (0, 255, 0), 2)
            centroid = get_centroid(x, y, w, h)
            matches.append(centroid)
            cv2.circle(frame1, centroid, 5, (0, 255, 0), -1)
            
            cx, cy = centroid
            for (mx, my) in matches:
                if my < (line_height+offset) and my > (line_height-offset):
                    vehicles += 1
                    matches.remove((mx, my))
            
        cv2.putText(frame1, f"Total Vehicles: {vehicles}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 170, 0), 2)
        cv2.imshow("Vehicle Detection", frame1)
        
        if cv2.waitKey(1) == 27:
            break
        
        frame1 = frame2
        ret, frame2 = cap.read()
        
    cap.release()
    cv2.destroyAllWindows()
    return vehicles  # Returning the vehicle count

def process_lane_detection(video_path):
    cap = cv2.VideoCapture(video_path)
    
    def process_frame(frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        height, width = edges.shape
        mask = np.zeros_like(edges)
        polygon = np.array([[
            (width * 0.1, height),
            (width * 0.45, height * 0.6),
            (width * 0.55, height * 0.6),
            (width * 0.9, height)
        ]], np.int32)
        cv2.fillPoly(mask, polygon, 255)
        masked_edges = cv2.bitwise_and(edges, mask)
        lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, 50, minLineLength=100, maxLineGap=50)
        line_img = np.zeros_like(frame)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), 5)
        result = cv2.addWeighted(frame, 0.8, line_img, 1, 0)
        return result
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        processed_frame = process_frame(frame)
        cv2.imshow("Lane Detection", processed_frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

st.title("Vehicle Detection and Lane Line Detection")
uploaded_file = st.file_uploader("Upload a video file (MP4)", type=["mp4"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Vehicle Detection"):
            st.success("Vehicle detection started. Check the display window.")
            vehicle_count = process_vehicle_detection(video_path)
            st.success(f"Vehicle detection completed. Total Vehicles Counted: {vehicle_count}")
    with col2:
        if st.button("Lane Line Detection"):
            st.success("Lane line detection Started. Check the display window.")
            process_lane_detection(video_path)
            st.success("Lane line detection completed.")
