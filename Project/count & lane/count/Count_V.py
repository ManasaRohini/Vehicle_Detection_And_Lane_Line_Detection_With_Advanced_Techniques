# to run the code command is
# streamlit run Count_V.py

# to stop command is
# ctrl+c

import cv2
import numpy as np
import streamlit as st
import tempfile

# Streamlit UI
st.title("Vehicle Detection and Counting")

# Upload video file
uploaded_file = st.file_uploader("Upload a video file (MP4)", type=["mp4"])

def get_centroid(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    return x + x1, y + y1

def process_video(video_path):
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

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name
    
    if st.button("Detect Vehicles"):
        process_video(video_path)
        st.success("Vehicle detection completed. Check the display window.")