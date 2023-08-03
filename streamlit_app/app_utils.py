from PIL import Image
import streamlit as st
import numpy as np
import time
import cv2
import os

def get_dircontent(path):
    return sorted(os.listdir(path))

@st.cache(hash_funcs={cv2.VideoCapture:id})
def get_video(path):
    video = cv2.VideoCapture(path)
    return video

def get_frame_count(video):
    frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
    return int(frame_count)

def get_frame(video, selected_frame):
    video.set(cv2.CAP_PROP_POS_FRAMES, selected_frame)
    _, frame = video.read()
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

def save_frame(frame, placeholder): 
    dataset_path = 'local_dataset/'    
    txt_file = open(dataset_path + 'index.txt', 'r+')
        
    lines = txt_file.readlines()

    if len(lines) == 0: 
        last_value = 0

    else:
        last_value = lines[-1].strip()

    next_value = int(last_value) + 1

    image = Image.fromarray(frame)
    image.save(dataset_path + f'{next_value}.png')

    txt_file.write(f'\n{next_value}')
    txt_file.close()

    placeholder.success('Salvo!')
    time.sleep(0.5)
    placeholder.text('')


def update_slider(value):
    st.session_state["test_slider"] = value

def update_selected_file(direction, value, files):

    index = files.index(value)
    if direction == 'left' and index > 0:
        st.session_state["selected_file"] = files[index-1]
    if direction == 'right' and index < len(files)-1:
        st.session_state["selected_file"] = files[index+1]
        