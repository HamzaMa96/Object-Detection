import glob
import streamlit as st
from ultralytics import YOLO
import os.path
from PIL import Image
import os
os.add_dll_directory(r"C:\Windows\System32\vlc-3.0.18")
import vlc


model = YOLO("D:/Object Detection YoloV8/runs/detect/train_25epochs/weights/best.pt")

def main():
    # st.title('Object Identification')
    html_temp="""
                <div style="background-color:#3B7FA8">
                <h2 style="color:white;text-align:center;">Object Identification</h2>
                </div>
              """
    st.markdown(html_temp,unsafe_allow_html=True)
    st.text("You can use this model to identify objects from images or videos,\nthe model can identify the following objects:")
    ul = """
        <table style="border:0;text-align:center;">
            <tr>
                <td>Person</td>
                <td>Chair</td>
                <td>Car</td>
                <td>Dog</td>
                <td>Bottle</td>
            </tr>
            <tr>
                <td>Cat</td>
                <td>Bird</td>
                <td>Bottedplant</td>
                <td>Sheep</td>
                <td>Boat</td>
            </tr>
            <tr>
                <td>Aeroplane</td>
                <td>TVMonitor</td>
                <td>Sofa</td>
                <td>Bicycle</td>
                <td>Horse</td>
            </tr>
            <tr>
                <td>Dinning Table</td>
                <td>Motor Bike</td>
                <td>Cow</td>
                <td>Train</td>
                <td>Bus</td>
            </tr>
        </table>
    """
    st.markdown(ul,unsafe_allow_html=True)
    
    image = st.file_uploader("Choose an image", type=['jpg','png','jpeg'])
    video = st.file_uploader("Choose a video", type=['mp4'])

    if image is not None:
        st.image(image)
        with open(image.name, "wb") as f:
            f.write(image.getbuffer())
            st.success("Image Saved")
        if st.button('identify'):
            model.predict(source="test"+image.name, show=True, save=True, conf=0.5)
            folder_path = r'D:\\Object Detection YoloV8\\runs\detect'
            file_type = r'\*'
            files = glob.glob(folder_path + file_type + "\\")
            max_file = max(files, key=os.path.getctime)
            prediction = max_file + image.name
            identified = Image.open(prediction)
            st.image(identified)
    
    if video is not None:
        st.video(video, start_time = 0)
        with open(video.name, "wb") as f:
            f.write(video.getbuffer())
            st.success("Video Saved")
        if st.button('identify'):
            model.predict(source=video.name, show=True, save=True, conf=0.5)
            folder_path = r'D:\Object Detection YoloV8\runs\detect'
            file_type = r'\*'
            files = glob.glob(folder_path + file_type + "\\")
            max_file = max(files, key=os.path.getctime)
            prediction = max_file + video
            prediction = prediction.replace("\\", "/")
            prediction = prediction.replace("test/", '')
            video = vlc.MediaPlayer(prediction)
            video.play()
            st.success(prediction)

    
if __name__ =='__main__':
    main()
