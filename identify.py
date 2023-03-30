import glob
import streamlit as st
from ultralytics import YOLO
import os.path
from PIL import Image


model = YOLO("D:/Object Detection YoloV8/runs/detect/train_25epochs/weights/best.pt")

def main():
    st.title('Object Identification')
    html_temp="""
                <div style="background-color:#3B7FA8">
                <h2 style="color:white;text-align:center;">Object Identification</h2>
                </div>
              """
    st.markdown(html_temp,unsafe_allow_html=True)
    st.text("You can use this model to identify objects from images,\nthe model can identify the following objects:")
    ul = """
            <table>
                <li>Person</li>
                <li>Chair</li>
                <li>Car</li>
                <li>Bottle</li>
                <li>Cat</li>
                <li>Bird</li>
                <li>Bottedplant</li>
                <li>Sheep</li>
                <li>Boat</li>
                <li>Aeroplane</li>
                <li>TVMonitor</li>
                <li>Sofa</li>
                <li>Bicycle</li>
                <li>Horse</li>
                <li>Dinning Table</li>
                <li>Motor Bike</li>
                <li>Cow</li>
                <li>Train</li>
                <li>Bus</li>
            </ul>
    """
    st.markdown(ul,unsafe_allow_html=True)
    
    image = st.file_uploader("Choose an image", type=['jpg','png','jpeg'])
    # video = st.file_uploader("Choose a video", type=['mp4'])

    if image is not None:
        st.image(image)
        # with open(image.name, "wb") as f:
        #     f.write(image.getbuffer())
        #     st.success("Image Saved")
        if st.button('identify'):
            model.predict(source="test/"+image.name, show=True, save=True, conf=0.5)
            folder_path = r'D:\\Object Detection YoloV8\\runs\detect'
            file_type = r'\*'
            files = glob.glob(folder_path + file_type + "\\")
            max_file = max(files, key=os.path.getctime)
            prediction = max_file + image.name
            identified = Image.open(prediction)
            st.image(identified)
    
    # if video is not None:
    #     st.video(video, start_time = 0)
    #     with open(video.name, "wb") as f:
    #         f.write(video.getbuffer())
    #         st.success("Video Saved")
    #     if st.button('identify'):
    #         model.predict(source=video.name, show=True, save=True, conf=0.5)
    #         folder_path = 'D:/Object Detection YoloV8/runs/detect'
    #         file_type = r'\*'
    #         files = glob.glob(folder_path + file_type + "/")
    #         max_file = max(files, key=os.path.getctime)
    #         prediction = max_file + video.name
    #         prediction = prediction.replace("\\", "/")
    #         video_file = open(prediction, 'rb')
    #         video_bytes = video_file.read()
    #         st.video(video_bytes, format="video/mp4" ,start_time = 0)
    #         st.success(prediction)

    
if __name__ =='__main__':
    main()
