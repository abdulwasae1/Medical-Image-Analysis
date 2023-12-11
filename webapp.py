import os
import cv2
import cv2 as cv
import numpy as np
import streamlit as st
import onnxruntime as ort
import tensorflow as tf
from matplotlib.colors import TABLEAU_COLORS
from pathlib import Path


ALLOWED_EXTENSIONS = {"txt", "pdf", "png", "jpg", "jpeg", "gif"}
parent_root = Path(__file__).parent.parent.absolute().__str__() # os.path.dirname(os.path.abspath(__file__))
h, w = 640, 640
model_onnx_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yolov7-p6-bonefracture.onnx")
device = "cuda"


brain_tumor_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "brain_tumor_detector.h5")
brain_tumor_model = tf.keras.models.load_model(brain_tumor_model_path)



def color_list():
    # Return first 10 plt colors as (r,g,b) https://stackoverflow.com/questions/51350872/python-from-color-name-to-rgb
    def hex2rgb(h):
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))

    return [hex2rgb(h) for h in TABLEAU_COLORS.values()]

colors = color_list()

def xyxy2xywhn(bbox, H, W):

    x1, y1, x2, y2 = bbox

    return [0.5*(x1+x2)/W, 0.5*(y1+y2)/H, (x2-x1)/W, (y2-y1)/H]

def xywhn2xyxy(bbox, H, W):

    x, y, w, h = bbox

    return [(x-w/2)*W, (y-h/2)*H, (x+w/2)*W, (y+h/2)*H]

def load_img(uploaded_file):
    """ Load image from bytes to numpy
    """

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    return opencv_image[..., ::-1]
    
    
    
    
class DisplayTumor:
    curImg = 0
    Img = 0

    def readImage(self, img):
        self.Img = np.array(img)
        self.curImg = np.array(img)
        gray = cv.cvtColor(np.array(img), cv.COLOR_BGR2GRAY)
        self.ret, self.thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

    def getImage(self):
        return self.curImg

    # noise removal
    def removeNoise(self):
        self.kernel = np.ones((3, 3), np.uint8)
        opening = cv.morphologyEx(self.thresh, cv.MORPH_OPEN, self.kernel, iterations=2)
        self.curImg = opening

    def displayTumor(self):
        # sure background area
        sure_bg = cv.dilate(self.curImg, self.kernel, iterations=3)

        # Finding sure foreground area
        dist_transform = cv.distanceTransform(self.curImg, cv.DIST_L2, 5)
        ret, sure_fg = cv.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

        # Find unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv.subtract(sure_bg, sure_fg)

        # Marker labelling
        ret, markers = cv.connectedComponents(sure_fg)

        # Add one to all labels so that sure background is not 0, but 1
        markers = markers + 1

        # Now mark the region of unknown with zero
        markers[unknown == 255] = 0
        markers = cv.watershed(self.Img, markers)
        self.Img[markers == -1] = [255, 0, 0]

        tumorImage = cv.cvtColor(self.Img, cv.COLOR_HSV2BGR)
        self.curImg = tumorImage
    
    
    
    
    
def preprocess_brain_tumor_img(img):
    img = cv2.resize(img, (240, 240))  # Change from (224, 224) to (240, 240)
    img = img / 255.0
    return np.expand_dims(img, axis=0)

def detect_and_display_brain_tumor(img):
    # Process the image for brain tumor detection
    preprocessed_img = preprocess_brain_tumor_img(img)
    prediction = brain_tumor_model.predict(preprocessed_img)

    # Initialize the DisplayTumor class and process the image
    display_tumor = DisplayTumor()
    display_tumor.readImage(img)
    display_tumor.removeNoise()
    display_tumor.displayTumor()
    tumor_img = display_tumor.getImage()

    # Check for tumor and return processed image
    tumor_detected = "Brain Tumor Detected" if np.argmax(prediction) == 1 else "Brain Tumor Region"
    return tumor_img, tumor_detected
    

def preproc(img):
    """ Image preprocessing
    """
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32).transpose(2, 0, 1)/255
    return np.expand_dims(img, axis=0)

def model_inference(model_path, image_np, device="cpu"):

    providers = ["CUDAExecutionProvider"] if device=="cuda" else ["CPUExecutionProvider"]
    session = ort.InferenceSession(model_path, providers=providers)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    output = session.run([output_name], {input_name: image_np})

    return output[0][:, :6]


def post_process(img, output, score_threshold=0.3):
    """
    Draw bounding boxes on the input image. Dump boxes in a txt file.
    """
    # assert format == "xyxy" or format == "xywh"

    det_bboxes, det_scores, det_labels = output[:, 0:4], output[:, 4], output[:, 5]
    id2names = {
        0: "boneanomaly", 1: "bonelesion", 2: "foreignbody", 
        3: "fracture", 4: "metal", 5: "periostealreaction", 
        6: "pronatorsign", 7:"softtissue", 8:"text"
    }

    if isinstance(img, str):
        img = cv2.imread(img)
    
    img = img.astype(np.uint8)
    H, W = img.shape[:2]
    label_txt = ""

    for idx in range(len(det_bboxes)):
        if det_scores[idx]>score_threshold:
            bbox = det_bboxes[idx]
            label = det_labels[idx]
            
            bbox = xyxy2xywhn(bbox, h, w)
            label_txt += f"{int(label)} {det_scores[idx]:.5f} {bbox[0]:.5f} {bbox[1]:.5f} {bbox[2]:.5f} {bbox[3]:.5f}\n"

            bbox = xywhn2xyxy(bbox, H, W)
            bbox_int = [int(x) for x in bbox]
            x1, y1, x2, y2 = bbox_int
            color_map = colors[int(label)]
            txt = f"{id2names[label]} {det_scores[idx]:.2f}"
            (text_width, text_height), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            
            cv2.rectangle(img, (x1, y1), (x2, y2), color_map, 2)
            cv2.rectangle(img, (x1-2, y1-text_height-10), (x1 + text_width+2, y1), color_map, -1)
            cv2.putText(img, txt, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return img, label_txt


if __name__ == "__main__":
    st.title("Medical Image Analysis")

    # Adding Google AdSense code snippet
    st.markdown(
        '<script async src="https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js?client=ca-pub-6840837431951843" crossorigin="anonymous"></script>',
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg", "gif"])

    if uploaded_file is not None:
        conf_thres = st.slider("Object confidence threshold", 0.2, 1., step=0.05)

        # load and display original image
        img = load_img(uploaded_file)

        # Bone fracture detection inference
        img_pp = preproc(img)
        out = model_inference(model_onnx_path, img_pp, device)
        out_img, out_txt = post_process(img, out, conf_thres)
        st.image(out_img, caption="Bone Fracture Detection", channels="RGB")

        # Brain tumor detection and display
        tumor_img, brain_tumor_result = detect_and_display_brain_tumor(img)
        st.write(brain_tumor_result)
        st.image(tumor_img, caption="Brain Tumor Region", channels="RGB")

        # Download buttons
        col1, col2 = st.columns(2)
        col1.download_button(
            label="Download Bone Fracture Prediction",
            data=cv2.imencode(".png", out_img[..., ::-1])[1].tobytes(),
            file_name=uploaded_file.name,
            mime="image/png"
        )
        col2.download_button(
            label="Download Detections",
            data=out_txt,
            file_name=uploaded_file.name[:-4] + ".txt",
            mime="text/plain"
        )
