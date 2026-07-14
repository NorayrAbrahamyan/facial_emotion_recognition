import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import torch
import torchvision.transforms as transforms

# Ներմուծում ենք քո մոդելի դասը src պապկայի միջից
from src.model import EmotionCNN

# Էմոցիաների լեյբլները
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# 1. Բեռնում ենք քո մարզած մոդելը
@st.cache_resource
def load_emotion_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_weight_path = os.path.join(base_dir, "models", "emotion_cnn.pth")
    model = EmotionCNN(num_classes=7)
    model.load_state_dict(torch.load(model_weight_path, map_location=torch.device('cpu')))
    model.eval()
    return model

try:
    emotion_model = load_emotion_model()
except Exception as e:
    st.error("Մոդելի բեռնման սխալ: Համոզվեք, որ models/emotion_cnn.pth ֆայլը գոյություն ունի:")
    st.info(str(e))

# Տրանսֆորմացիաները
data_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  
])

# Ձախ մենյուում (Sidebar) ընտրում ենք ռեժիմը
st.sidebar.title("Կարգավորումներ")
mode = st.sidebar.selectbox("Ընտրեք ռեժիմը", ["Ներբեռնել նկար", "Կենդանի տեսախցիկ (Webcam)"])

# Դեմքի դետեկտորի ֆունկցիան
def detect_and_predict(img_bgr, detector, height, width):
    _, faces = detector.detect(img_bgr)
    
    if faces is not None:
        for face in faces:
            box = list(map(int, face[:4]))
            x, y, w, h = box
            
            x, y = max(0, x), max(0, y)
            w, h = min(width - x, w), min(height - y, h)
            
            # Էմոցիայի կանխատեսում
            face_crop = img_bgr[y:y+h, x:x+w]
            if face_crop.size > 0:
                face_crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                face_pil = Image.fromarray(face_crop_rgb)
                
                input_tensor = data_transforms(face_pil).unsqueeze(0)
                
                with torch.no_grad():
                    outputs = emotion_model(input_tensor)
                    _, predicted = torch.max(outputs, 1)
                    emotion_text = EMOTIONS[predicted.item()]
                
                # Կանաչ շրջանակ և տեքստ
                cv2.rectangle(img_bgr, (x, y), (x + w, y + h), (0, 255, 0), 3)
                cv2.putText(img_bgr, emotion_text, (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return img_bgr

# ----------------- ՌԵԺԻՄ 1. ՆԿԱՐԻ ՆԵՐԲԵՌՆՈՒՄ -----------------
if mode == "Ներբեռնել նկար":
    st.title("Դեմքերի և Էմոցիաների Ճանաչում (Նկար)")
    uploaded_file = st.file_uploader("Ընտրեք նկար...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        img_rgb = np.array(image)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        height, width, _ = img_bgr.shape

        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, "face_detection_yunet_2026may.onnx")   
        
        try:
            detector = cv2.FaceDetectorYN.create(model_path, "", (width, height), 0.9, 0.3)
            result_img_bgr = detect_and_predict(img_bgr, detector, height, width)
            
            result_img_rgb = cv2.cvtColor(result_img_bgr, cv2.COLOR_BGR2RGB)
            st.success("Մշակումն ավարտվեց:")
            st.image(result_img_rgb, use_container_width=True)
        except Exception as e:
            st.error(f"Սխալ: {e}")

# ----------------- ՌԵԺԻՄ 2. ԿԵՆԴԱՆԻ ՏԵՍԱԽՑԻԿ -----------------
elif mode == "Կենդանի տեսախցիկ (Webcam)":
    st.title("Իրական Ժամանակով Էմոցիաների Ճանաչում")
    st.write("Սեղմեք ներքևի կոճակը՝ տեսախցիկը միացնելու համար:")
    
    # Checkbox՝ տեսախցիկը սկսելու/կանգնեցնելու համար
    run_webcam = st.checkbox("Միացնել տեսախցիկը")
    
    # Ստեղծում ենք դատարկ պատուհան Streamlit-ում, որտեղ կթարմացվեն կադրերը
    frame_placeholder = st.empty()
    
    if run_webcam:
        # Միացնում ենք համակարգչի հիմնական տեսախցիկը (0)
        cap = cv2.VideoCapture(0)
        
        # Իմանում ենք տեսախցիկի կադրի չափսերը դետեկտորի համար
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, "face_detection_yunet_2026may.onnx")
        
        try:
            detector = cv2.FaceDetectorYN.create(model_path, "", (width, height), 0.9, 0.3)
            
            while run_webcam:
                ret, frame = cap.read()
                if not ret:
                    st.error("Չհաջողվեց կարդալ տեսախցիկի կադրը:")
                    break
                
                # Քանի որ տեսախցիկը հայելային է ցույց տալիս, շրջում ենք այն հորիզոնական
                frame = cv2.flip(frame, 1)
                
                # Մշակում ենք կադրը (գտնում ենք դեմքը և գուշակում էմոցիան)
                processed_frame = detect_and_predict(frame, detector, height, width)
                
                # Փոխում ենք RGB, որպեսզի Streamlit-ը ճիշտ ցուցադրի
                frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                
                # Թարմացնում ենք նկարը մեր ստեղծած դատարկ պատուհանում
                frame_placeholder.image(frame_rgb, use_container_width=True)
                
        except Exception as e:
            st.error(f"Տեսախցիկի աշխատանքի սխալ: {e}")
        finally:
            # Երբ checkbox-ն անջատենք, անպայման անջատում ենք տեսախցիկը
            cap.release()
            frame_placeholder.empty()