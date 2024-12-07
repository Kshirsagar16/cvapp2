import cv2
import numpy as np
import streamlit as st
from camera_input_live import camera_input_live

"# Streamlit camera input live Demo"
"## Try holding a qr code in front of your webcam"

if "found_qr" not in st.session_state:
    st.session_state.found_qr = False

if "qr_code_image" not in st.session_state:
    st.session_state.qr_code_image = None

if not st.session_state["found_qr"]:
    image = camera_input_live()
else:
    image = st.session_state.qr_code_image

if image is not None:
    st.image(image)
    bytes_data = image.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    detector = cv2.QRCodeDetector()

    data, bbox, straight_qrcode = detector.detectAndDecode(cv2_img)

    if data:
        st.session_state["found_qr"] = True
        st.session_state["qr_code_image"] = image
        st.write("# Found QR code")
        st.write(data)
        with st.expander("Show details"):
            st.write("BBox:", bbox)
            st.write("Straight QR code:", straight_qrcode)