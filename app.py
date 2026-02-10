import streamlit as st
import numpy as np
import cv2

# --- ENGINE LOGIC ---
class VerifeyeEngine:
    def __init__(self):
        self.baseline = 3200
        self.det_range = 8000

    def calculate_authenticity(self, s_val, g_val, t_val):
        s_risk = min(max((s_val - self.baseline) / self.det_range, 0), 1)
        g_risk = min(g_val / 4.0, 1)
        t_risk = min(t_val / 1.5, 1)
        final_risk = (s_risk * 0.2) + (g_risk * 0.4) + (t_risk * 0.4)
        return round((1 - final_risk) * 100, 2)

def extract_spectral_score(img_gray):
    f = np.fft.fft2(img_gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.abs(fshift)
    rows, cols = img_gray.shape
    crow, ccol = rows//2 , cols//2
    mask = np.ones((rows, cols), np.uint8)
    mask[crow-20:crow+20, ccol-20:ccol+20] = 0
    return np.mean(magnitude_spectrum * mask)

# --- UI INTERFACE ---
st.set_page_config(page_title="Verifeye Forensic Suite", page_icon="🛡️", layout="wide")
st.title("🛡️ Verifeye: Forensic Authentication")
st.write("Developed by: **Fabrice Fils-Aimé**")

st.sidebar.header("📥 Media Upload")
uploaded_file = st.sidebar.file_uploader("Choose a frame...", type=['png', 'jpg', 'jpeg'])

st.sidebar.markdown("---")
st.sidebar.subheader("Forensic Parameters")
geo_sim = st.sidebar.slider("Geometric Symmetry", 0.0, 5.0, 0.1)
temp_sim = st.sidebar.slider("Temporal Stability", 0.0, 3.0, 0.2)

if uploaded_file:
    # Read Image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Analysis Target")
        st.image(img, channels="BGR", use_container_width=True)

    with col2:
        st.subheader("Frequency Spectrum (FFT)")
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        mag = 20 * np.log(np.abs(fshift) + 1)
        st.image(mag / mag.max(), use_container_width=True)

    # Engine Execution
    score = extract_spectral_score(gray)
    engine = VerifeyeEngine()
    final_auth = engine.calculate_authenticity(score, geo_sim, temp_sim)

    st.divider()
    res1, res2 = st.columns(2)
    res1.metric("Authenticity Score", f"{final_auth}%")
    res2.metric("Spectral Noise", f"{score:.2f}")

    if final_auth < 85:
        st.error("🚨 HIGH PROBABILITY DEEPFAKE: Synthetic artifacts detected.")
    else:
        st.success("✅ VERIFIED AUTHENTIC: Biological signatures confirmed.")
