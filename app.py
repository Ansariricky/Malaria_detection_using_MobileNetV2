import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.cm as cm
from keras.models import load_model
import tensorflow as tf
import os

# ✅ Load only MobileNetV2
@st.cache_resource
def load_mobilenet():
    model_path = "MobileNetV2.h5"
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        st.stop()
    return load_model(model_path)

model = load_mobilenet()
last_conv_layer_name = "Conv_1"
input_size = (224, 224)

def get_gradcam_heatmap(img_array, model, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if isinstance(predictions, list):
            predictions = tf.convert_to_tensor(predictions)
        if hasattr(predictions, 'ndim') and predictions.ndim > 1:
            top_class_channel = predictions[:, 0]
        else:
            top_class_channel = predictions
    grads = tape.gradient(top_class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# ✅ Page config
st.set_page_config(page_title="Malaria Detection AI", layout="centered")
st.markdown(
    """
    <style>
    .stApp {
        background-color:#2E2E2E;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🧬 Malaria Detection Portal (MobileNetV2)")

st.markdown("""
### About This App
This AI-powered tool predicts whether a blood smear image is **Parasitized** or **Uninfected** using MobileNetV2.

> This is for **educational and research** use only. Not a medical diagnostic tool.
""")

# ✅ File uploader
uploaded_file = st.file_uploader("📤 Upload a Blood Smear Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image_resized = image.resize(input_size)
    img_array = np.expand_dims(np.array(image_resized) / 255.0, axis=0)

    st.subheader("Input Preview")
    st.image(image, use_container_width=True)

    prediction = model.predict(img_array)[0][0]
    label = "Parasitized" if prediction > 0.5 else "Uninfected"
    confidence_percent = prediction * 100 if label == "Parasitized" else (1 - prediction) * 100

    st.markdown(f"""
    ## 🩸 Prediction: **{label}**
    **Confidence Score:** `{confidence_percent:.2f}%`
    """)

    if confidence_percent >= 85:
        st.success("🟢 High confidence prediction.")
    elif 65 <= confidence_percent < 85:
        st.warning("🟡 Moderate confidence. Consider double-checking.")
    elif 50 <= confidence_percent < 65:
        st.error("🔴 Low confidence. Result may be unreliable.")
    else:
        st.error("⚠️ Very low confidence. Please consult a doctor.")

    # ✅ Grad-CAM
    st.markdown("---")
    st.subheader("Grad-CAM Visual Explanation")
    heatmap = get_gradcam_heatmap(img_array, model, last_conv_layer_name)
    heatmap = np.uint8(255 * heatmap)
    heatmap_img = Image.fromarray(heatmap).resize(image.size)

    heatmap_colored = cm.jet(np.array(heatmap_img) / 255.0)[:, :, :3]
    heatmap_colored = np.uint8(heatmap_colored * 255)
    heatmap_colored_img = Image.fromarray(heatmap_colored).convert("RGBA")
    heatmap_colored_img.putalpha(128)

    overlay = Image.alpha_composite(image.convert("RGBA"), heatmap_colored_img)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Original Image**")
        st.image(image, use_container_width=True)
    with col2:
        st.markdown("**Model's Focus Area (Grad-CAM)**")
        st.image(overlay, use_container_width=True)

    st.markdown("""
<div style='
    padding: 12px; 
    background-color: black; 
    border-left: 4px solid #007acc; 
    border-radius: 8px;
    font-size: 15px;
    line-height: 1.6;
'>
<b>📊 Grad-CAM Interpretation:</b><br>
This heatmap shows which parts of the image the AI model focused on when making its decision.

<ul style="margin-top: 8px; margin-left: 20px;">
  <li><span style='display:inline-block; width:14px; height:14px; background-color:red; border-radius:3px; margin-right:6px;'></span> High importance (model focused here)</li>
  <li><span style='display:inline-block; width:14px; height:14px; background-color:yellow; border-radius:3px; margin-right:6px;'></span> Moderate importance</li>
  <li><span style='display:inline-block; width:14px; height:14px; background-color:blue; border-radius:3px; margin-right:6px;'></span> Low or no focus</li>
</ul>

</div>
""", unsafe_allow_html=True)

with st.expander("🔍 How It Works"):
    st.markdown("""
    - Trained on the NIH Malaria Dataset  
    - Uses transfer learning with **MobileNetV2**  
    - Predicts: **Parasitized** or **Uninfected**  
    - Includes Grad-CAM for transparency
    """)

st.markdown("---")
st.markdown("""
**📚 References**  
- [NIH Malaria Dataset](https://lhncbc.nlm.nih.gov/LHC-research/LHC-projects/image-processing/malaria-datasheet.html)  
- MobileNetV2 architecture paper  
- Grad-CAM: Selvaraju et al., 2017

Built by: Rekibuddin Ansari — Assam Engineering College, Guwahati  
*Intent : Educational and research purpose only*
""")
