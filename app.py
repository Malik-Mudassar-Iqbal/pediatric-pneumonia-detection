import numpy as np
import cv2
import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers
from PIL import Image

MODEL_PATH = "mobilenet_improved.keras"
IMG_SIZE = 224


# ---- Custom layers: notebook jaisi hi hon, warna model load nahi hoga ----
class ChannelAttention(layers.Layer):
    def __init__(self, reduction=16, **kwargs):
        super().__init__(**kwargs)
        self.reduction = reduction

    def build(self, input_shape):
        channels = input_shape[-1]
        self.dense1 = layers.Dense(channels // self.reduction, activation="relu")
        self.dense2 = layers.Dense(channels, activation="sigmoid")
        self.global_avg = layers.GlobalAveragePooling2D()
        self.global_max = layers.GlobalMaxPooling2D()

    def call(self, x):
        avg_pool = self.global_avg(x)
        max_pool = self.global_max(x)
        avg_out = self.dense2(self.dense1(avg_pool))
        max_out = self.dense2(self.dense1(max_pool))
        attention = layers.Add()([avg_out, max_out])
        attention = layers.Activation("sigmoid")(attention)
        attention = layers.Reshape((1, 1, -1))(attention)
        return layers.Multiply()([x, attention])


class SpatialAttention(layers.Layer):
    def __init__(self, kernel_size=7, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size

    def build(self, input_shape):
        self.conv = layers.Conv2D(1, self.kernel_size, padding="same", activation="sigmoid")

    def call(self, x):
        avg_pool = tf.reduce_mean(x, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(x, axis=-1, keepdims=True)
        concat = layers.Concatenate()([avg_pool, max_pool])
        attention = self.conv(concat)
        return layers.Multiply()([x, attention])


class CBAM(layers.Layer):
    def __init__(self, reduction=16, kernel_size=7, **kwargs):
        super().__init__(**kwargs)
        self.channel_att = ChannelAttention(reduction)
        self.spatial_att = SpatialAttention(kernel_size)

    def call(self, x):
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x


@st.cache_resource
def load_model():
    # compile=False: focal loss sirf training mein chahiye, prediction mein nahi
    return tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={
            "CBAM": CBAM,
            "ChannelAttention": ChannelAttention,
            "SpatialAttention": SpatialAttention,
        },
        compile=False,
    )


def preprocess(pil_img):
    img = pil_img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0  # training jaisa rescale=1/255
    return arr


def grad_cam(model, arr):
    """Model ke andar MobileNetV2 ek nested layer hai (model.layers[1]).
    Hum uske output (feature maps) par gradient nikalte hain."""
    base = model.layers[1]
    head_layers = model.layers[2:]  # CBAM, GAP, BN, Dropout, Dense, ... , Dense(sigmoid)

    x = tf.convert_to_tensor(arr[np.newaxis])
    with tf.GradientTape() as tape:
        feats = base(x, training=False)
        tape.watch(feats)
        y = feats
        for layer in head_layers:
            y = layer(y, training=False)
        score = y[:, 0]  # pneumonia probability
    grads = tape.gradient(score, feats)

    weights = tf.reduce_mean(grads, axis=(1, 2))              # har channel ka importance
    cam = tf.reduce_sum(feats * weights[:, None, None, :], axis=-1)[0]
    cam = tf.nn.relu(cam).numpy()
    cam = cam / (cam.max() + 1e-8)
    return cam, float(score.numpy()[0])


def overlay(arr, cam):
    heat = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
    heat = cv2.applyColorMap(np.uint8(255 * heat), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    base = np.uint8(arr * 255)
    return cv2.addWeighted(base, 0.6, heat, 0.4, 0)


# ---------------- UI ----------------
st.set_page_config(page_title="Pneumonia Detector", page_icon="🫁")
st.title("🫁 Pneumonia X-ray Detector")
st.caption("MobileNetV2 + CBAM + Focal Loss  |  Screening tool hai, doctor ka replacement nahi.")

file = st.file_uploader("Chest X-ray upload karein", type=["jpg", "jpeg", "png"])

if file:
    model = load_model()
    pil_img = Image.open(file)
    arr = preprocess(pil_img)

    with st.spinner("Analyzing..."):
        cam, prob = grad_cam(model, arr)

    label = "PNEUMONIA" if prob > 0.5 else "NORMAL"
    conf = prob if prob > 0.5 else 1 - prob

    col1, col2 = st.columns(2)
    col1.image(np.uint8(arr * 255), caption="Uploaded X-ray", use_container_width=True)
    col2.image(overlay(arr, cam), caption="Grad-CAM heatmap", use_container_width=True)

    if label == "PNEUMONIA":
        st.error(f"Prediction: {label} ({conf:.0%} confidence)")
    else:
        st.success(f"Prediction: {label} ({conf:.0%} confidence)")
    st.write(f"Pneumonia probability: **{prob:.1%}**")
    st.info("Lal/peela hissa wo area hai jis par model ne sabse zyada focus kiya.")
else:
    st.write("Koi X-ray image upload karein.")
