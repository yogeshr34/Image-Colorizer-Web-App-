import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import numpy as np
from torch import nn
import os

# Model Architecture
class ColorAutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = nn.Conv2d(1, 64, 3, stride=2)
        self.down2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.down3 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.down4 = nn.Conv2d(256, 512, 3, stride=2, padding=1)

        self.up1 = nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1)
        self.up2 = nn.ConvTranspose2d(512, 128, 3, stride=2, padding=1)
        self.up3 = nn.ConvTranspose2d(256, 64, 3, stride=2, padding=1, output_padding=1)
        self.up4 = nn.ConvTranspose2d(128, 3, 3, stride=2, output_padding=1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        d1 = self.relu(self.down1(x))
        d2 = self.relu(self.down2(d1))
        d3 = self.relu(self.down3(d2))
        d4 = self.relu(self.down4(d3))
        
        u1 = self.relu(self.up1(d4))
        u2 = self.relu(self.up2(torch.cat((u1, d3), dim=1)))
        u3 = self.relu(self.up3(torch.cat((u2, d2), dim=1)))
        u4 = self.sigmoid(self.up4(torch.cat((u3, d1), dim=1)))
        
        return u4

# Image preprocessing
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((150, 150)),
        transforms.ToTensor()
    ])
    return transform(image)

# Colorize function
def colorize_image(model, image):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    with torch.no_grad():
        input_tensor = preprocess_image(image).unsqueeze(0).to(device)
        output = model(input_tensor)
        output = output.squeeze(0).cpu().permute(1, 2, 0).numpy()
        output = (output * 255).clip(0, 255).astype(np.uint8)
        return output

# Model Loading Function
@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ColorAutoEncoder().to(device)
    try:
        model.load_state_dict(torch.load("colorization_model.pth", map_location=device))
        st.success("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

# Streamlit App
def main():
    st.set_page_config(
        page_title="AI Image Colorization",
        page_icon="🎨",
        layout="wide",
    )

    # Header Section
    st.markdown("""
        <style>
            .header {
                font-size: 2.5em; 
                font-weight: 600; 
                color: white;
                text-align: center;
                margin-bottom: 20px;
            }
            .subheader {
                text-align: center;
                color: #64748b;
                margin-top: -10px;
            }
            .center-btn {
                display: flex;
                justify-content: center;
                margin: 20px 0;
            }
        </style>
        <div class="header">🎨 AI Image Colorization Studio</div>
        <div class="subheader">Bring your black & white images to life with vibrant colors!</div>
    """, unsafe_allow_html=True)

    model = load_model()

    # Sidebar with Instructions
    with st.sidebar:
        st.title("🛠️ How to Use:")
        st.markdown(
            """
            1. 📤 Upload a black & white image.
            2. 🎨 Click **Colorize Image**.
            3. 💾 Download the colorized image.
            """,
            unsafe_allow_html=True
        )
        st.info("Ensure the uploaded image is a black & white photograph for best results.")

    # File Uploader Section
    uploaded_file = st.file_uploader("Upload your B&W image", type=["jpg", "jpeg", "png"])
    
    # Centered Colorize Button
    st.markdown("<div class='center-btn'>", unsafe_allow_html=True)
    colorize_button = st.button("🎨 Colorize Image", type="primary")
    st.markdown("</div>", unsafe_allow_html=True)
    
    if model is None:
        st.error("The model is not loaded properly. Please check the model file.")
        return

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        # Display Original and Colorized Images in the Same Section
        st.markdown("### 🖼️ Original & Colorized Image")
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption="Original Image", use_column_width=True)
        
        if colorize_button:
            with st.spinner("Colorizing your image... Please wait ⏳"):
                colorized_image = colorize_image(model, image)
                with col2:
                    st.image(colorized_image, caption="Colorized Image", use_column_width=True)
                
                # Download Button
                buf = io.BytesIO()
                colorized_pil = Image.fromarray(colorized_image)
                colorized_pil.save(buf, format="PNG")
                st.download_button(
                    label="💾 Download Colorized Image",
                    data=buf.getvalue(),
                    file_name="colorized_image.png",
                    mime="image/png"
                )
    else:
        st.info("📂 Upload an image to get started!")

    # Footer Section
    st.markdown("""
        <div style='text-align: center; margin-top: 30px; color: #9ca3af;'>
            <small>Built with ❤️ using Streamlit</small>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
