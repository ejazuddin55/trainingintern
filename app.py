import streamlit as st
import torch
import numpy as np
from PIL import Image
from train_vae_mnist import VAE, generate_images

# Configure the app
st.set_page_config(
    page_title="Handwritten Digit Generator",
    page_icon="✍️",
    layout="centered"
)

# App title and description
st.title("✍️ Handwritten Digit Generator")
st.markdown("""
Generate realistic handwritten digits using a Variational Autoencoder trained on MNIST.
Select a digit below and click **Generate** to create new samples!
""")

# Sidebar for additional info
with st.sidebar:
    st.header("About")
    st.markdown("""
    This app uses a Variational Autoencoder (VAE) to generate handwritten digits.
    The model was trained on the MNIST dataset of 28x28 pixel grayscale images.
    """)
    st.markdown("**Technical Details:**")
    st.markdown("- Latent space dimension: 20")
    st.markdown("- Trained for 20 epochs")
    st.markdown("- Uses binary cross-entropy + KL divergence loss")

# Load model with improved error handling
@st.cache_resource
def load_model():
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = VAE().to(device)
        model.load_state_dict(torch.load("vae_mnist.pth", map_location=device))
        model.eval()
        return model
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return None

model = load_model()

# Main interaction
st.subheader("Generate New Digits")
col1, col2 = st.columns([1, 3])

with col1:
    digit = st.selectbox(
        "Select digit:",
        options=list(range(10)),
        index=5,
        help="Choose which digit to generate (0-9)"
    )
    
    generate_btn = st.button(
        "✨ Generate Images",
        type="primary",
        help="Click to generate new samples"
    )

# Display results
if generate_btn and model:
    with st.spinner("Generating new digits..."):
        try:
            with torch.no_grad():
                images = generate_images(model, digit, num_images=5)
            
            st.success(f"Generated samples for digit {digit}:")
            
            # Display images in a grid
            cols = st.columns(5)
            for i, img in enumerate(images):
                img_pil = Image.fromarray((img * 255).astype(np.uint8))
                cols[i].image(
                    img_pil,
                    caption=f"Sample {i+1}",
                    use_column_width=True,
                    output_format="PNG"
                )
                
            # Add download button
            st.download_button(
                label="Download All Images",
                data=create_zip(images),  # You'd need to implement this
                file_name=f"generated_digits_{digit}.zip",
                mime="application/zip"
            )
            
        except Exception as e:
            st.error(f"Generation failed: {str(e)}")

elif generate_btn:
    st.warning("Model not loaded properly. Please check the error message above.")

# Footer
st.markdown("---")
st.caption("Note: Generated images are 28x28 pixels in grayscale. Quality may vary.")
