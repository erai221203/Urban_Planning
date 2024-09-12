import os
import streamlit as st
from PIL import Image
from io import BytesIO
import zipfile
import base64
import requests  

api_key = os.getenv("GROQ_API_KEY=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")

def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
load_css("styles.css")
def query_llm(prompt):
    api_endpoint = 'https://api.groq.com/openai/v1/chat/completions'
    api_key = "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"
    
    headers = {"Authorization": f"Bearer {api_key}","Content-Type": "application/json"}
    payload = {"model": "llama3-8b-8192",  #model
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1000,
                "temperature": 0.7}
    response = requests.post(api_endpoint, json=payload, headers=headers)
    if response.status_code == 200:
        return response.json().get("choices", [])[0]["message"]["content"]
    else:
        return f"Error: {response.status_code} - {response.text}"

TEMP_DIR = 'temp_files'#Temp dir
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)

st.title('🌆 Urban Planning Tool')#Title
st.subheader('🚀 Enhance Your City Planning')


col1, col2 = st.columns(2)#Columnsa

with col1:
    uploaded_images = st.file_uploader("Upload Image Files", accept_multiple_files=True, type=['png', 'jpg', 'jpeg','svg'])
with col2:
    uploaded_metadata = st.file_uploader("Upload Metadata Files", accept_multiple_files=True, type=['txt', 'csv', 'docx', 'pdf'])
def display_compact_grid(image_paths, max_images=2):
    displayed_images = image_paths[:max_images]
    remaining_images = len(image_paths) - max_images
    cols = st.columns(3)
    for i, image_path in enumerate(image_paths):
                with cols[i % 3]:
                    st.image(image_path, use_column_width=True)

    st.markdown('<div class="compact-grid">', unsafe_allow_html=True)
    for image_path in displayed_images:
        if image_path.endswith(('png', 'jpg', 'jpeg','svg')):  
            image = Image.open(image_path)
            st.markdown(
                f"""
                <div class="image-folder">
                    <img src="data:image/png;base64,{image_to_base64(image)}" alt="Image" />
                </div>
                """,
                unsafe_allow_html=True
            )
    if remaining_images > 0:
        st.markdown(f"<h4>+{remaining_images} more images</h4>", unsafe_allow_html=True)
        if st.button(f"Show all {len(image_paths)} images"):
            display_all_images(image_paths)
    st.markdown('</div>', unsafe_allow_html=True)

# Display all images
def display_all_images(image_paths):
    cols = st.columns(3)
    for i, image_path in enumerate(image_paths):
        with cols[i % 3]:
            image = Image.open(image_path)
            st.image(image, caption=os.path.basename(image_path), use_column_width=True)

# Convert image to base64
def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

# Display uploaded metadata
def display_metadata_files(metadata_paths):
    st.markdown("<h4>Uploaded Metadata Files</h4>", unsafe_allow_html=True)
    for file_path in metadata_paths:
        if file_path.endswith(('csv', 'txt', 'docx', 'pdf')):
            st.write(f"📄 {os.path.basename(file_path)}")

#Zip 
def zip_files(files):
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in files:
            zf.write(file_path, os.path.basename(file_path))
    zip_buffer.seek(0)
    return zip_buffer

image_paths = []
metadata_paths = []

# Handle uploaded files
if uploaded_images or uploaded_metadata:
    # Process uploaded images
    if uploaded_images:
        for file in uploaded_images:
            file_path = os.path.join(TEMP_DIR, file.name)
            with open(file_path, 'wb') as f:
                f.write(file.getvalue())
            image_paths.append(file_path)
    
    # Process uploaded metadata
    if uploaded_metadata:
        for file in uploaded_metadata:
            file_path = os.path.join(TEMP_DIR, file.name)
            with open(file_path, 'wb') as f:
                f.write(file.getvalue())
            metadata_paths.append(file_path)
    
    # Display uploaded images
    if image_paths:
        st.markdown("<h4>Uploaded Images</h4>", unsafe_allow_html=True)
        display_compact_grid(image_paths)
    
    # Display uploaded metadata files
    if metadata_paths:
        display_metadata_files(metadata_paths)

    # Display the training button only if both images and metadata are uploaded
    if uploaded_images and uploaded_metadata:
        if st.button('🛠️ Train'):
            st.write('Model training completed!')
            st.markdown("<h3>Processed Data</h3>", unsafe_allow_html=True)
            cols = st.columns(3)
            for i, image_path in enumerate(image_paths):
                with cols[i % 3]:
                    st.image(image_path, use_column_width=True)
                    st.download_button(
                        label=f'Download Image {i+1}',
                        data=open(image_path, 'rb').read(),
                        file_name=os.path.basename(image_path),
                        mime='image/png'
                    )

            # Zip and download processed images
            if image_paths:
                zip_buffer = zip_files(image_paths)
                st.download_button(
                    label='Download All Images as Zip',
                    data=zip_buffer,
                    file_name='processed_images.zip',
                    mime='application/zip'
                )

# Display the prompt text area after handling the files
st.markdown("<h4>Requirements</h4>", unsafe_allow_html=True)
prompt = st.text_area("Enter your requirements:", height=100)  # Adjust the height as needed

# Display LLM response
if prompt:
    st.markdown("<h4></h4>", unsafe_allow_html=True)
    response = query_llm(prompt)
    st.write(response=False)

# Clean up files after processing
all_files = image_paths + metadata_paths
for file_path in all_files:
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
        except Exception as e:
            st.error(f"Error removing file {file_path}: {e}")