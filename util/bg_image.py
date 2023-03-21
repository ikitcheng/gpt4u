import streamlit as st
import base64

@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    content: "";
    position: fixed;
    top: 0;
    left: 0;
    background-image: url("data:image/png;base64,%s");
    background-size: auto;
    background-repeat: no-repeat;
    background-position: center;
    z-index: -1;
    width: 100%%;
    height: 100%%;
    }
    
    .stApp::before {
    content: "";
    position: absolute;
    top: 0px;
    right: 0px;
    bottom: 0px;
    left: 0px;
    background-color: rgba(0,0,0,0.8);
    }

    </style>
    ''' % bin_str
    # the above line `background-color: rgba(0,0,0,0.85)` is to add a black transparent layer on top of the image
    
    st.markdown(page_bg_img, unsafe_allow_html=True)
    return