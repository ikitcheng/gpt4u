import streamlit as st
from util import bg_image
import config
import os
import hashlib
import shutil
from llama_index import (
    SimpleDirectoryReader,
    GPTListIndex,
    readers,
    GPTSimpleVectorIndex,
    LLMPredictor,
    PromptHelper,
)
from langchain import OpenAI


def construct_index(directory_path, outpath=None):
    # set maximum input size
    max_input_size = 4096
    # set number of output tokens
    num_outputs = 2000
    # set maximum chunk overlap
    max_chunk_overlap = 20
    # set chunk size limit
    chunk_size_limit = 600

    # define LLM
    llm_predictor = LLMPredictor(
        llm=OpenAI(
            temperature=0.5, model_name="text-davinci-003", max_tokens=num_outputs
        )
    )
    prompt_helper = PromptHelper(
        max_input_size,
        num_outputs,
        max_chunk_overlap,
        chunk_size_limit=chunk_size_limit,
    )

    documents = SimpleDirectoryReader(directory_path).load_data()

    index = GPTSimpleVectorIndex(
        documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper
    )

    if outpath:
        index.save_to_disk(outpath)

    return index


def ask_ai(path_to_index):
    st.write("Ask GPT-4-U your questions in the chat box below:")
    index = GPTSimpleVectorIndex.load_from_disk(path_to_index)
    cols = st.columns(2)

    query = cols[0].text_input("What do you want to ask? ")
    if query:
        response = index.query(query, response_mode="compact")
        cols[0].write(f"Response: <b>{response.response}</b>", unsafe_allow_html=True)

        return response


def save_uploadedfile(uploadedfile, outpath):
    with open(os.path.join(outpath, uploadedfile.name), "wb") as f:
        f.write(uploadedfile.getbuffer())

    st.success(f"Saved File: {uploadedfile.name} to {outpath}")


def update_session_state():
    shutil.rmtree(f"/app/data/{st.session_state['user_folder']}/", ignore_errors=True)
    del st.session_state['uploaded_files']
    del st.session_state['openai_api_key']
    del st.session_state['user_folder']

def main():
    # Create session state
    if 'uploaded_files' not in st.session_state:
        st.session_state['uploaded_files'] = []
    if 'openai_api_key' not in st.session_state:
        st.session_state['openai_api_key'] = None
    if 'user_folder' not in st.session_state:
        st.session_state['user_folder'] = None

    print(st.session_state)
    
    # Title
    st.title("GPT-4-U")

    # Columns
    cols = st.columns(2)
    
    # Enter API key
    with cols[0].container():
        with st.form("api_key_form", clear_on_submit=True):
            openai_api_key = st.text_input(
                "Enter your OpenAI API key here (New? Sign up [here](https://platform.openai.com/overview)):",
                type="password",
            )
            submitted = st.form_submit_button("Submit")
            if submitted:
                if openai_api_key:
                    os.environ["OPENAI_API_KEY"] = openai_api_key
                    st.session_state['openai_api_key'] = openai_api_key
                    openai_api_key = None
                    st.success("Your API key has been saved!", icon="✅")
                else:
                    st.error("Please enter your API key!", icon="❌")


    # Upload files
    with cols[0].container():
        with st.form("upload_files_form", clear_on_submit=True):
            uploaded_files = st.file_uploader(
                "Upload custom files (`.txt`, `.tex`, ...) or existing `index.json`",
                accept_multiple_files=True,
            )
            submitted = st.form_submit_button("Submit")

            if submitted:
                st.session_state['uploaded_files'] = uploaded_files
                uploaded_files = []

    if len(st.session_state['uploaded_files']) > 0 and st.session_state['openai_api_key']:

        # Create unique user folder
        st.session_state['user_folder'] = hashlib.md5(st.session_state['openai_api_key'].encode()).hexdigest()

        # Make unique folder for user
        if not os.path.isdir(f"/app/data/{st.session_state['user_folder']}/context_data/"):
            os.makedirs(f"/app/data/{st.session_state['user_folder']}/context_data/")


        # Loop through uploaded files
        for file in st.session_state['uploaded_files']:
            # save custom files to `context_data`
            if ".json" not in file.name and not os.path.isfile(
                f"/app/data/{st.session_state['user_folder']}/context_data/{file.name}"
            ):
                save_uploadedfile(file, outpath=f"/app/data/{st.session_state['user_folder']}/context_data/")
            elif ".json" in file.name and not os.path.isfile(f"/app/data/{st.session_state['user_folder']}/index.json"):
                save_uploadedfile(file, outpath=f"/app/data/{st.session_state['user_folder']}/")


        if not os.path.isfile(f"/app/data/{st.session_state['user_folder']}/index.json"):
            st.info("Creating `index.json`...", icon="ℹ️")
            llamaindex = construct_index(f"/app/data/{st.session_state['user_folder']}/context_data/", outpath=f"/app/{st.session_state['user_folder']}/index.json")
            st.success("Your data has been indexed successfully!", icon="✅")
            st.download_button(
                label="Download index as .json",
                data=llamaindex,
                file_name="index.json",
                mime="text/plain",
            )

        # start chat
        response = ask_ai(path_to_index=f"/app/data/{st.session_state['user_folder']}/index.json")

    # delete the saved file
    if cols[0].button('End Session', on_click=update_session_state) and 'uploaded_files' in st.session_state and 'openai_api_key' in st.session_state and 'user_folder' in st.session_state:
        st.success("Your session ended. All personal data has been removed.", icon="✅")


# setup page
st.set_page_config(
    page_title="GPT4U",
    page_icon=":shark:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# setup background image
bg_image.set_png_as_page_bg(f"{config.PATH_TO_IMAGES}/ChatGPTLogo.png")

if __name__ == "__main__":
    main()