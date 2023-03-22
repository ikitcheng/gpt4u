import streamlit as st
from util import bg_image
import config
import os
from llama_index import SimpleDirectoryReader, GPTListIndex, readers, GPTSimpleVectorIndex, LLMPredictor, PromptHelper
from langchain import OpenAI
import sys
import os

def construct_index(directory_path):
    # set maximum input size
    max_input_size = 4096
    # set number of output tokens
    num_outputs = 2000
    # set maximum chunk overlap
    max_chunk_overlap = 20
    # set chunk size limit
    chunk_size_limit = 600

    # define LLM
    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0.5, model_name="text-davinci-003", max_tokens=num_outputs))
    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)
 
    documents = SimpleDirectoryReader(directory_path).load_data()
    
    index = GPTSimpleVectorIndex(
        documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper
    )

    index.save_to_disk('/app/index.json')

    return index

def ask_ai():
    index = GPTSimpleVectorIndex.load_from_disk('/app/index.json')
    while True:
        cols = st.columns(2)
        query = cols[0].text_input("What do you want to ask? ")
        response = index.query(query, response_mode="compact")
        st.write(f"Response: <b>{response.response}</b>")
        
        if st.button("Stop"): 
            # delete the saved files
            files = os.listdir('/app/context_data/')
            for f in files:
                os.remove(f)

def save_uploadedfile(uploadedfile, outpath):
    with open(os.path.join(outpath, uploadedfile.name),"wb") as f:
        f.write(uploadedfile.getbuffer())
    return st.success(f"Saved File: {uploadedfile.name} to {outpath}")

def main():
    st.title("GPT-4-U")
    cols = st.columns(2)
    openai_api_key = cols[0].text_input("Enter your OpenAI API key here (New? Sign up [here](https://platform.openai.com/overview)):")

    if cols[0].button("Submit"):
        os.environ["OPENAI_API_KEY"] = openai_api_key

    uploaded_files = cols[0].file_uploader("Upload custom files (`.txt`, `.tex`, ...) or existing `index.json`",
                                            type=None, accept_multiple_files=True)

    if len(uploaded_files) > 0:
        for file in uploaded_files:
            # save custom files to `context_data`
            if '.json' not in file.name: 
                save_uploadedfile(file, outpath='/app/context_data/')
            elif '.json' in file.name:
                save_uploadedfile(file, outpath='/app/')

        
        if not os.path.isfile('/app/index.json'): 
            st.info('Creating `index.json`...', icon="ℹ️")
            llamaindex = construct_index("/app/context_data/")
            st.success("Your data has been indexed successfully!", icon="✅")
            st.download_button(
                label="Download index as .json",
                data=file,
                file_name='index.json',
                mime='text/plain',
                )

    if cols[0].button("Chat"):
        main2()

def main2():
    st.write("Ask GPT-4-U your questions in the chat box below:")
    ask_ai()

# setup page
st.set_page_config(page_title="GPT4U", page_icon=":shark:", layout="wide",
                initial_sidebar_state="expanded",
                )

# setup background image
bg_image.set_png_as_page_bg(
    f"{config.PATH_TO_IMAGES}/ChatGPTLogo.png"
)

if __name__ == "__main__":
    main()