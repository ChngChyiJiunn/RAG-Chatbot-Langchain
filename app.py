__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import os
from model import LLM
from rag_utils import EmbeddingModel, ChromaDB, load_and_split
import hashlib
import torch
import time



FILES_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "files")
)

st.set_page_config(page_title="Streamlit Chatbot", page_icon="🤖")
st.title("Otter")

@st.cache_resource
def load_model():
    model = LLM(model_id="microsoft/phi-2",trust_remote_code=True,torch_dtype= torch.bfloat16, device_map = "auto")
    return model


@st.cache_resource
def load_embedding_model():
    embedding_model = EmbeddingModel(embed_id="BAAI/bge-small-en-v1.5", device="cpu")
    return embedding_model

# Set to store hashes of documents
@st.cache_resource
def create_document_set():
    document_hashes = set()
    return document_hashes

@st.cache_resource
def load_db():
    db = ChromaDB(embedding_model.embed_model)
    return db

model = load_model()
embedding_model = load_embedding_model()
document_hashes = create_document_set()



def save_file(uploaded_file):

    os.makedirs(FILES_DIR, exist_ok=True)
    file_path = os.path.join(FILES_DIR, uploaded_file.name)
    
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path




def add_document(docs):
    """
    Check single documents name
    """
    
    # Compute SHA256 hash of the document
    for doc in docs:
        doc_hash = hashlib.sha256(doc.name.encode()).hexdigest()
    
    # Check if hash is in the set
    if doc_hash in document_hashes:
        #print("Duplicate document detected!")
        return False
    else:
        document_hashes.add(doc_hash)
        #print("Document added successfully!")
        return True



def main():        
    with st.sidebar:
        max_new_tokens = st.number_input("max_new_tokens",128,4096,512)
        k = st.number_input("k",1,10,2)
        temperature = st.number_input("temperature",0.1,1.0,0.25)
        uploaded_files = st.file_uploader("Upload a .pdf File",type=["PDF","pdf"],accept_multiple_files=True)
        # if there is any uploaded file, save documents to db
        if uploaded_files != []:
            db = load_db()
            #db = ChromaDB(embedding_model.embed_model)
            if add_document(uploaded_files):
                file_paths = [save_file(uploaded_file) for uploaded_file in uploaded_files]
                docs = load_and_split(file_paths)
                db.save(docs)
                
            
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("How can I help you human today?"):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        

        # Display LLM assistant reponse in chat message container
        with st.chat_message("assistant"):
            context = (None if uploaded_files == [] else db.retrieve(prompt,
                                                                    k=k))
            
            #response = model.generate(prompt,context,max_new_tokens=max_new_tokens,temperature=temperature)
            #response = st.write_stream(model.stream_response(prompt,context,max_new_tokens=max_new_tokens,temperature=temperature))

            response_container = st.empty()  # Create an empty container for the streaming response
            full_response = ""  # Accumulate the full response here
            response = model.chain_response(prompt, context, max_new_tokens=max_new_tokens, temperature=temperature)
            for chunk in response.replace("$","\$") :
                full_response += chunk  # Accumulate chunks into the full response
                time.sleep(0.01)
                response_container.markdown(full_response + "▌")  # Display current progress with a cursor

            # Final update to remove the cursor
            response_container.markdown(full_response)

        st.session_state.messages.append({"role": "assistant", "content": full_response})

        

if __name__ == "__main__":
    main()