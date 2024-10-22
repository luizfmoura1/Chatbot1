import os
import streamlit as st
import tempfile

from decouple import config

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings


os.environ['OPENAI_API_KEY'] = config('OPENAI_API_KEY')
persist_directory = 'db'


def process_pdf(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
        temp_file.write(file.read())
        temp_file_path = temp_file.name

    loader = PyPDFLoader(temp_file_path)
    docs = loader.load()

    os.remove(temp_file_path)

    text_spliter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=400,
    )
    chunks = text_spliter.split_documents(documents=docs)
    return chunks

def load_existing_vector_store():
    if os.path.exists(os.path.join(persist_directory)):
        vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=OpenAIEmbeddings(),
        )
        return vector_store
    return None


def add_to_vector_store(chunks, vector_store=None):
    if vector_store:
        vector_store.add_documents(chunks)
    else:
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=OpenAIEmbeddings(),
            persist_directory=persist_directory,
        )
    return vector_store
vector_store = load_existing_vector_store()

st.set_page_config(
    page_title='IA_teste1',
    page_icon='🤖',
    layout='wide'
)

st.header('Chatbot Oppem🤖')
st.write("Olá! Bem-vindo à página do chatbot.")

with st.sidebar:
    st.header('Upload de arquivos 📁')
    uploaded_files = st.file_uploader(
        label='Faça o upload de arquivos',
        type=['.xls, .xlsx, pdf'],
        accept_multiple_files=True,
    )

    if uploaded_files:
        with st.spinner('Processando documentos...'):
            all_chunks = []
            for uploaded_files in uploaded_files:
                chunks = process_pdf(file=uploaded_files)
                all_chunks.extend(chunks)
            vector_store = add_to_vector_store(
                chunks = all_chunks,
                vector_store=vector_store,
            )
    model_options = [
        'gpt-3.5-turbo',
        'gpt-4',
        'gpt-4-turbo',
        'gpt-4-mini',
        'gpt-4o',
    ]
    selected_model = st.sidebar.selectbox(
        label='Selecione o modelo LLM',
        options=model_options,
    )
question = st.chat_input('Como posso ajudar?')
st.chat_message('user').write(question)
