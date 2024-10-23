import os
import streamlit as st
import tempfile

from pymongo import MongoClient

from decouple import config

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


os.environ['OPENAI_API_KEY'] = config('OPENAI_API_KEY')
persist_directory = 'db'
OPENAI_MODEL_NAME = config('OPENAI_MODEL_NAME')

MONGO_URI = config('MONGO_URI')
client = MongoClient(MONGO_URI)
db = client['sample_mfix']
collection = db['movies']


collection.create_index([('title', 'text'), ('description', 'text')])


def process_pdf(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
        temp_file.write(file.read())
        temp_file_path = temp_file.name

    loader = PyPDFLoader(temp_file_path)
    docs = loader.load()

    os.remove(temp_file_path)

    text_spliter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=200,
    )

    chunks = text_spliter.split_documents(documents=docs)
    return chunks

def load_existing_vector_store():
    if os.path.exists(os.path.join(persist_directory)):
        vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=OpenAIEmbeddings(model="text-embedding-3-small", max_retries="2"),
        )
        return vector_store
    return None


def add_to_vector_store(chunks, vector_store=None):
    if vector_store:
        vector_store.add_documents(chunks)
    else:
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=OpenAIEmbeddings(model="text-embedding-3-small", max_retries="2"),
            persist_directory=persist_directory,
        )
    return vector_store

def ask_question(model, query, vector_store):
    llm = ChatOpenAI(model=model)
    retriever = vector_store.as_retriever()

    system_prompt = '''
    Use o contexto para responder as perguntas.
    Se não encontrar uma resposta no contexto,
    explique que não há informações disponíveis.
    Responda em formato de markdown e com visualizações
    elaboradas e interativas.
    Contexto: {context}
    '''

    max_context_messages = 5
    context_messages = st.session_state.messages[-max_context_messages:]

    messages = [('system', system_prompt)]
    for message in context_messages:
        if message.get('content') is not None and message.get('role') is not None:
            messages.append((message.get('role'), message.get('content')))
    
    # Adiciona a pergunta do usuário como 'human'
    messages.append(('human', query))

    if not messages:
        return "Erro: Nenhuma mensagem válida encontrada para gerar o prompt."

    try:
        # Consulta MongoDB antes de recorrer ao vector_store
        mongo_result = collection.find_one({"$text": {"$search": query}})
        
        if mongo_result:
            # Retornar as informações encontradas no MongoDB
            movie_info = f"Título: {mongo_result.get('title')}\nDescrição: {mongo_result.get('description')}"
            return movie_info
        else:
            # Se não encontrar no MongoDB, continuar com o vector_store
            prompt = ChatPromptTemplate.from_messages(messages)

            question_answer_chain = create_stuff_documents_chain(
                llm=llm,
                prompt=prompt,
            )

            chain = create_retrieval_chain(
                retriever=retriever,
                combine_docs_chain=question_answer_chain,
            )

            # Invocar o processo de cadeia
            response = chain.invoke({'input': query})
            answer = response.get('answer')

            if answer is None:
                answer = "Desculpe, não consegui gerar uma resposta adequada."

    except Exception as e:
        # Capture qualquer erro e forneça uma resposta padrão
        answer = f"Erro ao gerar a resposta: {e}"

    return answer





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
        type=['pdf'],
        accept_multiple_files=True,
    )

    if uploaded_files:
        with st.spinner('Processando documentos...'):
            all_chunks = []
            for uploaded_file in uploaded_files:
                chunks = process_pdf(file=uploaded_file)
                all_chunks.extend(chunks)

            vector_store = add_to_vector_store(
                chunks=all_chunks,
                vector_store=vector_store,
            )
    model_options = [
        'gpt-3.5-turbo',
        'gpt-4',
        'gpt-4-turbo',
        'gpt-4o-mini',
        'gpt-4o',
    ]
    selected_model = st.sidebar.selectbox(
        label='Selecione o modelo LLM',
        options=model_options,
        index=model_options.index(OPENAI_MODEL_NAME) if OPENAI_MODEL_NAME in model_options else 0
    )


if 'messages' not in st.session_state:
    st.session_state['messages'] = []

question = st.chat_input('Como posso ajudar?')

if vector_store and question:
    for message in st.session_state.messages:
        st.chat_message(message.get('role')).write(message.get('content'))

    st.chat_message('user').write(question)
    st.session_state.messages.append({'role': 'user', 'content': question})

    response = ask_question(
        model=selected_model,
        query=question,
        vector_store=vector_store,
    )

    st.chat_message('ai').write(response)
    st.session_state.messages.append({'role': 'ai', 'content': response})


def test_mongo_connection():
    try:
        # Testar leitura de documentos
        sample_document = collection.find_one()  # Buscar um documento da coleção
        if sample_document:
            st.success("Conexão com MongoDB bem-sucedida!")
            st.write("Documento de exemplo:", sample_document)
        else:
            st.warning("A conexão foi bem-sucedida, mas a coleção está vazia.")
    except Exception as e:
        st.error(f"Erro ao conectar com o MongoDB: {e}")

# Chamar a função de teste na interface Streamlit
st.sidebar.button("Testar Conexão com MongoDB", on_click=test_mongo_connection)



try:
    response = ask_question(
        model=selected_model,
        query=question,
        vector_store=vector_store,
    )
except Exception as e:
    st.error(f"Erro ao gerar a resposta: {e}")

