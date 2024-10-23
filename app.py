import os
import streamlit as st
from pymongo import MongoClient
from decouple import config

# Configurações iniciais
os.environ['OPENAI_API_KEY'] = config('OPENAI_API_KEY')
OPENAI_MODEL_NAME = config('OPENAI_MODEL_NAME')

# Conexão com MongoDB
MONGO_URI = config('MONGO_URI')
client = MongoClient(MONGO_URI)
db = client['sample_mflix']
collection = db['movies']

def ask_question(query):
    try:
        # Realiza a consulta no MongoDB usando o índice existente (busca por título, elenco, descrição)
        mongo_result = collection.find_one({"$text": {"$search": query}})
        
        # Se encontrar um resultado, formatamos a resposta com os dados encontrados
        if mongo_result:
            title = mongo_result.get('title', 'Título não disponível')
            plot = mongo_result.get('plot', 'Descrição não disponível')
            genres = ', '.join(mongo_result.get('genres', []))
            cast = ', '.join(mongo_result.get('cast', []))
            imdb_rating = mongo_result.get('imdb', {}).get('rating', 'Sem nota')

            response = f"**Título**: {title}\n\n"
            response += f"**Descrição**: {plot}\n\n"
            response += f"**Gêneros**: {genres}\n\n"
            response += f"**Elenco**: {cast}\n\n"
            response += f"**Nota IMDb**: {imdb_rating}\n\n"
            
            return response
        else:
            return "Não encontrei informações relevantes no banco de dados."

    except Exception as e:
        return f"Erro ao realizar a consulta: {e}"

# Interface do Streamlit
st.set_page_config(
    page_title='IA_teste1',
    page_icon='🤖',
    layout='wide'
)

st.header('Chatbot Oppem🤖')
st.write("Olá! Bem-vindo à página do chatbot. Pergunte algo relacionado a filmes!")

# Seleção de modelo, se necessário
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

# Inicializar o armazenamento de mensagens na sessão
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

# Caixa de input do usuário
question = st.chat_input('Pergunte sobre filmes')

if question:
    # Exibir a pergunta do usuário
    st.chat_message('user').write(question)
    st.session_state.messages.append({'role': 'user', 'content': question})

    # Processar a pergunta e exibir a resposta do chatbot
    response = ask_question(query=question)
    
    st.chat_message('ai').write(response)
    st.session_state.messages.append({'role': 'ai', 'content': response})

# Função de teste de conexão ao MongoDB
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

# Botão para testar a conexão com o MongoDB
st.sidebar.button("Testar Conexão com MongoDB", on_click=test_mongo_connection)
