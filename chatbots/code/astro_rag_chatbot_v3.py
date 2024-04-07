import streamlit as st
import os
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')
from langchain import hub
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain

# Load environment variables
load_dotenv()

# Setting necessary environment variables
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

# Setup for Langchain and OpenAI
model_name = "sentence-transformers/all-MiniLM-l6-v2"
embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={"device": "cpu"})

vectordb = Chroma(persist_directory='./arxiv_cosmo_chroma_db', embedding_function=embeddings)
retriever = vectordb.as_retriever()

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# Function to get response from the chatbot
def get_response(llm, retriever, query, chat_history):
    qa_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever)
    return qa_chain({"question": query, "chat_history": chat_history})

# Add background image
def add_bg_image():
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url("https://cdn.esahubble.org/archives/images/screen/potw1802a.jpg");
            background-size: cover;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Streamlit UI
if __name__ == '__main__':
    add_bg_image()
    # st.header("Chat About the Cosmos!")

    app_title = '<p style="font-family:sans-serif; color:White; \
    text-align: left; font-size: 32px;">⭐️Chat about the Cosmos!🌟</p>'

    st.markdown(app_title, unsafe_allow_html=True)

    query = st.text_input("Enter text here: ", placeholder="Type here...")

    if "user_history" not in st.session_state:
        st.session_state["user_history"] = []
    if "response_history" not in st.session_state:
        st.session_state["response_history"] = []
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    if query:
        with st.spinner("Thinking... "):
            output = get_response(llm=llm, retriever=retriever, query=query, chat_history=st.session_state["chat_history"])
            st.session_state["response_history"].append(output['answer'])
            st.session_state["user_history"].append(query)
            st.session_state["chat_history"].append((query, output['answer']))

    # Displaying the chat history in a cleaner way
    if st.session_state["response_history"]:
        st.write(r"$\textsf{\Large Chat History}$")
        for user_msg, bot_msg in zip(st.session_state["user_history"], st.session_state["response_history"]):
            st.write(f"User: {user_msg}")
            st.write(f"Chatbot: {bot_msg}")
            st.markdown("---")
