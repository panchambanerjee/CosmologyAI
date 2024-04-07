import streamlit as st
import os
from dotenv import load_dotenv, find_dotenv
import warnings
warnings.filterwarnings('ignore')
from langchain import hub

from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


_ = load_dotenv(find_dotenv())

os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'

os.environ['LANGCHAIN_API_KEY'] = os.environ['LANGCHAIN_API_KEY']
os.environ['OPENAI_API_KEY'] = os.environ['OPENAI_API_KEY']

model_name = "sentence-transformers/all-MiniLM-l6-v2"
model_kwargs = {"device": "cpu"} # Since we are running on local machine, we will use CPU

embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

vectordb = Chroma(persist_directory='./arxiv_cosmo_chroma_db', embedding_function=embeddings)
retriever = vectordb.as_retriever()

prompt = hub.pull("rlm/rag-prompt")
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

def get_response(llm, retriever, query, chat_history):

    qa_chain = ConversationalRetrievalChain.from_llm(
       llm=llm, 
       retriever=retriever
    )

    # Test qa_chain.invoke({"question": "What is a Galaxy Cluster?", "chat_history": ""})

    return qa_chain({"question":query, "chat_history":chat_history})

# Streamlit UI

if __name__ == '__main__':

    st.header("QA ChatBot")
    # ChatInput
    query = st.chat_input("Enter your questions here")

    if "user_history" not in st.session_state:
       st.session_state["user_history"]=[]
    if "response_history" not in st.session_state:
       st.session_state["response_history"]=[]
    if "chat_history" not in st.session_state:
       st.session_state["chat_history"]=[]

    if query:
       with st.spinner("Generating......"):
           output=get_response(llm=llm, retriever=retriever, query=query, chat_history = st.session_state["chat_history"])

          # Storing the questions, answers and chat history

           st.session_state["response_history"].append(output['answer'])
           st.session_state["user_history"].append(query)
           st.session_state["chat_history"].append((query,output['answer']))

    # Displaying the chat history
    if st.session_state["response_history"]:
       for i, j in zip(st.session_state["response_history"],st.session_state["user_history"]):
          message1 = st.chat_message("user")
          message1.write(j)
          message2 = st.chat_message("assistant")
          message2.write(i)