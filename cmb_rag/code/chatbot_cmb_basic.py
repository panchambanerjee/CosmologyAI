# %% [markdown]
# ### Here we build a basic RAG chatbot using Ollama Nomic Embeddings, Llama2 70b from Groq LPU, LangChain, Gradio
# ### For the data sources we use articles on The Cosmic Microwave Background Radiation (CMB)::
# * https://arxiv.org/abs/1210.6008v1 - CMB Review, Challinor, 2012
# * https://arxiv.org/abs/1606.03606 - CMB Foreground review, Challinson, 2016
# * https://arxiv.org/abs/0803.0834 - CMB Review, Samtleben at al., 2008

# %% [markdown]
# You should have a .env file in the environment in which you are running this notebook/script, which contains the line
# GROQ_API_KEY='Your API Key here'
# 
# Ollama should also be running on your local machine
# 
# #### References:: 
# * https://www.linkedin.com/pulse/build-lightning-fast-rag-chatbot-powered-groqs-lpu-ollama-multani-ssloc/
# * https://colab.research.google.com/drive/1Obrby8RniFfjUvf3DhbNHC6-CmBdiXbY?usp=sharing
# 

# %%
# Get the Nomic embeddings
import os

os.system("ollama pull nomic-embed-text")

# %%
import gradio as gr

import time
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community import embeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

import textwrap

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

# openai_api_key = os.environ['OPENAI_API_KEY'] # Not needed for this example
# hf_api_key = os.environ['HF_API_KEY'] # Not needed for this example

groq_api_key = os.environ['GROQ_API_KEY']

# %% [markdown]
# ### Define the RAG Components

# %%
# Load PDF documents from the 'data' directory
loader = PyPDFDirectoryLoader("../data")
the_text = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
chunks = text_splitter.split_documents(the_text)

# %%
# Setup the vector store and retriever
vectorstore = Chroma.from_documents(
    documents=chunks,
    collection_name="ollama_embeds",
    embedding=embeddings.ollama.OllamaEmbeddings(model='nomic-embed-text'),
)
retriever = vectorstore.as_retriever()

# %%
# Get the LLM (Llama2-70b) from Groq

llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name='llama2-70b-4096'
)

# %%
def process_question(question):

    # Define the RAG template and chain
    template = """Answer the question based only on the following context:
    {context}
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain.invoke(question)

# %%
# Setup the Gradio interface
iface = gr.Interface(fn=process_question,
                     inputs=gr.Textbox(label='User Question', lines=2, placeholder="Type your question here... "),
                     outputs=gr.Textbox(label='LLM Response'),
                     title="A Chat about the Oldest Light in the Universe",
                     description="Ask any question about The Cosmic Microwave Background Radiation. It's niche but it's magical! ",
                    )

# Launch the interface
iface.launch()

# %%



