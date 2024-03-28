# %% [markdown]
# ### In this notebook/script we use Chroma to create a vectordB with the cosmology arxiv dataset that we have already prepared

# %%
import pandas as pd
from langchain.document_loaders import DataFrameLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

# %%
df_cosmo = pd.read_csv('arxiv_astro-ph_data_cosmo.csv')    
df_cosmo.head()

# %%
df_cosmo.shape

# %% [markdown]
# So we will create the vectordB from these ~66k documents

# %%
# Create a DataFrameLoader
loader = DataFrameLoader(df_cosmo, page_content_column='prepared_text')
arxiv_documents = loader.load()

arxiv_documents[0]

# %% [markdown]
# ### Refer to the MTEB Embeddings Leaderboard for the best performing Embedding: https://huggingface.co/spaces/mteb/leaderboard
# 
# #### Here we optimize for time, will improve on this later, embedding model used here: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

# %%
# Get the embedding model

model_name = "sentence-transformers/all-MiniLM-l6-v2" #"BAAI/bge-small-en-v1.5"#"sentence-transformers/all-MiniLM-l6-v2" #"sentence-transformers/all-mpnet-base-v2"
# bge-base-en-v1.5 or bge-small taking too much time for all the cosmo docs, ~66k
model_kwargs = {"device": "cpu"} # Since we are running on local machine, we will use CPU

embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

# %% [markdown]
# **Note** What is the optimal chunking strategy here?

# %%
### Split the documents into smaller chunks

splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20) 
# Keeping this small initially, since these are just abstracts, not full paper text

chunked_docs = splitter.split_documents(arxiv_documents)

# %%
chunked_docs[0]

# %% [markdown]
# **Note** Try this with FAISS etc as well, how does it affect the performance?

# %%
# Create the vectordb using Chroma and persist it for future use -> Took about ~40 minutes on a Macbook M2 Pro 2023

vectordb = Chroma.from_documents(documents=chunked_docs, embedding=embeddings, persist_directory="arxiv_cosmo_chroma_db")

# %%

vectordb.persist()
