# %% [markdown]
# ### In this notebook, we build a simple RAG application using the Cosmology data that we download from the Arxiv dataset 
# 
# ### The notebook is divided into 2 parts::
# * **Part 1**: We test out a minimal RAG chatbot (No memory) with the Context augmented LLM with several of the latest Cosmology papers from Arxiv (that are not a part of the training corpus of the model, as of March, 2024)
# * **Part 2**: We try a context retrieval search 
# 
# ### Techstack: 
# * LangChain - Framework
# * Mixtral-8x7B from NVIDIA - LLM 
# * Chromadb - Vector database
# * all-MiniLM-L6-v2 - Embedding Model

# %%
import pandas as pd
import textwrap
from openai import OpenAI

from langchain_community.vectorstores import Chroma
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain.embeddings import HuggingFaceEmbeddings

from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# %%
import os

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

# openai_api_key = os.environ['OPENAI_API_KEY']
# hf_api_key = os.environ['HF_API_KEY']

groq_api_key = os.environ['GROQ_API_KEY']
nvidia_api_key = os.environ['NVIDIA_API_KEY']   

# %% [markdown]
# ### Load in the vectordB that we build with the ~66k arxiv cosmology title+abstracts

# %%
# Get the embedding model, we need this again to load in the persisted vectordb

model_name = "sentence-transformers/all-MiniLM-l6-v2" #"BAAI/bge-small-en-v1.5"#"sentence-transformers/all-MiniLM-l6-v2" #"sentence-transformers/all-mpnet-base-v2"
# bge-base-en-v1.5 or bge-small taking too much time for all the cosmo docs, ~66k
model_kwargs = {"device": "cpu"} # Since we are running on local machine, we will use CPU

embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

# %%
vectordb = Chroma(persist_directory='./arxiv_cosmo_chroma_db', embedding_function=embeddings)
retriever = vectordb.as_retriever()

# %% [markdown]
# ## Part 1: Test out a minimal RAG "chatbot" (No memory) with the given context

# %% [markdown]
# ### Load in Mixtral 8x7B from the LangChain and NVIDIA integration, and build the RAG application
# https://build.nvidia.com/mistralai/mixtral-8x7b-instruct

# %%
llm = ChatNVIDIA(model="mixtral_8x7b")

rag_template = """Answer the question based only on the following context:
{context}
Question: {question}
"""
rag_prompt = ChatPromptTemplate.from_template(rag_template)
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)

# %%
# Test it out

response = rag_chain.invoke("What is a Galaxy Cluster?")
print(textwrap.fill(response, width=80))

# %%
response = rag_chain.invoke("What is the Cosmological Constant?")
print(textwrap.fill(response, width=80))

# %% [markdown]
# ### We will evaluate this properly later, but as a sanity check, let's test out the same queries on the Mixtral 8x7B model without any RAG context

# %%
# Define the method
def query_no_context(question):
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=nvidia_api_key  # Make sure 'nvidia_api_key' is defined or passed as an argument
    )

    completion = client.chat.completions.create(
        model="mistralai/mixtral-8x7b-instruct-v0.1",
        messages=[{"role": "user", "content": question}],
        temperature=0.5,
        top_p=1,
        max_tokens=1024,
        stream=True
    )

    output = ""
    for chunk in completion:
        if chunk.choices[0].delta.content is not None:
            output += chunk.choices[0].delta.content

    # `output` contains the complete summed up output.
    return output


# %%
# Example usage
print(textwrap.fill(query_no_context("What is a Galaxy Cluster?"), width=80))

# %%
# Example usage
print(textwrap.fill(query_no_context("What is the Cosmological Constant?"), width=80))

# %% [markdown]
# ### The problem with these 2 questions (Galaxy Cluster, Cosmological Constant) is that they are well-established areas of research and were likely present in whatever corpus was used to train the Mixtral 8x7B model. Let us instead check how well the context augmentation is working by asking a question from a more recent paper

# %%
df_data = pd.read_csv('arxiv_astro-ph_data_cosmo.csv')
df_data = df_data.loc[df_data['categories']=='astro-ph.CO']

df_data = df_data.reset_index(drop=True)

df_data.tail(10) # 10 most recent Cosmology papers in the dataset, not cross-disciplinary

# %%
print(df_data.loc[df_data['id']=='2403.13068']['prepared_text'].iloc[0]) # Q1

# %%
print(df_data.loc[df_data['id']=='2403.13709']['prepared_text'].iloc[0]) # Q2

# %%
print(df_data.loc[df_data['id']=='2403.14580']['prepared_text'].iloc[0]) # Q3

# %% [markdown]
# ### Now, let us test out these questions one by one 

# %% [markdown]
# #### Q1. JWST and exotic high-z objects

# %%
# No Context

print(textwrap.fill(query_no_context("What does the James Webb Space Telescope tell us about exotic objects at high redshift?"), width=120))

# %%
# With context

print(textwrap.fill(rag_chain.invoke("What does the James Webb Space Telescope tell us about exotic objects at high redshift?"), width=120))

# %% [markdown]
# #### Q2. Testing General Relativity with the Weyl potential

# %%
# No context

print(textwrap.fill(query_no_context("How can we test GR at cosmological scales with the Weyl potential"), width=120))

# %%
# With context

print(textwrap.fill(rag_chain.invoke("How can we test GR at cosmological scales with the Weyl potential"), width=120))

# %% [markdown]
# #### Q3. Testing the Cosmological Principle using the CMB Dipole

# %%
# No context

print(textwrap.fill(query_no_context("What can the CMB dipole tell us about the validity of the Cosmological principle?"), width=120))

# %%
# With context

print(textwrap.fill(rag_chain.invoke("What can the CMB dipole tell us about the validity of the Cosmological principle?"), width=120))

# %% [markdown]
# ### So, in all 3 cases, using RAG results in very context specific refined answers, as opposed to the more generic answers from the Mixtral-8x7B model without context

# %% [markdown]
# ## Part 2: Test out a context retrieval (semantic) search 
# 
# **Note** For this part we don't need the LLM, just the embedding model, so there is the scope to compare different embedding models here
# 
# **Also Note** The similarity score returned here is the L2 distance from the query to the relevant document, so lower score is better

# %%
def search_vectordb_and_format_output(query, k, vectordb=vectordb):
    # Perform the similarity search
    results = vectordb.similarity_search_with_score(query, k=k)
    
    # Initialize an empty list to hold formatted results
    formatted_results = []
    
    # Iterate over the results to extract and format the desired information
    for doc, score in results:
        formatted_result = {
            'paper_id': doc.metadata['id'],
            'paper_title': doc.metadata['title'],
            'similarity_score': score
        }
        formatted_results.append(formatted_result)
    
    # Return or print the formatted results
    return formatted_results # Top k results

# %%
query = "What is a galaxy cluster?"
k=5

formatted_results = search_vectordb_and_format_output(query, k=k)


print(query)
print()

# Printing the results
for result in formatted_results:
    print(f"Paper ID: {result['paper_id']}, Paper Title: {result['paper_title']}, Similarity Score: {result['similarity_score']}")
    print()

# %%
query = "What is the cosmological constant?"
k=5

formatted_results = search_vectordb_and_format_output(query, k=k)

print(query)
print()

# Printing the results
for result in formatted_results:
    print(f"Paper ID: {result['paper_id']}, Paper Title: {result['paper_title']}, Similarity Score: {result['similarity_score']}")
    print()

# %% [markdown]
# ### Now let's try out the 3 questions we tested out the RAG Mixtral-8x7B application with; the most recent papers should have the lowest similarity score (i.e. they should be closest semantically to the query)

# %%
query_1 = "What does the James Webb Space Telescope tell us about exotic objects at high redshift?"
k=5

formatted_results = search_vectordb_and_format_output(query_1, k=k)

print(query_1)
print()

# Printing the results
for result in formatted_results:
    print(f"Paper ID: {result['paper_id']}, Paper Title: {result['paper_title']}, Similarity Score: {result['similarity_score']}")
    print()

# %% [markdown]
# Interestingly, the recent JWST paper, which does seem intuitively to be most relevant, does not show up in the top 5 results. Is this expected? Or is this a deficiency of how the RAG application is constructed? (embedding model, chunking strategies, etc) **Something to explore**

# %%
query_2 = "How can we test GR at cosmological scales with the Weyl potential"
k=5

formatted_results = search_vectordb_and_format_output(query_2, k=k)

print(query_2)
print()

# Printing the results
for result in formatted_results:
    print(f"Paper ID: {result['paper_id']}, Paper Title: {result['paper_title']}, Similarity Score: {result['similarity_score']}")
    print()

# %%
query_3 = "What can the CMB dipole tell us about the validity of the Cosmological principle?"
k=5

formatted_results = search_vectordb_and_format_output(query_3, k=k)

print(query_3)
print()

# Printing the results
for result in formatted_results:
    print(f"Paper ID: {result['paper_id']}, Paper Title: {result['paper_title']}, Similarity Score: {result['similarity_score']}")
    print()

# %% [markdown]
# For Q3 as well, the 2403... paper does not show up in the top results, however, this does seem to be a more general question than the JWST one

# %%



