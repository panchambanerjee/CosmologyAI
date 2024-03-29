## CosmologyAI

**This repository is intended to be a collection of various Data Science, Data Analytics, AI and LLM-based experiments (RAG, Fine-Tuning) in the Cosmology and Extragalactic Astronomy domain**

![nasa-hubble-space-telescope-EsS5IAvx9rc-unsplash](https://github.com/panchambanerjee/CosmologyAI/assets/17071658/898bbfe5-d873-422f-8ca5-7bd4f3ff2b5c)

(The image above is a Hubble Space Telescope image of the Star-Forming Region LH 95 in the Large Magellanic Cloud)

**So far**:

#### 1. Assemble Cosmology-related abstracts from the ArXiv dataset (Kaggle, Cornell) - Notebook and script (get_cosmo_data_from_arxiv.*) uploaded to **arxiv_project/code**
  
#### 2. Build a basic chatbot (No memory) with LangChain and Ollama embeddings, run it locally on a Mac, use Groq for LPU, Gradio for the interface - Notebook and script (chatbot_cmb_basic.*) uploaded to **cmb_rag/code**. Relevant CMB review papers are in **cmb_rag/cmb_data**

Screenshot of RAG QA::

<img width="1103" alt="Screenshot 2024-03-27 at 10 34 32 AM" src="https://github.com/panchambanerjee/CosmologyAI/assets/17071658/1af04401-2d21-4cc4-a279-78e00c11566e">

#### 3. Create vectordb and persist it using Chroma and the Cosmology arxiv abstracts (~66k abstracts) - Notebook and script (create_cosmo_vectordb.*) uploaded to  **arxiv_project/code**

#### 4. Code to take the assembled dataset and build a RAG chatbot (No memory) utilizing Mixtral-8x7B from NVIDIA (LangChain integration), all-MiniLM-L6-v2, LangChain and ChromadB - Notebook and script (create_cosmo_vectordb.*) uploaded to  **arxiv_project/code**

Screenshot of RAG QA (With and without context):: (For the question: "How can we test General Relativity (GR) at cosmological scales with the Weyl potential")

<img width="825" alt="Screenshot 2024-03-29 at 2 02 24 PM" src="https://github.com/panchambanerjee/CosmologyAI/assets/17071658/77e42b73-5a50-460c-8f04-3d56fefa3b67">
  
#### 5. Using the same techstack, build a context-based retrieval search - Notebook and script (create_cosmo_vectordb.*) uploaded to  **arxiv_project/code**

Screenshot of Context Retrieval (Semantic) Search results:: (For the question: "How can we test General Relativity (GR) at cosmological scales with the Weyl potential")

<img width="1354" alt="Screenshot 2024-03-29 at 2 02 47 PM" src="https://github.com/panchambanerjee/CosmologyAI/assets/17071658/875520d0-aba8-4116-a620-933b9c8d9df4">

### Next Steps and Ideas:
* Use Bonito, make instruction-tuned dataset to evaluate RAG application
* Evaluate RAG application using RAGAS
* Explore alternative ways to evaluate RAG application
* Visualize RAG application
* Explore fine-tuning an LLM using instruction-tuned dataset
* Evaluate fine-tuned LLM vs pre-trained
* Explore Advanced RAG (Reranking etc) using both LangChain and LlamaIndex
* Explore context evaluation using TruLens
* Explore different fine-tuning methods, perhaps DPO if we can build a Cosmology preference dataset
* Try DSPy for RAG
* Create a proper chatbot with memory
* Get the paper text and build datasets with that
* Build full applications (RAG, Fine-tuning) based on full paper texts
* Build Knowledge Graph RAGs
* Auto-detect formulae from papers, convert them to LaTex, and verify the correctness

![nasa-hubble-space-telescope-V-B9vdPfJxw-unsplash](https://github.com/panchambanerjee/CosmologyAI/assets/17071658/051e789b-8989-4589-b5e0-46bafb086650)

(The image above is the Hubble Interacting Galaxy IRAS 18090)
