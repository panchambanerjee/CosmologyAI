## CosmologyAI

### This repository is intended to be a collection of various Data Science, Data Analytics, AI and LLM-based experiments (RAG, Fine-Tuning) in the Cosmology and Extragalactic Astronomy domain

![nasa-hubble-space-telescope-aRrf665Cqx8-unsplash](https://github.com/panchambanerjee/CosmologyAI/assets/17071658/869ecbc1-74c4-4af2-ad28-9e066b2d1136)

(The image above is a Hubble Space Telescope image of the Star Cluster NGC 2074 in the Large Magellanic Cloud)

### So far:
* Code to assemble Cosmology-related abstracts from the ArXiv dataset (Kaggle, Cornell) - Notebook and script (get_cosmo_data_from_arxiv.*) uploaded to **arxiv_project/code**
  
* Code to build a basic Chatbot with LangChain and Ollama embeddings, run it locally on a Mac, use Groq for LPU, Gradio for the interface - Notebook and script (chatbot_cmb_basic.*) uploaded to **cmb_rag/code**. Relevant CMB review papers are in **cmb_rag/cmb_data**

Chatbot Screenshot::

<img width="1103" alt="Screenshot 2024-03-27 at 10 34 32 AM" src="https://github.com/panchambanerjee/CosmologyAI/assets/17071658/1af04401-2d21-4cc4-a279-78e00c11566e">


* Code to take the assembled dataset and build a RAG chatbot utilizing Mistral 7b, all-MiniLM-L6-v2, LangChain and ChromadB - Notebook to be uploaded
  
* Using the same techstack, build a context-based retrieval search - Notebook to be uploaded

### Next Steps:
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
  
