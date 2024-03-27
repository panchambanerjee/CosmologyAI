## CosmologyAI

This repository is intended to be a collection of various Data Science, Data Analytics, AI and LLM-based experiments (RAG, Fine-Tuning) in the Cosmology and Extragalactic Astronomy domain

### So far:
* Code to assemble Cosmology-related abstracts from the ArXiv dataset (Kaggle, Cornell) - Notebook and script (get_cosmo_data_from_arxiv.*) uploaded to get_data/arxiv_data/code
* Code to build a basic Chatbot with Ollama embeddings, run it locally on a Mac, use Groq for LPU, Gradio for the interface - 
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
  
