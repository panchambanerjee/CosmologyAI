# CosmologyAI 🌠

**This repository is a collection of various Data Science, Data Analytics, AI, and LLM-based experiments (RAG, Fine-Tuning) in the Cosmology and Extragalactic Astronomy domain.**

![Hubble Space Telescope image of the Star-Forming Region LH 95 in the Large Magellanic Cloud](https://github.com/panchambanerjee/CosmologyAI/assets/17071658/898bbfe5-d873-422f-8ca5-7bd4f3ff2b5c)

The image above is a Hubble Space Telescope image of the Star-Forming Region LH 95 in the Large Magellanic Cloud.

## So far

<details>
  
<summary><b>Assemble Cosmology-related abstracts from the ArXiv dataset (Kaggle, Cornell)</b></summary>

- Notebook and script (`get_cosmo_data_from_arxiv.*`) uploaded to `arxiv_project/code`.

</details>

<details>
  
<summary><b>Build a basic chatbot (No memory) with LangChain and Ollama embeddings, running it locally on a Mac, using Groq for LPU, Gradio for the interface</b></summary>

  - Notebook and script (`chatbot_cmb_basic.*`) uploaded to `cmb_rag/code`. Relevant CMB review papers are in `cmb_rag/cmb_data`.

  ![Screenshot of RAG QA](https://github.com/panchambanerjee/CosmologyAI/assets/17071658/1af04401-2d21-4cc4-a279-78e00c11566e)

  The above is a screenshot of the RAG QA (using Gradio)

</details>

<details>
  
<summary><b>Create vectordb and persist it using Chroma and the Cosmology arxiv abstracts (~66k abstracts)</b></summary>

- Notebook and script (`create_cosmo_vectordb.*`) uploaded to `arxiv_project/code`.

</details>

<details>
  
<summary><b>Code to take the assembled dataset and build a RAG chatbot (No memory) utilizing Mixtral-8x7B from NVIDIA (LangChain integration), all-MiniLM-L6-v2, LangChain, and ChromadB</b></summary>

 - Notebook and script (`create_cosmo_vectordb.*`) uploaded to `arxiv_project/code`.
 
  <img width="707" alt="Screenshot 2024-04-07 at 12 45 13 PM" src="https://github.com/panchambanerjee/CosmologyAI/assets/17071658/77e42b73-5a50-460c-8f04-3d56fefa3b67">

  The image above is a screenshot of the Mixtral chatbot (No memory)

</details>

<details>
  
<summary><b>Using the same tech stack, build a context-based retrieval search</b></summary>

 - Notebook and script (`create_cosmo_vectordb.*`) uploaded to `arxiv_project/code`.

  <img width="707" alt="Screenshot 2024-04-07 at 12 45 13 PM" src="https://github.com/panchambanerjee/CosmologyAI/assets/17071658/875520d0-aba8-4116-a620-933b9c8d9df4">

  The above is a screenshot of the Semantic search results.

</details>

<details>
  
<summary><b>Using Bonito, an A100 GPU on Google Colab, a Dark Matter Review paper, create an Instruction tuning QA dataset</b></summary>

 - Notebook and script (`Instruction_Dataset_Synth_bonito_Dark_Matter_Review.ipynb`) uploaded to `miscellaneous/code`. The Dataset is available on HuggingFace Hub: delayedkarma/dark_matter_instruction_qa.

  ![Screenshot of Questions and Answers generated](https://github.com/panchambanerjee/CosmologyAI/assets/17071658/a1a113e6-5a81-47c2-8577-6f7b7febf968)

  The above is a screenshot of the generated dataset

</details>

<details>
  
<summary><b>LangChain RAG from Scratch (https://github.com/langchain-ai/rag-from-scratch/blob/main/README.md), using ArXiv Cosmology data</b></summary>

* First notebook (Overview) uploaded to langchain_astro_rag
* Second notebook (Multi-Query, RAG-Fusion, Decomposition, Step-back Prompting, HyDE) uploaded to langchain_astro_rag
* Third notebook (Logical and Semantic Routing, Query Structuring for Metadata filters) uploaded to langchain_astro_rag

</details>

<details>
  
<summary><b>Working RAG based conversational chatbots (with memory) using Langchain and Streamlit</b></summary>

  - Uploaded v1, v2 and v3 of the scripts to chatbots/code

   <img width="707" src="https://github.com/panchambanerjee/CosmologyAI/assets/17071658/a87b86fa-7881-4bfb-af55-917bba11efe0">

  The above is a screenshot of the current working version of the astro_v3.. chatbot (Using Streamlit)

  Streamlit Chatbot To-Do:
  * Optimize load-time
  * Add background image
  * Make history display cleaner

</details>

<details>
  
<summary><b>Initial RAG Evaluate notebook using RAGAS</b></summary>

* Uploaded synthetic dataset for evaluation to rag_evaluate/ragas_evaluate/data
* Uploaded v1 notebook to rag_evaluate/ragas_evaluate/code

</details>
 

<details>

<summary><b>Download relevant papers from arxiv programmatically</b></summary>

* Uploaded initial notebook to miscellaneous/code/notebooks/download_arxiv_papers.ipynb

</details>
 

<details>
  
<summary><b>Next Steps and Ideas</b></summary>

* Use Bonito, make instruction-tuned dataset to evaluate RAG application.
* Evaluate RAG application using RAGAS.
* Explore alternative ways to evaluate RAG application.
* Visualize RAG application.
* Explore fine-tuning an LLM using instruction-tuned dataset.
* Evaluate fine-tuned LLM vs pre-trained.
* Explore Advanced RAG (Reranking etc) using both LangChain and LlamaIndex.
* Explore context evaluation using TruLens.
* Explore different fine-tuning methods, perhaps DPO if we can build a Cosmology preference dataset.
* Try DSPy for RAG.
* Create a proper chatbot with memory.
* Get the paper text and build datasets with that.
* Build full applications (RAG, Fine-tuning) based on full paper texts.
* Build Knowledge Graph RAGs.
* Auto-detect formulae from papers, convert them to LaTeX, and verify the correctness.
* Agents.
* Use AssemblyAI (Or some other tool) to summarize lectures, specifically Cosmology lectures (Leonard Susskind etc): https://www.youtube.com/watch?v=P-medYaqVak&list=PLvh0vlLitZ7c8Avsn6gUaWX05uD5cedO-&ab_channel=Stanford
* Evaluate RAG for several different methods, Query decomposition, Step-back Prompting, RAG-Fusion etcetera

</details>


![Hubble Interacting Galaxy IRAS 18090](https://github.com/panchambanerjee/CosmologyAI/assets/17071658/051e789b-8989-4589-b5e0-46bafb086650)

The image above is the Hubble Interacting Galaxy IRAS 18090
