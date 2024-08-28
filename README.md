# RAG Chatbot Langchain 🔗

## Overview 🔎
This repository contains the code for a Retrieval-Augmented Generation (RAG) Chatbot built using the LangChain framework. The chatbot leverages both generative language models and external knowledge sources to provide accurate and contextually relevant responses. On top of that, streamlit is used as frontend framework to provide user interface to interact with LLM. The whole RAG application is deployed entirely **LOCALLYYY** without any external hosting.

## What is RAG ❓
Retrieval-Augmented Generation (RAG) is a hybrid approach that combines the strengths of generative language models with the precision of information retrieval systems. The system first retrieves relevant documents from a knowledge base and then uses a generative model to craft a response based on the retrieved information.

## Features 👑
### ▶ Instruct Mode without Context
![](https://github.com/deeplyneuralicious/RAG-Chatbot-Langchain/blob/main/img/Normal%20Instruct%20mode.gif)

### ▶ Upload the Documents to be saved to ChromaDB
![](https://github.com/deeplyneuralicious/RAG-Chatbot-Langchain/blob/main/img/Upload%20PDF%20document.gif)

### ▶ RAG Query Mode with Context
![](https://github.com/deeplyneuralicious/RAG-Chatbot-Langchain/blob/main/img/RAG.gif)

### ▶ Change the parameters to generate better response.
![](https://github.com/deeplyneuralicious/RAG-Chatbot-Langchain/blob/main/img/parameters.png)

## Key Architecture ⚙
The RAG Chatbot consists of the following components:
1. LLM:
   The underlying default is ["microsoft/phi-2"](https://huggingface.co/microsoft/phi-2) running with bfloat16 precision to compromise the local hardware limitation
2. Embedding Model:
   Bi-encoder model ["BAAI/bge-small-en-v1.5"](https://huggingface.co/BAAI/bge-small-en-v1.5) is used to convert text documents into vector representation in high-dimensional vector space.
3. Vector Database:
   Chroma DB is used to store vector index of the text documents.
4. Chunking & Compression:
   Utilizes recursive chunking and FlashRank contextual compression to enhance document retrieval and response quality.
5. Retriever:
   Leverage ChromaDB to search and retrieve relevant documents based on user query. 
6. Frontend:
   A Streamlit-based web application that provides an interface for users to interact with the chatbot.

## Getting Started 📌
1. Clone the repository.
```
git clone https://github.com/deeplyneuralicious/RAG-Chatbot-Langchain
cd RAG-Chatbot-Langchain
```
2. Install dependencies.
```
pip install -r requirements.txt
```
3. Run the app with Streamlit.
```
streamlit run app.py
```
