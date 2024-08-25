from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain.schema.runnable import RunnablePassthrough,RunnableParallel
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers.document_compressors import FlashrankRerank,EmbeddingsFilter
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain_community.document_transformers import EmbeddingsRedundantFilter
from langchain.retrievers import ContextualCompressionRetriever
import chromadb
#from flashrank import Ranker, RerankRequest

from uuid import uuid4

import os
from model import LLM


CACHE_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),"..", "models"))

class EmbeddingModel:
    def __init__(self, embed_id:str = "BAAI/bge-small-en-v1.5", device ="cpu") -> None:

        self.embed_model = HuggingFaceEmbeddings(
            model_name=embed_id,
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': False}
        )



class ChromaDB:
    def __init__(self,embed_model,persist_directory: str =  "./chroma_db", llm = None) -> None:

        # self.vector_store = Chroma.from_documents(docs, 
        #                                           embedding= embed_model,
        #                                           persist_directory=save_persist_directory)
        self.embed_model = embed_model
        self.persist_directory = persist_directory
        self.vector_store = Chroma(persist_directory=persist_directory,
                                   embedding_function=embed_model)
        self.llm = llm if llm is None else LLM().get_pipeline()
    

    def split_batch(self,docs,max_batch_size:int = 166):
        # Can use client.get_max_batch_size() to get the max batch size but required chomadb package and instantiate a client API

        for i in range(0,len(docs),max_batch_size):
            split_doc_batch = docs[i:i+max_batch_size]
            yield split_doc_batch


    def save(self,docs):

        #directory = save_persist_directory if save_persist_directory else self.persist_directory
        #self.vector_store = Chroma.from_documents(docs,
                                            # embedding= self.embed_model,
                                            # persist_directory=directory)
        for split_doc_batch in self.split_batch(docs):
            uuids = [str(uuid4()) for _ in range(len(split_doc_batch))]
            self.vector_store.add_documents(documents=split_doc_batch,ids=uuids)


    def similarity_search(self,question:str,k:int=2):
        
        contexts = self.vector_store.similarity_search(question,k=k)
        print(contexts)
        context = "".join(context.page_content + "\n" for context in contexts)
        #retriever = self.vector_store.as_retriever(k=no_of_retrievers)

        return context
    
    def retrieve(self, question:str, k:int=4,llm =None):
        #llm = llm if llm is not None else self.llm 
        #mq_retriever = MultiQueryRetriever.from_llm(retriever=self.vector_store.as_retriever(search_kwargs={"k": k}),llm = llm)
        # retrieved_docs = mq_retriever.get_relevant_documents(query = question)
        compressor = FlashrankRerank()
        compression_retriever = ContextualCompressionRetriever(base_compressor=compressor,
                                                               base_retriever=self.vector_store.as_retriever(search_kwargs={"k": k}))

        compressed_docs = compression_retriever.invoke(question)
        context = "".join(context.page_content + "\n" for context in compressed_docs)
        return context





def load_and_split(file_paths:list,chunk_size:int = 400,chunk_overlap:int=35)->list:
    
    loaders = [PyPDFLoader(file_path) for file_path in file_paths]
    pages = []
    
    for loader in loaders:
        pages.extend(loader.load())
    text_splitter = RecursiveCharacterTextSplitter(chunk_size= chunk_size,chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(pages)
    return docs