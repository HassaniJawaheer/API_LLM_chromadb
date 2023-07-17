#Import des bibliothèque
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceInstructEmbeddings
import os


#-------------------------------XXXXXX---------------------------------
# Le séparateur de document:
def splitter(document):
    "Découpe le document en chunks de 500  tokens maximuns"
    if type(document)  == str:
        Separator = "."
        cutting_model = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=500,separator= Separator, chunk_overlap=0)
        chunks_document = cutting_model.split_text(document)
        return {"chunks": chunks_document, "type": "txt"}
    else:
        cutting_model = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
        chunks_document = cutting_model.split_documents(document)
        return {"chunks": chunks_document, "type": "pdf"}


#-------------------------------XXXXXX---------------------------------
#La vectorisation : Embedding
def chunk_embedding(chunks_document, openai_key, model_name):    
    os.environ["OPENAI_API_KEY"]  = openai_key
    if chunks_document["type"] == "txt":
        #Création de la base de données avec chromadb:
        model_embedding = OpenAIEmbeddings(model=model_name)
        embedding_db = Chroma.from_texts(texts= chunks_document["chunks"], 
                                            embedding= model_embedding)
        return embedding_db
    else:
        embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl", 
                                                      model_kwargs={"device": "cuda"})
        embedding_db = Chroma.from_documents(documents= chunks_document["chunks"], 
                  embedding = embeddings)
        
        return embedding_db


#-------------------------------XXXXXX---------------------------------
#Le répondeur
def answering(question, embedding_db, openai_key, chain_type, k):
    os.environ["OPENAI_API_KEY"]  = openai_key
    question = str(question)
    retriever = embedding_db.as_retriever(search_kwargs={"k": k})
    #Création du modèle LLM:
    model_llm = OpenAI()
    #On crée la chaine:
    model_chain = RetrievalQA.from_chain_type(model_llm,  
                              chain_type=chain_type, 
                              retriever=retriever)
    #Répondre à la question:
    response = model_chain({"query": question})
    return response['result']