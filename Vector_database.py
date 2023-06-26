from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma
import os
class VectorDatabase:
    
    #Initialisation avec un clés de l'openai:
    def __init__(self, openai_key):
        self.openai_key = openai_key
        
    def vectorization(self, split_text):    
        os.environ["OPENAI_API_KEY"]  = self.openai_key
        #Création de la base de données avec chromadb:
        model_vectorization = OpenAIEmbeddings()
        vector_DataBase = Chroma.from_texts(texts= split_text, 
                                            embedding= model_vectorization)
        return vector_DataBase