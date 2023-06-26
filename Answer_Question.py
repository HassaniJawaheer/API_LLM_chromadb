from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

def Answering(question, data_base):
    retriever = data_base.as_retriever(search_kwargs ={"k": 3})
    #Création du modèle LLM:
    model_llm = OpenAI()
    #On crée la chaine:
    model_chain = RetrievalQA.from_chain_type(model_llm,  
                              chain_type="stuff", 
                              retriever=retriever, 
                              return_source_documents=True)
    #Répondre à la question:
    response = model_chain({"query": question})
    return response['result']