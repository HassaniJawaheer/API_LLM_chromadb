#Import des Librairies
from fastapi import FastAPI, UploadFile, File
from fastapi.exceptions import HTTPException
import uvicorn
from pydantic import BaseModel
from langchain.document_loaders import PyPDFLoader, DirectoryLoader

#import du module du projet:
import document_analyser as da


#Création de l'api:
app_proj = FastAPI(title= "API TEXT ANALYSER", version = "v1")


#Présentation et mise en page:
@app_proj.get("/")
async def presentation():
    Text = str(" Crée une api, avec fastapi qui prend en entrée une" + " " + 
               "question et va chercher la réponse dans un texte que je fournis" + " " + 
               "en pièce jointe.Pour chercher cette réponse, tu dois utiliser" +  " " + 
               "un mix de base de données vectorielle, ainsi qu’un" + " " + 
               "LLM pour trouver la réponse à ta question dans le texte")
    return {"Fonction de l'API": Text }

#creat objet to upload the parameters of models
class Parameters(BaseModel):
    key_openai: str
    model_embedding: str
    chain_type: str
    
store_parameters= []

#register document
file_document = []

#register the openai key and the model of embedding and chain type:
@app_proj.post("/models")
async def enter_parameters_model(parameters_model: Parameters):
    store_parameters.append(parameters_model)
    return parameters_model

#change template settings:
@app_proj.put("/modified_parameters_models")
async def uptdate_parameter(new_model: Parameters):
    try:
        store_parameters[0] = new_model
        return store_parameters[0]
    except:
        raise HTTPException(status_code = 404, detail = "Invalid model ")

#upload the document
@app_proj.post("/upload_document")
def upload(file: UploadFile = File(...)):
    if file.content_type == "text/plain":
        try: 
            contents = open(file.filename, encoding="utf8")
            doc_contents = contents.read()
            file_document.append(doc_contents)
        except Exception:
           return {"message": "There was an error uploading the file"}
        finally:
           file.file.close()
        return {"message": f"Successfully uploaded {file.filename}"}
    else:
        try:
           documents = DirectoryLoader("", glob = file.filename, loader_cls=PyPDFLoader)
           doc_contents = documents.load()
           file_document.append(doc_contents)
        except Exception:
            return {"message": "There was an error uploading the file"}
        finally:
            file.file.close()
        return {"message": f"Successfully uploaded {file.filename}"}

#Poser la question:
@app_proj.post("/answer_question")
async def register_question(query: str):
    # Découpage du texte
    split_doc = da.splitter(file_document[0])
    #Création de la base de données vectorielle:
    chunks_db = da.chunk_embedding(split_doc, store_parameters[0].key_openai, 
                                   store_parameters[0].model_embedding)
    #On répond à la question : 
    response = da.generate_response(query, chunks_db, store_parameters[0].key_openai,
                            chain_type= store_parameters[0].chain_type, k = 3)
    return {query: response}

#Open a the window:
import webbrowser
webbrowser.open("http://127.0.0.1:8000/docs#/")

#Execution:
if __name__ == "__main__":
    uvicorn.run(app_proj, host="127.0.0.1", port=8000)