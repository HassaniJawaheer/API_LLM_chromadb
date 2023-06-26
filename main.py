#Bibiothèque standart 
from fastapi import FastAPI
import uvicorn

import Subtext_Creator
import Vector_database
import Answer_Question
#
#On récupère le texte et la clés de l'api d'openai:
myfile = open("Python\env1\Text.txt", encoding="utf8").read()
openai_key = "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"
# On crée l'API:
app = FastAPI()
@app.get("/")
def answer():
    #On récupère la question
    Question = input("Posez votre Question ?")
    Question = str(Question)
    # Découpage du texte
    objet_SubtextCreator = Subtext_Creator.SubtextCreator()
    split_text = objet_SubtextCreator.text_splitter(myfile)
    #Création de la base de données vectorielle:
    objet_VectorDataBase = Vector_database.VectorDatabase(openai_key)
    vector_database = objet_VectorDataBase.vectorization(split_text)
    #On répond à la question : 
    return Answer_Question.Answering(Question, vector_database)

uvicorn.run(app, host = "127.0.0.1", port = 8000)

