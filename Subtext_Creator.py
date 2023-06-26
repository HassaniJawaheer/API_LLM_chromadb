from langchain.text_splitter import CharacterTextSplitter
import tiktoken
class SubtextCreator:
    
    def __init__(self):
        pass
    
    # Le séparateur de texte:
    def text_splitter(self, myfile):
        Separator = "."
        cutting_model = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=500,separator= Separator, chunk_overlap=0)
        split_texts = cutting_model.split_text(myfile)
        return split_texts
