from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

#PHASE 1:
#step1 : RAW DATA

DATA_PATH="data/"
def load_pdf_files(data):
    loader = DirectoryLoader(data,
                             glob='*.pdf',
                             loader_cls=PyPDFLoader)
    
    documents= []
    try:
        documents =loader.load()
    except Exception as e:
        print(f"Error loading PDF : {e}")
        
    return documents
#till this returning all documents of books like all pages in form of documents and its is returning for all pdfs in data folder

documents=load_pdf_files(data=DATA_PATH)

#if documents:
    #print("Length of PDF PAges:",len(documents))
#else:
    #print("No Doc Were Loaded.")

#step2 : Create Chunks

def create_chunks(extracted_data):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,
                                                 chunk_overlap=50)
    text_chunks=text_splitter.split_documents(extracted_data)
    return text_chunks

text_chunks=create_chunks(extracted_data=documents)
#print("Length of Text chunks:",len(text_chunks))

#here buffer of each chunk is 500 and chunk overlap size is 50 for maintaining understandibility 

#step3 : Creating a Vector Embedding

def get_embedding_model():
    embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model

embedding_model=get_embedding_model()

#step4 : Stroe Embeddings in FAISS

DB_FAISS_PATH="vectorstore/db_faiss"
db=FAISS.from_documents(text_chunks, embedding_model)
db.save_local(DB_FAISS_PATH)
#now it will store at my folder


#phase 2-----------------------


 