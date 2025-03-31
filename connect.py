#Phase 2:
import os
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv()) #loaded virtual enviroment to acees it

# Step 1 : Setup LLM(MistralAI)
HF_TOKEN=os.environ.get("HF_TOKEN") #to use token we created the variable
HUGGINGFACE_REPO_ID="mistralai/Mistral-7B-Instruct-v0.3" #to use llm 

#hugging face model setting up with api 
def load_llm(huggingface_repo_id):
    llm=HuggingFaceEndpoint(
        repo_id=huggingface_repo_id, 
        temperature=0.5,#help to give crative ans and increase word count
        model_kwargs={"token":HF_TOKEN,
                      "max_length":"512"} #model key words argument
    )
    return llm

# Step 2 : Connect LLM 
# To stop hallucination......
CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, just say that you dont know, dont try to make up an answer. 
Dont provide anything out of the given context

Context: {context}
Question: {question}
 
Start the answer directly. No small talk please.
"""
#this overal prompt will go to now to the llm....


def set_custom_prompt(custom_prompt_template):
    prompt=PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

#Load Database
DB_FAISS_PATH="vectorstore/db_faiss"
embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db=FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# Step 3 : Create Chain to create a RAG pipeline
qa_chain=RetrievalQA.from_chain_type(
    llm=load_llm(HUGGINGFACE_REPO_ID),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k':3}) ,#ranked results
    return_source_documents=True,
    chain_type_kwargs={'prompt':set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)

# Now invoke with a single query(chain activation)
user_query=input("Write Query Here: ")
response=qa_chain.invoke({'query': user_query})


print("RESULT: ", response["result"])
#print("SOURCE DOCUMENTS: ", response["source_documents"])

#Extract Retived Context
retrived_context = [doc.page_content for doc in response["source_documents"]]

print("\nRetrived Contexts:")
for i, context in enumerate(retrived_context, 1):
    print(f"\nContext {i}: {context}")
