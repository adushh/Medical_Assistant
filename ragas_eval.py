import os
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv, find_dotenv
from ragas import evaluate
from ragas.metrics import (faithfulness, answer_relevancy, context_recall, context_precision)
import pandas as pd

load_dotenv(find_dotenv())

HF_TOKEN=os.environ.get("HF_TOKEN") #to use token we created the variable
HUGGINGFACE_REPO_ID="mistralai/Mistral-7B-Instruct-v0.3"

def load_llm(huggingface_repo_id):
    llm=HuggingFaceEndpoint(
        repo_id=huggingface_repo_id, 
        temperature=0.5,#help to give crative ans and increase word count
        model_kwargs={"token":HF_TOKEN,
                      "max_length":"512"} #model key words argument
    )
    return llm
llm=load_llm(HUGGINGFACE_REPO_ID)

DB_FAISS_PATH="vectorstore/db_faiss"
embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db=FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)


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

qa_chain=RetrievalQA.from_chain_type(
    llm=load_llm(HUGGINGFACE_REPO_ID),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k':3}) ,#ranked results
    return_source_documents=True,
    chain_type_kwargs={'prompt':set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)

retriever = db.as_retriever(search_kwargs={'k':3})

#user_query=input("Write Query Here: ")
#response=qa_chain.invoke({'query': user_query})

#print("RESULT: ", response["result"])

questions =["What is Cancer?",
           "What is Diabetes",
           "How to maintain good health?"]

ground_truth = ["Cancer is a disease in which some of the body’s cells grow uncontrollably and spread to other parts of the body. Normally, human cells grow and divide in a controlled way to form new cells as needed. However, when this process goes wrong, abnormal or damaged cells grow and multiply uncontrollably, forming tumors (lumps of tissue). Some tumors are benign (non-cancerous), but others are malignant (cancerous) and can invade nearby tissues or spread to distant parts of the body through the blood and lymphatic systems.",
                "Diabetes is a chronic disease where the body cannot properly regulate blood sugar (glucose) due to a lack of insulin or insulin resistance.",
                "Maintain good health by eating a balanced diet, staying active, getting 7-9 hours of sleep, managing stress, avoiding smoking and excessive alcohol, staying hydrated, and having regular health check-ups."]

answer=[]
content=[]

#inference
for query in questions:
    response=qa_chain.invoke({'query': query})
    answer.append(response["result"])
    content.append([docs.page_content for docs in retriever.get_relevant_documents(query)])
    


#df = pd.DataFrame({"Question": question, "Answer": answer, "Retrieved Content": content})
#print(df)

#To dataset format
data={"question": questions,
      "ground_truth":ground_truth,
      "answer":answer,
      "contexts":content}

from datasets import Dataset
dataset = Dataset.from_dict(data)

#print(dataset.features)
#print(len(dataset))


result = evaluate(dataset = dataset,
                  metrics=[context_precision, context_recall, faithfulness, answer_relevancy],
                  llm = llm,
                  embeddings = embedding_model)
print(result)





      


