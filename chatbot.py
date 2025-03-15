import os
import streamlit as st
import speech_recognition as sr
import tempfile

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

DB_FAISS_PATH="vectorstore/db_faiss"
@st.cache_resource
def get_vectorstore():
    embedding_model=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db=FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db


def set_custom_prompt(custom_prompt_template):
    prompt=PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt


def load_llm(huggingface_repo_id, HF_TOKEN):
    llm=HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        model_kwargs={"token":HF_TOKEN,
                      "max_length":"512"}
    )
    return llm

def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("🎤 Say...")
        recognizer.adjust_for_ambient_noise(source)  # Reduce background noise
        audio = recognizer.listen(source)  # Capture audio

        try:
            text = recognizer.recognize_google(audio)  # Convert speech to text
            return text
        except sr.UnknownValueError:
            return "Sorry, could not understand the audio."
        except sr.RequestError:
            return "Could not request results.Can you check your internet connection."
            
def main():
    st.title("🩺 MediBot - Your AI Healthcare Assistant")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])
        
    # --- Fix Input and Mic Button at the Bottom ---
    st.markdown("""
        <style>
            .bottom-container {
                position: fixed;
                bottom: 0;
                left: 0; 
                width: 100%;
                background: white;
                padding: 10px;
                border-top: 1px solid #ccc;
                display: flex;
                justify-content: space-between;
                align-items: center;
                z-index: 1000;
            }
            .stChatInput {
                flex-grow: 1;
                margin-right: 10px;
            }
            .stButton {
                flex-shrink: 0;
            }
            .st-emotion-cache-1kyxreq{
                bottom : 80px ! important;
            }
        </style>
     """, unsafe_allow_html=True)    
        
    st.markdown("""   
        <script>
            function scrollToBottom() {
                var chatContainer = window.parent.document.querySelector(".stChatMessage");
                if (chatContainer) {
                    chatContainer.scrollTop = chatContainer.scrollHeight;
                }
            }
            setInterval(scrollToBottom, 100);
        </script>
        
    """, unsafe_allow_html=True)
    #Layout :Text Input(Left) + Mic Button(Right)
    
    st.markdown('<div class="bottom-container">', unsafe_allow_html=True)
    
    col1, col2 = st.columns([5, 1])  
    with col1:
        prompt = st.chat_input("Type your message...")

    with col2:
        if st.button("🎙️"):
            voice_text = recognize_speech()  # Get voice input
            if voice_text and "Sorry, could not understand" not in voice_text:
                st.write(f"**You said:** {voice_text}")
                prompt = voice_text  # Assign voice text to chatbot input
            else:
                st.write("🚫 Could not understand the audio. Please try again.")

    st.markdown('</div>', unsafe_allow_html=True)
        
    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role':'user', 'content': prompt})

        CUSTOM_PROMPT_TEMPLATE = """
                Use the pieces of information provided in the context to answer user's question.
                If you dont know the answer, just say that you dont know, dont try to make up an answer. 
                Dont provide anything out of the given context

                Context: {context}
                Question: {question}

                Start the answer directly. No small talk please.
                """
        
        HUGGINGFACE_REPO_ID="mistralai/Mistral-7B-Instruct-v0.3"
        HF_TOKEN=os.environ.get("HF_TOKEN")
        llm=load_llm(huggingface_repo_id=HUGGINGFACE_REPO_ID, HF_TOKEN=HF_TOKEN)

        try: 
            vectorstore=get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load the vector store")

            qa_chain=RetrievalQA.from_chain_type(
                llm=load_llm(huggingface_repo_id=HUGGINGFACE_REPO_ID, HF_TOKEN=HF_TOKEN),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k':3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt':set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )

            response=qa_chain.invoke({'query':prompt})

            result=response["result"]
            #source_documents=response["source_documents"]
            #result_to_show=result+"\nSource Docs:\n"+str(source_documents)
            
            st.chat_message('assistant').markdown(result)
            st.session_state.messages.append({'role':'assistant', 'content': result})

        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 
    
