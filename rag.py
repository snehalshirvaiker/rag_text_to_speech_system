from fastapi import FastAPI, Request
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.vectorstores.utils import filter_complex_metadata
# from langchain_community.embeddings import FastEmbedEmbeddings
# from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from datetime import datetime
import re

app = FastAPI()

class ChatPDF:
    def __init__(self):
        self.model = ChatOllama(model="mistral")
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        self.prompt = PromptTemplate.from_template(
            """
            <s> [INST] You are an assistant that provides an answer of the given query from the document. 
            Keep your answer to two or three sentences. [/INST] </s> 
            [INST] Question: {question} 
            Context: {context} 
            Answer: [/INST]
            """
        )
        self.vector_store = None
        self.retriever = None
        self.chain = None
        #Defining simple conversational queries 
        self.simple_queries = {
            "hello": "Hello! How can I assist you today?",
            "how are you": "I'm good and here to help you!",
            "whats up": "Not much, just here to answer your questions.",
            "weather": "I'm not equipped to give weather updates, but rainy? hehe",
            "who are you": "I am batman hehe",
            "time": f"The time right now is {datetime.now().strftime('%H:%M:%S')}",
            "date": f"Today's date is {datetime.now().strftime('%Y-%m-%d')}",
            "today": f"Today's date is {datetime.now().strftime('%Y-%m-%d')}"
        }

    def ingest(self, pdf_file_path: str):
        docs = PyPDFLoader(file_path=pdf_file_path).load()
        chunks = self.text_splitter.split_documents(docs)
        chunks = filter_complex_metadata(chunks)
        # embeddings = FastEmbedEmbeddings()
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        vector_store = Chroma.from_documents(documents=chunks, embedding=embeddings)
        self.retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 5,
                "score_threshold": 0.5,
            },
        )

        self.chain = ({"context": self.retriever, "question": RunnablePassthrough()}
                      | self.prompt
                      | self.model
                      | StrOutputParser())

    def ask(self, question: str):
        question = question.lower()
        for key, response in self.simple_queries.items():
            if re.search(rf"\b{key}\b", question):
                if callable(response):
                    return response()
                return response
        return self.chain.invoke(question)
    
chat_pdf = ChatPDF()
pdf_path = "iesc111.pdf" 
chat_pdf.ingest(pdf_path)

@app.post("/query/")
async def query_endpoint(request: Request):
    body = await request.json()
    print("body", body)
    question = body.get("question")
    if question:
        answer = chat_pdf.ask(question)  
        return {"response": answer} 
    else:
        return {"error": "Question is required."}
