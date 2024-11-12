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
        self.model = ChatOllama(model="llama3.1")
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        self.prompt = PromptTemplate.from_template(
            """
            <s> [INST] You are an assistant that provides an answers of the given given query from the document. 
            Keep your answers complete and relevent that cover all the points in the context. [/INST] </s> 
            [INST] Document: {context} 
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
            "who are you": "I am a Chatbot who will help you in answering questions about any pdf you insert",
            "time": f"The time right now is {datetime.now().strftime('%H:%M:%S')}",
            "date": f"Today's date is {datetime.now().strftime('%Y-%m-%d')}",
            "today": f"Today's date is {datetime.now().strftime('%Y-%m-%d')}"
        }

    def ingest(self, pdf_file_path: str):
        global embeddings
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
        print("context is",self.retriever)
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

        # Get the result and sources
        print("on chain invoke for question ", question)
        result = self.chain.invoke(question)
        print("on retriever invoke ")
        sources = self.retriever.invoke(question)
        context_texts = [source.page_content for source in sources]
        return result, context_texts


chat_pdf = ChatPDF()
pdf_file_path = 'SRB-PART-I&PART-II-2024-25.pdf'
chat_pdf.ingest(pdf_file_path)
print("PDF ingested successfully")

queries=[
    "What are the Attendance and leave of absence guidelines for all students?",
    "Tell me about the Choice based credit system.",
    "What happens if the marks of re-evaluation decrease?",
    "What are the Guidelines for Scholarship?"
]
results = []
contexts = []

for que in queries:
    answer, context_texts = chat_pdf.ask(que)
    results.append(answer)
    contexts.append(context_texts)

from ragas import evaluate, metrics
from datasets import Dataset
ground_truths=[
'''
A student is required to monitor his /her own attendance regularly.  All doubts regarding attendance should be 
clarified with the concerned faculty within the appropriate time (at the end of each class during recess or at the 
end  of  class  hours  of  the  day).  If  the  student  is  marked  absent  even  when  present,  he/she  should  immediately  
inform the concerned faculty by submitting a written application  informing of his/her presence and if required to 
justify his/her stand. Attendance reports of all the students would be displayed on the notice board / Student Portal on a monthly basis. 
If a student has any issue or finds any discrepancy in his/her attendance, he/she should inform the office in writing 
regarding the same within 3 days of the publication of the attendance record. No claim shall be entertained under 
any circumstances after that. No changes will be permitted once attendance reports are finalized. 
For all absences, prior intimation through a prescribed application form is to be given to Course Coordinator. In 
emergent situations, intimation must be given to Course Coordinator preferably by email within 24 hours of the 
absence. Any absence without written intimation will be treated as unapproved and will be reflected in the records 
as such. It  is  the  sole  responsibility  of  the  student  to  confirm  his  /  her  email  Id  and  parent's  email  Id  with  the  school 
academic office. Parents of the students shall be intimated with the attendance record of their ward if the student has less than 80% 
attendance  monthly  through  email/SMS.  Students  have  to  notify  the  office  in  case  of  a  change  of  any  contact  
information of parents. Parents may be called to the school to discuss the consequences of remaining absent. Students must refrain from approaching the visiting and full-time faculty members for attendance related issues 
and exemptions. They must apply to the concerned coordinator for necessary approvals. If the student remains absent due to any medical issues he / she should submit a medical certificate along with a 
copy  of  all  the  medical  reports  to  the  office  within  3  days  of  resuming  the  classes  after  medical  leave.  No  
certificate shall be entertained under any circumstances thereafter. Final attendance of the Trimester/Semester will be published on Student Portal / Notice board after the end of the 
session. If the  student has any  issues regarding attendance will notify the Dean/Director of  campus  in  writing.  
Dean/Director will ensure data is factual and act accordingly.
''',
    """The CBCS offering is a concept, which is in line with the international academic system which is Applicable for Mumbai campus only.  
Students may opt for the course from the bouquet of courses offered in CBCS in lieu of a course/s dropped (as defined in their school's course structure) or 
take the courses offered by their Schools. The students are also allowed to choose CBCS courses as an additional subject for 
extra credits as a credit course or audit course. The CBCS offerings of courses will  be opted from a UG Program to a UG 
program and PG Program to PG Program across all schools where the grading system is the same. The UG students of 1st 
year and PG students of Term I of 1st year are not eligible to opt CBCS courses. Students studying in the school where the 
teaching/exam scheme is governed by a statutory body can take only extra credit courses.""",
    '''
If the marks of re-evaluation decrease by less than ten percent (10%) There will be no 
change in the marks and original marks awarded to the candidate in the subject will be retained.  
If the marks of re-evaluation decrease between 10.01% and 20%: In such cases the  
marks originally awarded to the candidate in the subject shall be treated as null and void and the marks 
obtained by the candidate after re-evaluation shall be  accepted as the  marks obtained in that subject. 
Fractional  marks  if  any  shall  be  rounded  off  for  the  purpose  of  computing  the  ten  percent  (10%)  
difference in marks. If the marks of re-evaluation increase or decrease by more than 20% A second revaluation of the 
said answer books shall be done by the examiner from the panel of examiners for the said subject and 
then an average of the marks obtained in the first re-evaluation and marks obtained in the second 
re- evaluation shall be taken and this  shall be accepted by the  University as the final  marks for the 
said subject (fractional marks if any shall be rounded off to the next integer)
''',
    """Students are advised to apply for awards and participate in the process enthusiastically. Students are also advised to keep a good performance track record if they wish to apply for these awards. Students 
obtaining F in any subject or with a record of misconduct or a record of low attendance will be automatically disqualified from the awards process. Certificate of merit to be given by Dean at school level. 
A student who wish to apply for scholarship will have to go through school academic office.
"""

]


# Create dictionary for Dataset
d = {
    "question": queries,
    "answer": results,
    "contexts": contexts,
    "ground_truth": ground_truths
}

# Convert to Dataset and evaluate
dataset = Dataset.from_dict(d)
score = evaluate(dataset, metrics=[metrics.faithfulness, metrics.answer_relevancy, metrics.context_precision, 
                                    metrics.context_recall, metrics.context_entity_recall, metrics.answer_similarity, 
                                    metrics.answer_correctness])

# Convert to a DataFrame and save or return the scores
score_df = score.to_pandas()
score_df.to_csv("EvaluationScores.csv", encoding="utf-8", index=False)

# Return the mean scores
score_df[['faithfulness','answer_relevancy', 'context_precision', 'context_recall',
                'context_entity_recall', 'answer_similarity', 'answer_correctness',
                'harmfulness']].mean(axis=0).to_dict()
