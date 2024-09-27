sarvam_rag_assignmemnt/

    .
    ├── app.py                  #Streamlit app script with audio
    ├── iesc111.pdf             #NCERT pdf
    ├── rag.py                  #RAG logic 
    ├── output_audio.wav        #Temporary audio file (created and deleted dynamically)
    ├── README.md  
    ├── requirements.txt        #Python dependencies
    ├── run.sh                  #bash file to execute everything step by step in venv   
    └── text2speech.py          # Trying out the text2speech api here 


# This is a RAG Chatbot 
For simple queries that the bot has to answer, it is retreiving from a dictionary in rag.py as self.simple_queries
More Simple queries can be added can in that dictionary

For other document related questions it checks the similarity score with a threshold of 0.5 and retreives the top 5 related chunks
If found in that document, it will give us the answer; if not, It will use ollama model mistral to retrieve an answer


## Below are the steps to run:

### create a virtual env in python using below commands:

python3 -m venv venv
source /venv/bin/actibvate


### now run the bash file run.sh in terminal using below command:

bash run.sh

##once we see the follwing lines in terminal
INFO:     Started server process [<some_number>]
INFO:     Waiting for application startup.
INFO:     Application startup complete.

### then we proceed further

### open a new terminal and run app.py using below command:

python3 -m streamlit run app.py





