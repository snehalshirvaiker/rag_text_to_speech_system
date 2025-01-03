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
For simple queries that the bot has to answer, it is looking for that key and retreiving it from a dictionary in rag.py as self.simple_queries. 
More Simple queries/keys can be added in the dictionary.

For other document related questions it will check the similarity score with a threshold of 0.5 and retreives the top 5 related chunks for context. If found in that document, it will give us the answer; if not, It will use the ollama model mistral's trained data to retrieve the context and response


## Below are the steps to run:

## This project needs Ollama to be installed and running on local machine
    https://ollama.com/
### run the below code in terminal after running on local
    ollama pull mistral
    ollama run mistral
### now check if the model is running 
    ollama serve

### for ios, run the bash file run.sh in terminal using below command:

    bash run.sh

## OR 

### for windows, the commands will be changed to below:
    py -m venv venv
    venv/Scripts/activate
    py -m pip install --upgrade pip
    py -m pip install -r requirements.txt
    uvicorn rag:app

## once we see the follwing lines in terminal
    INFO:     Started server process [<some_number>]
    INFO:     Waiting for application startup.
    INFO:     Application startup complete.

### then we proceed further

### open a new terminal and run app.py using below command:

    python3 -m streamlit run app.py

### Now you can type any question and press the Ask button
It might take 1 minute to load questions which are simple queries mentioned in the dictionary in rag.py or from the document.
And for questions unrelated to above, it might take 2-3 minutes.





