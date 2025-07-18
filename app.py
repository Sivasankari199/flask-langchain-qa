from flask import Flask, render_template, request
import os
import tempfile
import json
import boto3
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

# Load variables from .env
load_dotenv()

# LangChain + OpenAI + Chroma components
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.base import Embeddings
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

# Load environment variables 
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'pdf'}
S3_BUCKET = os.getenv("S3_BUCKET")
AWS_REGION = os.getenv("AWS_REGION")
SAGEMAKER_ENDPOINT = os.getenv("SAGEMAKER_ENDPOINT")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHROMA_DIR = os.path.join(tempfile.gettempdir(), "chroma_db_cohere")

# Flask app setup 
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

#  AWS Clients 
s3 = boto3.client('s3', region_name=AWS_REGION)
sm_runtime = boto3.client('sagemaker-runtime', region_name=AWS_REGION)

# File extension check 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Custom SageMaker Embedding Class 
class SageMakerCohereEmbeddings(Embeddings):
    def __init__(self):
        self.endpoint_name = SAGEMAKER_ENDPOINT
        self.sm_runtime = sm_runtime

    def embed_documents(self, texts):
        BATCH_SIZE = 48
        all_embeddings = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i:i + BATCH_SIZE]
            payload = {"texts": batch, "input_type": "search_document"}
            response = self.sm_runtime.invoke_endpoint(
                EndpointName=self.endpoint_name,
                ContentType="application/json",
                Body=json.dumps(payload)
            )
            result = json.loads(response['Body'].read().decode('utf-8'))
            all_embeddings.extend(result["embeddings"]["float"])
        return all_embeddings

    def embed_query(self, text):
        return self.embed_documents([text])[0]

# Main Route 
@app.route('/', methods=['GET', 'POST'])
def index():
    answer = None

    if request.method == 'POST':
        file = request.files.get('pdf_file')
        query = request.form.get('query')

        if file and allowed_file(file.filename) and query:
            filename = secure_filename(file.filename)

            # Step 1: Upload to S3
            s3.upload_fileobj(file, S3_BUCKET, filename)

            # Step 2: Download from S3 to local
            local_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            s3.download_file(Bucket=S3_BUCKET, Key=filename, Filename=local_path)

            # Step 3: Load and chunk PDF
            loader = PyPDFLoader(local_path)
            pages = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
            chunks = splitter.split_documents(pages)

            # Step 4: Clear ChromaDB dir to avoid dim mismatch
            if os.path.exists(CHROMA_DIR):
                import shutil
                shutil.rmtree(CHROMA_DIR)

            # Step 5: Create embedding model and vector store
            embedding_model = SageMakerCohereEmbeddings()
            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embedding_model,
                persist_directory=CHROMA_DIR
            )
            retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

            # Step 6: Prompt + OpenAI LLM
            QA_PROMPT = PromptTemplate(
                template="""Use the context below to answer the question.

Context:
{context}

Question: {question}

Helpful Answer:""",
                input_variables=["context", "question"]
            )
            llm = ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=0.2,
                openai_api_key=OPENAI_API_KEY
            )

            # Step 7: Build Retrieval-Based QA Chain
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=retriever,
                chain_type="stuff",
                chain_type_kwargs={"prompt": QA_PROMPT}
            )

            result = qa_chain.invoke({"query": query})
            answer = result["result"]

        else:
            answer = "Please upload a valid PDF and enter your question."

    return render_template('index.html', answer=answer)

# Run the Flask App
if __name__ == '__main__':
    app.run(debug=True)
