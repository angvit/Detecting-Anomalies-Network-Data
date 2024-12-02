import os
import sys
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain_community.document_loaders import TextLoader
from langchain_community.chat_models import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate
import gradio as gr
from dotenv import load_dotenv

# Load the API key from the .env file
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if api_key is None:
    raise ValueError("OPENAI_API_KEY is not set in the .env file.")
os.environ["OPENAI_API_KEY"] = api_key

# Load the text files from the rag_data folder
rag_data_dir = "./rag_data/"
sys.path.append(rag_data_dir)

# Initialize document loaders
loaders = []
for filename in os.listdir(rag_data_dir):
    file_path = os.path.join(rag_data_dir, filename)
    try:
        loaders.append(TextLoader(file_path, encoding="utf-8"))
    except Exception as e:
        print(f"Failed to load: {filename}. Error: {e}")

# Load and merge documents
data = []
for loader in loaders:
    try:
        data.extend(loader.load())
    except Exception as e:
        print(f"Error loading data from {loader}: {e}")

# Merge the documents
merged_documents = [Document(page_content=" ".join([doc.page_content for doc in data]))]

# Initialize embedding model from Hugging Face
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Initialize Chroma vector store to store the embedding data
persist_directory = "./chroma_db"  # Directory to store the Chroma index
vectorstore = Chroma.from_documents(merged_documents, embeddings, persist_directory=persist_directory)
vectorstore.persist()  # Save the database to disk

# Custom RAG prompt template
template = """
You are a teaching assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question at the end.
The following pieces of retrieved context are from a Network Security textbook.
The name of the textbook is Computer Security: Principles and Practice, 4th Edition.
If you don't know the answer, just say that you don't know. Don't try to make up an answer.
Don't say anything mean or offensive.

Context: {context}

Question: {question}
"""
custom_rag_prompt = ChatPromptTemplate.from_template(template)

# Initialize the ChatOpenAI model
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",  # Specify the model
    temperature=0.2  # Control the creativity of the response
)

# Regular chain format: chain = prompt | model
rag_chain = (
    {"context": vectorstore.as_retriever(), "question": RunnablePassthrough()}  # Use vector store as retriever
    | custom_rag_prompt
    | llm
)

# Function to handle RAG chain response with detailed logging
def get_response(query):
    try:
        # Get the response from the RAG chain
        response = rag_chain.invoke({"question": query})

        # Log the full response for debugging
        print("Full Response:", response)

        # If the response is a dictionary, we need to inspect its structure
        if isinstance(response, dict):
            if 'text' in response:
                return response['text']
            else:
                return f"Unexpected response format: {response}"

        elif isinstance(response, str):
            return response

        else:
            return f"Unexpected response type: {type(response)}"

    except Exception as e:
        return f"Error processing your request: {e}"

# Gradio interface
def gradio_interface(query):
    return get_response(query)

# Create the Gradio app
with gr.Blocks() as demo:
    gr.Markdown("### Ask a question about a Network Attack")
    question_input = gr.Textbox(label="Question:")
    answer_output = gr.Textbox(label="Answer:", lines=5)
    submit_button = gr.Button("Submit")
    
    submit_button.click(gradio_interface, inputs=question_input, outputs=answer_output)

demo.launch(share=True)
