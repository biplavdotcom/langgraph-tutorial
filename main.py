from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_community.document_loaders import WebBaseLoader
import getpass
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tiktoken
from dotenv import load_dotenv
load_dotenv()

##########################################
# FETCH DOCUMENT USING THE DOCUMENT LOADER 
urls = [
    "https://lilianweng.github.io/posts/2024-11-28-reward-hacking/",
    "https://lilianweng.github.io/posts/2024-07-07-hallucination/",
    "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/",
]

docs = [WebBaseLoader(url).load() for url in urls]
docs[0][0].page_content.strip()[:1000]

#################################
# SPLIT THE DOCUMENTS INTO CHUNKS

docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size = 100, chunk_overlap = 50
)
docs_chunks = text_splitter.split_documents(docs_list)

docs_chunks[0].page_content.strip()[:500]

##############################################
# INDEXING IN VECTOR STORE FOR SEMANTIC SEARCH
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

vector_store = InMemoryVectorStore.from_documents(
    documents = docs_chunks,
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
)

##############################
#RETREIVER FOR SEMANTIC SEARCH

retriever = vector_store.as_retriever()

#########################################
# USING LANGCHAIN PREBUILT RETRIEVER TOOL
from langchain_classic.tools.retriever import create_retriever_tool

retriever_tool = create_retriever_tool(
    retriever = retriever,
    name = "retrieve_blog_posts",
    description = "Search and return information about Lilian Weng blog posts.",
)

found_text = retriever_tool.invoke({"query": "types of hallucinations"})

###############
# BUILDING NODE
from langgraph.graph import MessagesState
from langchain.chat_models import init_chat_model
from langchain_groq import ChatGroq

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GOOGLE_API = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ")
print(GROQ_API_KEY)

# response_model = init_chat_model("google_genai:gemini-2.0-flash-lite", max_retries = 2, timeout = 60, api_key = GEMINI_API_KEY)

# response_model = ChatGoogleGenerativeAI(
#     model = "gemini-2.0-flash-lite",
#     api_key = GEMINI_API_KEY
# )

response_model = ChatGroq(
    api_key = GROQ_API_KEY,
    model = "openai/gpt-oss-20b"
)

def generate_query_or_respond(state: MessagesState) -> str:
    """Call the model to generate a response based on the current state. Given the question, it will decide to retrieve using the retriever tool, or simply respond to the user.
    """
    response = response_model.bind_tools([retriever_tool]).invoke(state["messages"])

    return {
        "messages": [response]
    }

input = {
    "messages": [
        {
            "role": "user",
            "content": "Hello!"
        }
    ]
}

generate_query_or_respond(input)["messages"][-1].pretty_print()

input = {
    "messages": [
        {
            "role": "user",
            "content": "What does Lilian Weng say about types of reward hacking?"
        }
    ]
}

generate_query_or_respond(input)["messages"][-1].pretty_print()

##########################################################
# CONDITIONAL EDGES AND FUNCTION WILL RETURN NAME OF NODE TO GO TO BASED ON THE GRADING DECISION

from pydantic import BaseModel, Field
from typing import Literal

GRADE_PROMPT = (
    "You are a grader assessing relevance of a retrieved document to a user question. \n "
    "Here is the retrieved document: \n\n {context} \n\n"
    "Here is the user question: {question} \n"
    "If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n"
    "Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."
)

class GradeDocuments(BaseModel):
    """Grade documents using a binary score for relevance check."""
    binary_score: str = Field(
        description="Relevance score: 'yes' if relevant, or 'no' if not relevant"

    )

grader_model = ChatGroq(
    api_key = GROQ_API_KEY,
    model = "openai/gpt-oss-20b",
    temperature = 0
)

def grade_documents(state: MessagesState) -> str:
    """Determine whether the retrieved documents are relevant to the question."""

    question = state["messages"][0].content
    context = state["messages"][-1].content
    
    prompt = GRADE_PROMPT.format(question=question, context=context)
    
    response = (
        grader_model.with_structured_output(GradeDocuments).invoke([
            {"role": "user", "content": prompt}
        ])
    )
    
    score = response.binary_score
    if score == "yes":
        return "generate_answer"
    else:
        return "rewrite_question"

from langchain_core.messages import convert_to_messages
input = {
    "messages": convert_to_messages(
        [
            {
                "role": "user",
                "content": "What does Lilian Weng say about types of reward hacking?",
            },
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "1",
                        "name": "retrieve_blog_posts",
                        "args": {
                            "query": "types of reward hacking"
                        }
                    }
                ]
            },
            {
                "role": "tool",
                "content": "meow",
                "tool_call_id": "1"
            },
        ]
    )
}

grade_documents(input)

input = {
    "messages": convert_to_messages(
        [
            {
                "role": "user",
                "content": "What does Lilian Weng say about types of reward hacking?",
            },
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "1",
                        "name": "retrieve_blog_posts",
                        "args": {"query": "types of reward hacking"},
                    }
                ],
            },
            {
                "role": "tool",
                "content": "reward hacking can be categorized into two types: environment or goal misspecification, and reward tampering",
                "tool_call_id": "1",
            },
        ]
    )
}
grade_documents(input)