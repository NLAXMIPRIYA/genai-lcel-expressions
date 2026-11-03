# Design and Implementation of LangChain Expression Language (LCEL) Expressions

## AIM:

To design and implement LangChain Expression Language (LCEL) expressions using prompt templates, a language model, and an output parser. The implementation demonstrates a simple LCEL chain and a complex Retrieval-Augmented Generation (RAG) chain using multiple components. The functionality is evaluated using practical examples involving joke generation and document-based question answering.

## PROBLEM STATEMENT:

To develop an AI-powered application using LangChain Expression Language (LCEL) that demonstrates how multiple components can be connected to process user inputs and generate meaningful responses.

The implementation consists of two chains:

1. **Simple LCEL Chain:** Accepts a topic as input and generates a short joke using a language model.
2. **Complex LCEL Chain using RAG:** Retrieves relevant information from a vector store based on a user's question and generates an answer using the retrieved context.

The complex chain uses two prompt parameters, `context` and `question`, along with three key components: prompt template, language model, and output parser.

## DESIGN STEPS:

### STEP 1: Import Libraries and Initialize the Language Model

- Import the required Python and LangChain libraries.
- Load the OpenAI API key securely using environment variables.
- Initialize the ChatOpenAI model using `gpt-4o-mini` with temperature set to 0.
- Initialize `StrOutputParser()` to convert model responses into plain text.

### STEP 2: Design and Implement the Simple LCEL Chain

- Create a prompt template using `ChatPromptTemplate.from_template()`.
- Define `{topic}` as the prompt parameter.
- Connect the prompt, language model, and output parser using the LCEL pipe operator (`|`).
- Execute the chain using `invoke()` with a sample topic such as "bears".
- Display the generated joke.

### STEP 3: Design and Implement the Complex RAG Chain

- Import `OpenAIEmbeddings` and `DocArrayInMemorySearch`.
- Create a vector store containing sample documents:
  - "Harrison worked at Kensho"
  - "Bears like to eat honey"
- Convert the vector store into a retriever using `as_retriever()`.
- Test the retriever with sample questions to retrieve relevant documents.
- Create a prompt template with two parameters: `{context}` and `{question}`.
- Use `RunnableMap` to retrieve relevant context and preserve the original question.
- Connect the input mapping, prompt, language model, and output parser using the LCEL pipe operator.
- Execute the complex chain using `invoke()` and display the final answer.

## PROGRAM:

```python
# STEP 1: Import libraries and load API key

import os
import openai

from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())
openai.api_key = os.environ['OPENAI_API_KEY']


# STEP 2: Import LangChain components

from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser


# STEP 3: Initialize the model

model = ChatOpenAI(
    model_name="gpt-4o-mini",
    temperature=0
)


# STEP 4: Initialize the output parser

output_parser = StrOutputParser()


# ==========================================
# PART A: SIMPLE LCEL CHAIN
# ==========================================

# STEP 5: Create a prompt template

prompt = ChatPromptTemplate.from_template(
    "Tell me a short joke about {topic}"
)


# STEP 6: Build the simple chain

chain = prompt | model | output_parser


# STEP 7: Execute the simple chain

simple_result = chain.invoke({"topic": "bears"})

print("SIMPLE CHAIN OUTPUT:")
print(simple_result)


# ==========================================
# PART B: COMPLEX LCEL CHAIN USING RAG
# ==========================================

# STEP 8: Import embedding and vector store components

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import DocArrayInMemorySearch


# STEP 9: Create a vector store

vectorstore = DocArrayInMemorySearch.from_texts(
    [
        "Harrison worked at Kensho",
        "Bears like to eat honey"
    ],
    embedding=OpenAIEmbeddings()
)


# STEP 10: Create a retriever

retriever = vectorstore.as_retriever()


# STEP 11: Test the retriever

print("\nRETRIEVER OUTPUT:")

print(
    retriever.get_relevant_documents(
        "Where did Harrison work?"
    )
)

print(
    retriever.get_relevant_documents(
        "What do bears like to eat?"
    )
)


# STEP 12: Create the RAG prompt template

template = """
Answer the question based only on the following context:

{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)


# STEP 13: Import RunnableMap

from langchain.schema.runnable import RunnableMap


# STEP 14: Create the input mapping

inputs = RunnableMap({
    "context": lambda x: retriever.get_relevant_documents(
        x["question"]
    ),
    "question": lambda x: x["question"]
})


# STEP 15: Inspect the intermediate input

print("\nINTERMEDIATE INPUT:")
print(
    inputs.invoke({
        "question": "Where did Harrison work?"
    })
)


# STEP 16: Build the complete complex RAG chain

complex_chain = (
    inputs
    | prompt
    | model
    | output_parser
)


# STEP 17: Execute the complex chain

complex_result = complex_chain.invoke({
    "question": "Where did Harrison work?"
})

print("\nCOMPLEX RAG CHAIN OUTPUT:")
print(complex_result)
```

## OUTPUT:

### 1. Simple LCEL Chain Output

```text
SIMPLE CHAIN OUTPUT:
Why do bears have hairy coats?

Because they look silly in sweaters!
```

### 2. Retriever Output

For the question "Where did Harrison work?":

```text
[
    Document(page_content='Harrison worked at Kensho'),
    Document(page_content='Bears like to eat honey')
]
```

For the question "What do bears like to eat?":

```text
[
    Document(page_content='Bears like to eat honey'),
    Document(page_content='Harrison worked at Kensho')
]
```

### 3. Intermediate Input

```text
INTERMEDIATE INPUT:

{
    'context': [
        Document(page_content='Harrison worked at Kensho'),
        Document(page_content='Bears like to eat honey')
    ],
    'question': 'Where did Harrison work?'
}
```

### 4. Complex RAG Chain Output

```text
COMPLEX RAG CHAIN OUTPUT:
Harrison worked at Kensho.
```


<img width="641" height="307" alt="image" src="https://github.com/user-attachments/assets/f741b02b-f6af-4ced-be12-28382c6bb62f" />


## RESULT:

Thus, the LangChain Expression Language (LCEL) expressions were successfully designed and implemented using prompt templates, the ChatOpenAI language model, and the StrOutputParser.

The simple LCEL chain successfully generated a joke based on the given topic. The complex RAG chain successfully retrieved relevant documents from the vector store using embeddings and generated an accurate answer based on the retrieved context.

The implementation demonstrates how LCEL components can be combined to build both simple AI applications and complex retrieval-based question-answering systems.
