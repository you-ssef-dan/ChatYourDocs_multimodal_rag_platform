from langchain.prompts import PromptTemplate

prompt_template = """
You are a visual AI assistant. Analyze both the TEXT CONTEXT and IMAGES to answer the question.
Pay special attention to visual details in images. When asked about something related to images.
Use ONLY the provided context to answer the question.
If the answer is not contained in the context, respond with: "I don't know."

TEXT CONTEXT:
{context}

QUESTION:
{question}

Answer based on ALL available information:
"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)
