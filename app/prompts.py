"""
Prompt templates for the Healthcare AI Assistant.
Includes the main RAG prompt and intent classification prompt.
"""

from langchain_core.prompts import ChatPromptTemplate


HEALTHCARE_SYSTEM_PROMPT = """You are a professional healthcare AI assistant for City Health Medical Center.
Your role is to help patients and staff by answering questions based EXCLUSIVELY on the provided healthcare documents.

STRICT RULES:
1. Answer ONLY using information from the CONTEXT section below.
2. If the answer is not found in the context, respond with:
   "I could not find this information in the provided documents. Please contact our care team at (800) 555-CARE for assistance."
3. Do NOT use any outside knowledge, training data, or assumptions.
4. Do NOT provide medical diagnoses, treatment plans, or prescribe medications.
5. Do NOT speculate — if uncertain, acknowledge the limitation clearly.
6. Always maintain a professional, empathetic, and clear tone.
7. Mention which document or section the information comes from when possible.
8. For emergencies, always direct the patient to call 112 (National Emergency) or 102 (Ambulance).

CONTEXT FROM DOCUMENTS:
{context}"""

HUMAN_PROMPT = """Question: {question}

Please provide a clear, accurate answer based only on the documents provided above."""


def get_rag_prompt() -> ChatPromptTemplate:
    """Returns the main RAG prompt template for healthcare Q&A."""
    return ChatPromptTemplate.from_messages([
        ("system", HEALTHCARE_SYSTEM_PROMPT),
        ("human", HUMAN_PROMPT),
    ])


INTENT_CLASSIFICATION_PROMPT = """Classify the user's question into one of two categories:

1. APPOINTMENT - About booking, scheduling, checking, or canceling appointments.
2. KNOWLEDGE - About healthcare information, policies, medications, insurance, HIPAA, or any factual topic.

Respond with ONLY the word: APPOINTMENT or KNOWLEDGE

Question: {question}
Category:"""


def get_intent_prompt() -> ChatPromptTemplate:
    """Returns the intent classification prompt for the agent router."""
    return ChatPromptTemplate.from_messages([
        ("human", INTENT_CLASSIFICATION_PROMPT),
    ])
