"""
Agentic routing workflow.
Classifies user intent and routes questions to either the
RAG pipeline or the mock appointment scheduling tool.
"""

import logging
import random
from typing import Dict, Any

from app.llm import get_llm
from app.prompts import get_intent_prompt
from langchain_core.output_parsers import StrOutputParser

logger = logging.getLogger(__name__)

# Keywords that indicate appointment-related queries
APPOINTMENT_KEYWORDS = [
    "book", "booking", "appointment", "schedule", "scheduling",
    "available slots", "available times", "open slots", "visit time",
    "can i come in", "when can i see", "see a doctor",
    "cardiology appointment", "dermatology appointment",
    "orthopedics appointment", "pediatrics appointment",
    "next available", "earliest appointment", "book a slot",
]


def classify_intent(question: str) -> str:
    """
    Classify question intent as APPOINTMENT or KNOWLEDGE.

    Uses keyword matching first (fast), falls back to LLM classification
    for ambiguous cases.
    """
    question_lower = question.lower()

    # Fast keyword check
    for keyword in APPOINTMENT_KEYWORDS:
        if keyword in question_lower:
            logger.info(f"Intent: APPOINTMENT (keyword: '{keyword}')")
            return "APPOINTMENT"

    # LLM classification for ambiguous questions
    logger.info("Using LLM for intent classification...")
    try:
        chain = get_intent_prompt() | get_llm() | StrOutputParser()
        result = chain.invoke({"question": question}).strip().upper()
        intent = "APPOINTMENT" if "APPOINTMENT" in result else "KNOWLEDGE"
        logger.info(f"LLM classified intent as: {intent}")
        return intent
    except Exception as e:
        logger.warning(f"LLM classification failed: {e}. Defaulting to KNOWLEDGE.")
        return "KNOWLEDGE"


def extract_appointment_details(question: str) -> Dict[str, str]:
    """Extract department and date from an appointment-related question."""
    question_lower = question.lower()

    departments = {
        "cardiology": ["cardiology", "heart", "cardiac", "cardiologist"],
        "dermatology": ["dermatology", "skin", "dermatologist"],
        "general_medicine": ["general", "primary", "gp", "family doctor"],
        "orthopedics": ["orthopedics", "ortho", "bone", "joint"],
        "pediatrics": ["pediatrics", "pediatric", "child", "children", "kids"],
        "psychiatry": ["psychiatry", "mental health", "psychiatrist", "therapy"],
        "gynecology": ["gynecology", "gynecologist", "obgyn", "women's health"],
    }

    detected_department = "general_medicine"
    for dept, keywords in departments.items():
        if any(kw in question_lower for kw in keywords):
            detected_department = dept
            break

    days = {
        "monday": "Monday", "tuesday": "Tuesday", "wednesday": "Wednesday",
        "thursday": "Thursday", "friday": "Friday", "saturday": "Saturday",
        "sunday": "Sunday", "today": "Today", "tomorrow": "Tomorrow",
        "next week": "Next Week",
    }

    detected_date = "this week"
    for day_key, day_name in days.items():
        if day_key in question_lower:
            detected_date = day_name
            break

    return {"department": detected_department, "date": detected_date}


def check_available_slots(department: str, date: str) -> Dict[str, Any]:
    """
    Mock appointment availability tool.

    Returns simulated slot data. In production, this would connect
    to a real EHR scheduling system (e.g., Epic, Cerner).
    """
    logger.info(f"Mock tool: check_available_slots({department}, {date})")

    mock_slots = {
        "cardiology": ["9:00 AM", "11:30 AM", "2:00 PM"],
        "dermatology": ["10:00 AM", "1:00 PM", "3:30 PM"],
        "general_medicine": ["8:30 AM", "10:00 AM", "11:30 AM", "2:00 PM", "4:00 PM"],
        "orthopedics": ["9:30 AM", "12:00 PM", "2:30 PM"],
        "pediatrics": ["8:00 AM", "9:30 AM", "11:00 AM", "2:30 PM"],
        "psychiatry": ["10:00 AM", "11:30 AM", "2:00 PM"],
        "gynecology": ["9:00 AM", "10:30 AM", "1:00 PM", "3:00 PM"],
    }

    available = mock_slots.get(department.lower(), ["9:00 AM", "2:00 PM"])
    if len(available) > 2:
        available = random.sample(available, len(available) - 1)

    return {
        "department": department.replace("_", " ").title(),
        "date": date,
        "available_slots": available,
        "booking_contact": "(800) 555-CARE",
        "portal_link": "myhealthportal.cityhealthmc.org",
        "note": "Please confirm availability by calling (800) 555-CARE or via our patient portal.",
    }


def format_appointment_response(slots_data: Dict[str, Any]) -> Dict[str, Any]:
    """Format mock slot data into the standard API response structure."""
    dept = slots_data["department"]
    date = slots_data["date"]
    slots = slots_data["available_slots"]

    if slots:
        slots_str = ", ".join(slots)
        answer = (
            f"I checked our appointment system for {dept}. "
            f"Available slots on {date}: {slots_str}. "
            f"To confirm and book, please call {slots_data['booking_contact']} "
            f"or visit {slots_data['portal_link']}.\n\n"
            f"Note: {slots_data['note']}"
        )
    else:
        answer = (
            f"No available slots found for {dept} on {date}. "
            f"Please call {slots_data['booking_contact']} for assistance."
        )

    return {
        "answer": answer,
        "sources": [{
            "document": "mock_appointment_system",
            "chunk": f"Available slots for {dept} on {date}: {slots}",
            "relevance_score": 1.0,
        }],
        "confidence": "high",
        "route": "appointment_tool",
        "tool_data": slots_data,
    }


def process_question(question: str) -> Dict[str, Any]:
    """
    Main agent entry point. Classifies intent and routes to the
    appropriate handler (RAG pipeline or appointment tool).

    Args:
        question: User's input question

    Returns:
        Standardized response dict with answer, sources, confidence, route
    """
    logger.info(f"Agent processing: {question[:80]}...")

    intent = classify_intent(question)

    if intent == "APPOINTMENT":
        logger.info("Routing to appointment tool.")
        details = extract_appointment_details(question)
        slots_data = check_available_slots(
            department=details["department"],
            date=details["date"],
        )
        return format_appointment_response(slots_data)

    logger.info("Routing to RAG pipeline.")
    from app.rag import run_rag_query
    return run_rag_query(question)
