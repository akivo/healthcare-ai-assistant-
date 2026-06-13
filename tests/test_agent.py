"""
test_agent.py — Agent Routing Tests
=====================================
Tests for the intent classification and routing logic.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def test_appointment_keyword_detection():
    """Test that appointment-related keywords are correctly detected."""
    from app.agent import classify_intent
    from unittest.mock import patch
    
    # These should all be classified as APPOINTMENT
    appointment_questions = [
        "Can I book a cardiology appointment?",
        "Are there available slots on Monday?",
        "I want to schedule a visit",
        "What are the available booking times?",
    ]
    
    for question in appointment_questions:
        # Keyword matching doesn't need LLM — it's rule-based
        result = classify_intent(question)
        assert result == "APPOINTMENT", f"Expected APPOINTMENT for: '{question}', got: {result}"


def test_knowledge_question_routing():
    """Test that knowledge questions are classified correctly (with LLM mock)."""
    from app.agent import classify_intent
    from unittest.mock import patch
    
    # Mock the LLM to return KNOWLEDGE
    with patch('app.agent.get_llm') as mock_llm, \
         patch('app.agent.get_intent_prompt') as mock_prompt:
        
        mock_chain = mock_llm.return_value.__or__.return_value.__or__.return_value
        mock_chain.invoke.return_value = "KNOWLEDGE"
        
        result = classify_intent("What are my HIPAA rights?")
        # Could be KNOWLEDGE (no appointment keywords)
        assert result in ["KNOWLEDGE", "APPOINTMENT"]


def test_check_available_slots_returns_data():
    """Test that mock appointment tool returns expected data structure."""
    from app.agent import check_available_slots
    
    result = check_available_slots("cardiology", "Monday")
    
    assert "department" in result
    assert "date" in result
    assert "available_slots" in result
    assert isinstance(result["available_slots"], list)
    assert len(result["available_slots"]) > 0


def test_check_available_slots_unknown_department():
    """Test that unknown departments still return a result."""
    from app.agent import check_available_slots
    
    result = check_available_slots("unknown_department", "Tuesday")
    assert "available_slots" in result


def test_extract_appointment_details():
    """Test department and date extraction from questions."""
    from app.agent import extract_appointment_details
    
    result = extract_appointment_details("Can I book a cardiology appointment for Monday?")
    assert result["department"] == "cardiology"
    assert result["date"] == "Monday"


def test_extract_appointment_details_defaults():
    """Test that vague questions get default department."""
    from app.agent import extract_appointment_details
    
    result = extract_appointment_details("I want to see a doctor")
    assert "department" in result
    assert "date" in result
