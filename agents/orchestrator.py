import json

from typing import Dict, Any
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)

load_dotenv()


class Orchestrator:
    """
    Orchestrator with Intent Agent via OpenAI
    """

    def __init__(self, model_name="gpt-3.5-turbo", temperature=0):
        # Use init_chat_model which works with newer LangChain versions
        self.llm = init_chat_model(model_name, temperature=temperature)

        # Prompt for Intent Agent
        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                "You are an assistant that classifies user queries about weather and news. "
                "Return ONLY valid JSON, no markdown, no code blocks, no explanations. "
                "JSON format: "
                '{{"intent": "weather" | "news" | "both", "location": "string or null", "topic": "string or null", "time": "string or null"}}'
            ),
            HumanMessagePromptTemplate.from_template("{user_input}")
        ])

    def run(self, user_input: str) -> Dict[str, Any]:
        """
        Process user input and return parsed intent classification.
        
        Args:
            user_input: User's query string
            
        Returns:
            Dictionary with intent, location, topic, and time fields
        """
        try:
            # Step 1: Create messages from prompt template
            messages = self.prompt.format_messages(user_input=user_input)

            # Step 2: Call LLM using invoke() method
            response = self.llm.invoke(messages)

            # Step 3: Get response text (should be pure JSON)
            response_text = response.content.strip()
            
            # Step 4: Parse JSON
            result = json.loads(response_text)
            
            # Step 5: Return structured result
            return {
                "intent": result.get("intent", "unknown"),
                "location": result.get("location"),
                "topic": result.get("topic"),
                "time": result.get("time"),
                "success": True
            }
            
        except json.JSONDecodeError as e:
            # JSON parsing failed
            return {
                "intent": "unknown",
                "location": None,
                "topic": None,
                "time": None,
                "success": False,
                "error": f"Failed to parse JSON: {str(e)}",
                "raw_response": response.content
            }
        except Exception as e:
            # Any other error
            return {
                "intent": "unknown",
                "location": None,
                "topic": None,
                "time": None,
                "success": False,
                "error": f"Error: {str(e)}"
            }