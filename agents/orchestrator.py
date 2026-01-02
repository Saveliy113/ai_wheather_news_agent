import json

from typing import Dict, Any
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from mcp_config.weather_mcp import WeatherMCP
from mcp_config.news_mcp import NewsMCP

load_dotenv()


class Orchestrator:
    """
    Orchestrator with Intent Agent via OpenAI
    """

    def __init__(self, model_name="gpt-3.5-turbo", temperature=0):
        # Use init_chat_model which works with newer LangChain versions
        self.llm = init_chat_model(model_name, temperature=temperature)
        
        # Initialize MCP servers
        self.weather_mcp = WeatherMCP()
        self.news_mcp = NewsMCP()

        # Prompt for Intent Agent
        self.intent_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                "You are an assistant that classifies user queries about weather and news. "
                "Return ONLY valid JSON, no markdown, no code blocks, no explanations. "
                "For weather queries, also parse the time reference into a day_index. "
                "Day index: 0 = today/now, 1 = tomorrow, 2 = day after tomorrow, 3 = in 3 days, etc. (max 6). "
                "If time is not specified or unclear, day_index should be null. "
                "JSON format: "
                '{{"intent": "weather" | "news" | "both", "location": "string or null", "topic": "string or null", "time": "string or null", "day_index": number (0-6) or null}}'
            ),
            HumanMessagePromptTemplate.from_template("{user_input}")
        ])
        
        # Prompt for Response Generation
        self.response_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                "You are a helpful assistant that provides clear, natural responses about weather and news. "
                "Based on the user's question and the data provided, give a direct, conversational answer that directly addresses what they asked. "
                "Be specific and use the actual data values from the provided data. "
                "If the user asks a yes/no question (like 'Will it rain?'), answer directly with yes/no and then provide details. "
                "If asking about precipitation/rain, check the 'precipitation' field in the data. "
                "If asking about temperature, use the temperature values from the data. "
                "Match the time period in your response to what the user asked about (today, tomorrow, etc.)."
            ),
            HumanMessagePromptTemplate.from_template(
                "User question: {user_question}\n\n"
                "Weather data: {data}\n\n"
                "Provide a natural, conversational response that directly answers the user's question:"
            )
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
            messages = self.intent_prompt.format_messages(user_input=user_input)

            # Step 2: Call LLM using invoke() method
            response = self.llm.invoke(messages)

            # Step 3: Get response text (should be pure JSON)
            response_text = response.content.strip()
            
            # Step 4: Parse JSON
            intent_data = json.loads(response_text)
            
            # Step 5: Handle queries using MCP servers
            intent = intent_data.get("intent", "unknown")
            location = intent_data.get("location")
            topic = intent_data.get("topic")
            time_ref = intent_data.get("time")
            day_index = intent_data.get("day_index")
            
            result = {
                "intent": intent,
                "location": location,
                "topic": topic,
                "time": time_ref,
                "day_index": day_index,
                "success": True,
                "user_question": user_input
            }
            
            # Handle weather queries
            if intent in ["weather", "both"] and location:
                weather_result = self.weather_mcp.get_weather(location, day_index=day_index)
                result["weather_data"] = weather_result
                
                # Generate natural language response using OpenAI
                if weather_result.get("success"):
                    response_text = self._generate_weather_response(user_input, weather_result)
                    result["response"] = response_text
            
            # Handle news queries
            if intent in ["news", "both"]:
                news_result = self.news_mcp.get_news(topic=topic)
                result["news_data"] = news_result
                
                # Generate natural language response using OpenAI
                if news_result.get("success"):
                    response_text = self._generate_news_response(user_input, news_result)
                    result["response"] = response_text
            
            return result
            
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
    
    def _generate_weather_response(self, user_question: str, weather_data: Dict[str, Any]) -> str:
        """
        Generate natural language response for weather queries using OpenAI.
        
        Args:
            user_question: Original user question
            weather_data: Weather data from MCP server
            
        Returns:
            Natural language response string
        """
        try:
            messages = self.response_prompt.format_messages(
                user_question=user_question,
                data=json.dumps(weather_data, indent=2)
            )
            response = self.llm.invoke(messages)
            return response.content.strip()
        except Exception as e:
            # Fallback to simple response if generation fails
            return f"Weather data retrieved, but failed to generate response: {str(e)}"
    
    def _generate_news_response(self, user_question: str, news_data: Dict[str, Any]) -> str:
        """
        Generate natural language response for news queries using OpenAI.
        
        Args:
            user_question: Original user question
            news_data: News data from MCP server
            
        Returns:
            Natural language response string
        """
        try:
            messages = self.response_prompt.format_messages(
                user_question=user_question,
                data=json.dumps(news_data, indent=2)
            )
            response = self.llm.invoke(messages)
            return response.content.strip()
        except Exception as e:
            # Fallback to simple response if generation fails
            return f"News data retrieved, but failed to generate response: {str(e)}"