import json

from typing import Dict, Any, Optional
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder
)
# Custom ConversationBufferMemory implementation
from langchain_core.messages import HumanMessage, AIMessage
from typing import List


class ConversationBufferMemory:
    """
    Simple conversation memory implementation.
    Stores conversation history for multi-turn conversations.
    """
    def __init__(self, memory_key="chat_history", return_messages=True, input_key="input", output_key="output"):
        self.memory_key = memory_key
        self.return_messages = return_messages
        self.input_key = input_key
        self.output_key = output_key
        self.chat_history: List = []
    
    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Load conversation history."""
        return {self.memory_key: self.chat_history}
    
    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]):
        """Save user input and AI response to memory."""
        user_input = inputs.get(self.input_key, inputs.get("user_question", ""))
        ai_output = outputs.get(self.output_key, outputs.get("response", ""))
        
        if user_input:
            self.chat_history.append(HumanMessage(content=str(user_input)))
        if ai_output:
            self.chat_history.append(AIMessage(content=str(ai_output)))
    
    def clear(self):
        """Clear conversation history."""
        self.chat_history = []


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
        
        # Initialize conversation memory for multi-turn conversations
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            input_key="user_question",
            output_key="response"
        )

        # Prompt for Intent Agent with conversation history support
        self.intent_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                "You are an assistant that classifies user queries about weather and news. "
                "You have access to conversation history to understand follow-up questions. "
                "Return ONLY valid JSON, no markdown, no code blocks, no explanations. "
                "IMPORTANT: "
                "- Use conversation history to understand follow-up questions. If user asks 'and tomorrow?' or 'what about that?', use previous context. "
                "- If the query is NOT about weather or news (e.g., asking about cars, recipes, general questions), set intent to 'unknown'. "
                "- 'location' is ONLY for weather queries (geographic places like cities, countries: 'Paris', 'New York', 'Almaty'). "
                "- 'topic' is ONLY for news queries (what the news is about: 'Russia', 'technology', 'sports', 'politics'). "
                "- For weather queries, also parse the time reference into a day_index. "
                "Day index: 0 = today/now, 1 = tomorrow, 2 = day after tomorrow, 3 = in 3 days, etc. (max 6). "
                "If time is not specified or unclear, day_index should be null. "
                "Examples: "
                "- 'weather in Paris' -> intent: 'weather', location: 'Paris', topic: null "
                "- 'news about Russia' -> intent: 'news', location: null, topic: 'Russia' "
                "- 'and tomorrow?' (after weather query) -> intent: 'weather', location: from previous context, topic: null, day_index: 1 "
                "- 'Can you recommend me a car?' -> intent: 'unknown', location: null, topic: null "
                "JSON format: "
                '{{"intent": "weather" | "news" | "both" | "unknown", "location": "string or null", "topic": "string or null", "time": "string or null", "day_index": number (0-6) or null}}'
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{user_input}")
        ])
        
        # Prompt for Response Generation with conversation history
        self.response_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                "You are a helpful assistant that provides clear, natural responses about weather and news ONLY. "
                "CRITICAL: If the user's question is NOT about weather or news, you MUST respond with: "
                "'I'm a specialized assistant that provides information only about weather and news. I cannot answer questions about other topics. "
                "Please ask me about weather conditions or news information.' "
                "Based on the user's question and the data provided, give a direct, conversational answer that directly addresses what they asked. "
                "You have access to conversation history, so you can understand follow-up questions and references to previous topics. "
                "If the user asks a follow-up question (like 'what about tomorrow?' or 'show me news about that'), use the conversation history to understand the context. "
                "Be specific and use the actual data values from the provided data. "
                "For weather: If the user asks a yes/no question (like 'Will it rain?'), answer directly with yes/no and then provide details. "
                "If asking about precipitation/rain, check the 'precipitation' field in the data. "
                "If asking about temperature, use the temperature values from the data. "
                "Match the time period in your response to what the user asked about (today, tomorrow, etc.). "
                "For news: Summarize the articles naturally, mention key headlines and topics. "
                "ALWAYS include clickable links to the original articles using markdown format: [Article Title](url). "
                "Each article mentioned must have its corresponding URL link."
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template(
                "User question: {user_question}\n\n"
                "Data: {data}\n\n"
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
            # Step 1: Load conversation history for intent detection
            memory_variables = self.memory.load_memory_variables({})
            chat_history = memory_variables.get("chat_history", [])
            
            # Step 2: Create messages from prompt template with conversation history
            messages = self.intent_prompt.format_messages(
                chat_history=chat_history,
                user_input=user_input
            )

            # Step 3: Call LLM using invoke() method
            response = self.llm.invoke(messages)

            # Step 4: Get response text (should be pure JSON)
            response_text = response.content.strip()
            
            # Step 5: Parse JSON
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
            
            # Handle unknown/unrelated queries - return immediately without processing
            if intent == "unknown":
                response_text = (
                    "I'm a specialized assistant that provides information **only** about weather and news. "
                    "I cannot answer questions about other topics.\n\n"
                    "Please ask me about:\n"
                    "• **Weather**: Current conditions or forecasts for any location (e.g., 'What's the weather in Paris?' or 'Will it rain tomorrow in London?')\n"
                    "• **News**: Latest headlines or news about specific topics (e.g., 'Show me news about technology' or 'What's the news about Russia?')\n\n"
                    "How can I help you with weather or news information?"
                )
                result["response"] = response_text
                result["success"] = True
                
                # Save to memory even for unknown queries to maintain context
                self.memory.save_context(
                    {"user_question": user_input},
                    {"response": response_text}
                )
                
                return result
            
            # Handle weather queries
            weather_result = None
            if intent in ["weather", "both"]:
                if location:
                    weather_result = self.weather_mcp.get_weather(location, day_index=day_index)
                    result["weather_data"] = weather_result
                else:
                    # Weather intent but no location
                    result["weather_data"] = {
                        "success": False,
                        "error": "Please specify a location for weather information."
                    }
            
            # Handle news queries
            news_result = None
            if intent in ["news", "both"]:
                # Use topic for news search (not location)
                # If topic is None but location exists for news query, use location as topic
                news_topic = topic if topic else (location if intent == "news" else None)
                news_result = self.news_mcp.get_news(topic=news_topic)
                result["news_data"] = news_result
            
            # Generate combined response for "both" intent
            if intent == "both":
                response_text = self._generate_combined_response(user_input, weather_result, news_result)
                result["response"] = response_text
            # Generate response for weather only
            elif intent == "weather":
                if weather_result and weather_result.get("success"):
                    response_text = self._generate_weather_response(user_input, weather_result)
                    result["response"] = response_text
                else:
                    # Handle weather query without location or failed fetch
                    error_msg = weather_result.get("error", "Please specify a location for weather information.") if weather_result else "Please specify a location for weather information."
                    result["response"] = f"❌ {error_msg}"
                    # Still save to memory for context
                    self.memory.save_context(
                        {"user_question": user_input},
                        {"response": result["response"]}
                    )
            # Generate response for news only
            elif intent == "news":
                if news_result and news_result.get("success"):
                    response_text = self._generate_news_response(user_input, news_result)
                    result["response"] = response_text
                else:
                    # Handle news query without topic or failed fetch
                    error_msg = news_result.get("error", "Unable to fetch news information.") if news_result else "Please specify a topic for news information."
                    result["response"] = f"❌ {error_msg}"
                    # Still save to memory for context
                    self.memory.save_context(
                        {"user_question": user_input},
                        {"response": result["response"]}
                    )
            
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
        Generate natural language response for weather queries using OpenAI with conversation memory.
        
        Args:
            user_question: Original user question
            weather_data: Weather data from MCP server
            
        Returns:
            Natural language response string
        """
        try:
            # Load conversation history from memory
            memory_variables = self.memory.load_memory_variables({})
            chat_history = memory_variables.get("chat_history", [])
            
            # Format messages with conversation history
            messages = self.response_prompt.format_messages(
                chat_history=chat_history,
                user_question=user_question,
                data=json.dumps(weather_data, indent=2)
            )
            response = self.llm.invoke(messages)
            response_text = response.content.strip()
            
            # Save to memory
            self.memory.save_context(
                {"user_question": user_question},
                {"response": response_text}
            )
            
            return response_text
        except Exception as e:
            # Fallback to simple response if generation fails
            return f"Weather data retrieved, but failed to generate response: {str(e)}"
    
    def _generate_news_response(self, user_question: str, news_data: Dict[str, Any]) -> str:
        """
        Generate natural language response for news queries using OpenAI with conversation memory.
        
        Args:
            user_question: Original user question
            news_data: News data from MCP server
            
        Returns:
            Natural language response string
        """
        try:
            # Load conversation history from memory
            memory_variables = self.memory.load_memory_variables({})
            chat_history = memory_variables.get("chat_history", [])
            
            # Format messages with conversation history
            messages = self.response_prompt.format_messages(
                chat_history=chat_history,
                user_question=user_question,
                data=json.dumps(news_data, indent=2)
            )
            response = self.llm.invoke(messages)
            response_text = response.content.strip()
            
            # Save to memory
            self.memory.save_context(
                {"user_question": user_question},
                {"response": response_text}
            )
            
            return response_text
        except Exception as e:
            # Fallback to simple response if generation fails
            return f"News data retrieved, but failed to generate response: {str(e)}"
    
    def _generate_combined_response(self, user_question: str, weather_data: Optional[Dict[str, Any]], news_data: Optional[Dict[str, Any]]) -> str:
        """
        Generate natural language response for combined weather and news queries using OpenAI.
        
        Args:
            user_question: Original user question
            weather_data: Weather data from MCP server (can be None)
            news_data: News data from MCP server (can be None)
            
        Returns:
            Natural language response string combining both weather and news
        """
        try:
            # Combine both data sources
            combined_data = {}
            if weather_data:
                combined_data["weather"] = weather_data
            if news_data:
                combined_data["news"] = news_data
            
            # Update prompt for combined response with conversation history
            combined_prompt = ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template(
                    "You are a helpful assistant that provides clear, natural responses about weather and news. "
                    "The user asked about both weather and news. Provide a comprehensive response that includes both. "
                    "You have access to conversation history, so you can understand follow-up questions and references to previous topics. "
                    "For weather: Be specific about temperature, conditions, precipitation if relevant. "
                    "For news: Summarize key headlines and topics naturally. "
                    "ALWAYS include clickable links to the original news articles using markdown format: [Article Title](url). "
                    "Each article mentioned must have its corresponding URL link. "
                    "Structure your response clearly, mentioning weather first, then news, or vice versa based on what makes sense."
                ),
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessagePromptTemplate.from_template(
                    "User question: {user_question}\n\n"
                    "Data: {data}\n\n"
                    "Provide a natural, conversational response that addresses both weather and news:"
                )
            ])
            
            # Load conversation history from memory
            memory_variables = self.memory.load_memory_variables({})
            chat_history = memory_variables.get("chat_history", [])
            
            messages = combined_prompt.format_messages(
                chat_history=chat_history,
                user_question=user_question,
                data=json.dumps(combined_data, indent=2)
            )
            response = self.llm.invoke(messages)
            response_text = response.content.strip()
            
            # Save to memory
            self.memory.save_context(
                {"user_question": user_question},
                {"response": response_text}
            )
            
            return response_text
        except Exception as e:
            # Fallback: format manually if generation fails
            response_parts = []
            if weather_data and weather_data.get("success"):
                response_parts.append("🌤️ **Weather Information:**\n")
                loc = weather_data.get("location", {})
                weather = weather_data.get("weather", {})
                response_parts.append(f"Weather in {loc.get('name', 'location')}: {weather.get('temperature', 'N/A')}°C, {weather.get('weather', 'N/A')}\n\n")
            elif weather_data:
                response_parts.append(f"❌ Weather: {weather_data.get('error', 'Unable to fetch weather')}\n\n")
            
            if news_data and news_data.get("success"):
                response_parts.append("📰 **News Information:**\n")
                articles = news_data.get("articles", [])
                for i, article in enumerate(articles[:3], 1):
                    title = article.get('title', 'No title')
                    url = article.get('url', '')
                    if url:
                        response_parts.append(f"{i}. [{title}]({url})\n")
                    else:
                        response_parts.append(f"{i}. {title}\n")
            elif news_data:
                response_parts.append(f"❌ News: {news_data.get('error', 'Unable to fetch news')}\n")
            
            return "".join(response_parts) if response_parts else f"Failed to generate combined response: {str(e)}"
    
    def clear_memory(self):
        """Clear conversation memory."""
        self.memory.clear()