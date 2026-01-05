import json
import re

from typing import Dict, Any, Optional, List
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
                "You are a JSON-only classification assistant. Your ONLY job is to classify user queries and return JSON. "
                "You have access to conversation history to understand follow-up questions. "
                "CRITICAL RULES: "
                "1. You MUST return ONLY valid JSON. Nothing else. "
                "2. Do NOT generate weather data, news articles, or any final responses. "
                "3. Do NOT include explanations, markdown, code blocks, or any text outside the JSON. "
                "4. Your response must be a single JSON object starting with {{ and ending with }}. "
                "5. You are ONLY classifying intent - you are NOT answering the question. "
                "IMPORTANT: "
                "- Use conversation history to understand follow-up questions. If user asks 'and tomorrow?' or 'what about that?', use previous context. "
                "- If user says polite responses (like 'thanks', 'ok', 'thank you', 'got it'), set intent to 'polite' - these don't need weather/news data. "
                "- If user says initial greetings (like 'hello', 'hi', 'hey', 'good morning', 'good afternoon'), set intent to 'greeting' - keep conversation friendly. "
                "- If user says conversational reactions, comments, or farewells (like 'goodbye', 'bye', 'nice', 'cool', 'great', 'see you', 'have a good day'), set intent to 'conversational' - these need natural LLM responses. "
                "- If the query is a QUESTION about topics NOT related to weather or news (e.g., 'Can you recommend me a car?', 'What is Python?', 'How to cook pasta?'), set intent to 'unknown'. "
                "- Greetings, polite conversation, and conversational reactions should be 'greeting', 'polite', or 'conversational', NOT 'unknown'. "
                "- 'location' is ONLY for weather queries (geographic places like cities, countries: 'Paris', 'New York', 'Almaty'). "
                "- 'topic' is ONLY for news queries (what the news is about: 'Russia', 'technology', 'sports', 'politics'). "
                "- For weather queries, also parse the time reference into a day_index. "
                "- If location/topic is missing but intent is weather/news, use previous conversation context to fill it. "
                "Day index: 0 = today/now, 1 = tomorrow, 2 = day after tomorrow, 3 = in 3 days, etc. (max 6). "
                "If time is not specified or unclear, day_index should be null. "
                "Examples: "
                "- 'weather in Paris' -> intent: 'weather', location: 'Paris', topic: null "
                "- 'news about Russia' -> intent: 'news', location: null, topic: 'Russia' "
                "- 'and tomorrow?' (after weather query) -> intent: 'weather', location: from previous context, topic: null, day_index: 1 "
                "- 'thanks' or 'ok' -> intent: 'polite', location: null, topic: null "
                "- 'Hello' or 'Hi' -> intent: 'greeting', location: null, topic: null "
                "- 'goodbye' or 'bye' or 'nice' -> intent: 'conversational', location: null, topic: null "
                "- 'Can you recommend me a car?' -> intent: 'unknown', location: null, topic: null "
                "JSON format: "
                '{{"intent": "weather" | "news" | "both" | "polite" | "greeting" | "conversational" | "unknown", "location": "string or null", "topic": "string or null", "time": "string or null", "day_index": number (0-6) or null}}'
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
        
        # Prompt for Conversational Responses (greetings, reactions, farewells)
        self.conversational_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                "You are a friendly and helpful assistant that specializes in weather and news information. "
                "The user is having a conversational interaction with you (greeting, reaction, comment, or farewell). "
                "Respond naturally and appropriately to what they said. "
                "Keep your response brief, friendly, and conversational. "
                "If it's a greeting, welcome them and briefly mention you can help with weather and news. "
                "If it's a farewell (like 'goodbye', 'bye'), respond politely and wish them well. "
                "If it's a reaction or comment (like 'nice', 'cool', 'great'), respond naturally and keep the conversation going. "
                "You have access to conversation history, so you can understand the context of the conversation. "
                "Be warm, friendly, and human-like in your responses."
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{user_input}")
        ])

    def _extract_json_from_response(self, response_text: str) -> Dict[str, Any]:
        """
        Extract JSON from LLM response, handling markdown code blocks and extra text.
        
        Args:
            response_text: Raw response from LLM
            
        Returns:
            Parsed JSON dictionary
            
        Raises:
            ValueError: If JSON cannot be extracted or parsed
        """
        # Try to parse as-is first
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            pass
        
        # Try to extract JSON from markdown code blocks
        import re
        # Look for JSON in ```json ... ``` or ``` ... ``` blocks
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Try to find JSON object in the text
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass
        
        # If all else fails, raise an error with the raw response
        raise ValueError(f"Failed to extract JSON from response: {response_text[:200]}...")
    
    def _infer_intent_from_response(self, response_text: str, user_input: str, chat_history: List) -> Optional[Dict[str, Any]]:
        """
        Fallback: Try to infer intent from LLM response when JSON parsing fails.
        This handles cases where LLM ignores JSON format requirement.
        
        Args:
            response_text: Raw response from LLM
            user_input: Original user input
            chat_history: Conversation history
            
        Returns:
            Inferred intent data dictionary or None if inference fails
        """
        response_lower = response_text.lower()
        user_lower = user_input.lower()
        
        # Check if response contains weather-related content
        weather_keywords = ["temperature", "weather", "rain", "snow", "cloudy", "sunny", "humidity", "wind", "forecast"]
        news_keywords = ["news", "article", "headline", "published", "source"]
        
        has_weather = any(keyword in response_lower for keyword in weather_keywords)
        has_news = any(keyword in response_lower for keyword in news_keywords)
        
        # Try to extract location from response or user input
        location = None
        # Look for common location patterns
        location_patterns = [
            r'\bin\s+([A-Z][a-zA-Z\s]+?)(?:\s|,|\.|\?|$)',
            r'([A-Z][a-zA-Z\s]+?)(?:\s*,?\s*Russia|\s*,?\s*USA|\s*,?\s*France)',
        ]
        for pattern in location_patterns:
            match = re.search(pattern, response_text, re.IGNORECASE)
            if match:
                location = match.group(1).strip()
                break
        
        # If no location in response, try user input
        if not location:
            for pattern in location_patterns:
                match = re.search(pattern, user_input, re.IGNORECASE)
                if match:
                    location = match.group(1).strip()
                    break
        
        # Try to extract location from conversation history
        if not location:
            for msg in reversed(chat_history):
                if hasattr(msg, 'content') and isinstance(msg, HumanMessage):
                    prev_content = msg.content
                    match = re.search(r'weather\s+(?:in|at|for)\s+([A-Z][a-zA-Z\s]+?)(?:\?|$|\.|,)', prev_content, re.IGNORECASE)
                    if match:
                        location = match.group(1).strip()
                        break
        
        # Determine intent
        if has_weather and has_news:
            intent = "both"
        elif has_weather:
            intent = "weather"
        elif has_news:
            intent = "news"
        else:
            # Check user input for keywords
            if any(kw in user_lower for kw in ["weather", "temperature", "rain", "forecast"]):
                intent = "weather"
            elif any(kw in user_lower for kw in ["news", "headline", "article"]):
                intent = "news"
            elif any(kw in user_lower for kw in ["hello", "hi", "hey"]):
                intent = "greeting"
            elif any(kw in user_lower for kw in ["thanks", "thank you", "ok"]):
                intent = "polite"
            elif any(kw in user_lower for kw in ["goodbye", "bye", "see you", "nice", "cool", "great", "awesome"]):
                intent = "conversational"
            else:
                return None  # Cannot infer intent
        
        # Try to extract day_index from user input
        day_index = None
        if "tomorrow" in user_lower:
            day_index = 1
        elif "today" in user_lower or "now" in user_lower:
            day_index = 0
        
        return {
            "intent": intent,
            "location": location,
            "topic": None,  # Hard to infer topic from response
            "time": None,
            "day_index": day_index
        }
    
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

            # Step 4: Get response text and extract JSON
            response_text = response.content.strip()
            
            # Step 5: Extract JSON from response (handle markdown code blocks or extra text)
            try:
                intent_data = self._extract_json_from_response(response_text)
            except ValueError as e:
                # If JSON extraction fails, try to infer intent from the response
                # This is a fallback when LLM ignores JSON format requirement
                intent_data = self._infer_intent_from_response(response_text, user_input, chat_history)
                if not intent_data:
                    # If inference also fails, return error
                    result = {
                        "intent": "unknown",
                        "location": None,
                        "topic": None,
                        "time": None,
                        "day_index": None,
                        "success": False,
                        "error": f"Failed to parse intent classification. {str(e)}",
                        "raw_response": response_text[:500],  # Store first 500 chars for debugging
                        "user_question": user_input
                    }
                    return result
            
            # Step 6: Handle queries using MCP servers
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
            
            # Handle conversational intents (greetings, reactions, farewells) - use LLM for natural responses
            if intent in ["greeting", "polite", "conversational"]:
                # Load conversation history
                memory_variables = self.memory.load_memory_variables({})
                chat_history = memory_variables.get("chat_history", [])
                
                # Generate conversational response using LLM
                messages = self.conversational_prompt.format_messages(
                    chat_history=chat_history,
                    user_input=user_input
                )
                
                llm_response = self.llm.invoke(messages)
                response_text = llm_response.content.strip()
                
                result["response"] = response_text
                result["success"] = True
                
                # Save to memory
                self.memory.save_context(
                    {"user_question": user_input},
                    {"response": response_text}
                )
                
                return result
            
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
                # If location is missing, try to extract from previous conversation
                if not location:
                    # Get previous intent data from memory to extract location
                    memory_variables = self.memory.load_memory_variables({})
                    chat_history = memory_variables.get("chat_history", [])
                    # Look backwards through history for previous weather query with location
                    for i in range(len(chat_history) - 1, -1, -1):
                        msg = chat_history[i]
                        if hasattr(msg, 'content') and isinstance(msg, HumanMessage):
                            # Try to extract location from previous user message
                            prev_content = msg.content
                            # Look for "weather in [location]" pattern (case-insensitive)
                            match = re.search(r'weather\s+(?:in|at|for)\s+([A-Z][a-zA-Z\s]+?)(?:\?|$|\.|,)', prev_content, re.IGNORECASE)
                            if match:
                                location = match.group(1).strip()
                                break
                
                if location:
                    weather_result = self.weather_mcp.get_weather(location, day_index=day_index)
                    result["weather_data"] = weather_result
                else:
                    # Weather intent but no location found - return helpful message
                    result["weather_data"] = {
                        "success": False,
                        "error": "Please specify a location for weather information (e.g., 'What's the weather in Paris?')."
                    }
            
            # Handle news queries
            news_result = None
            if intent in ["news", "both"]:
                # If topic is missing, try to extract from previous conversation
                if not topic:
                    memory_variables = self.memory.load_memory_variables({})
                    chat_history = memory_variables.get("chat_history", [])
                    # Look backwards through history for previous news query with topic
                    for i in range(len(chat_history) - 1, -1, -1):
                        msg = chat_history[i]
                        if hasattr(msg, 'content') and isinstance(msg, HumanMessage):
                            prev_content = msg.content
                            # Try to extract topic from previous user message
                            # Look for "news about [topic]" pattern (case-insensitive)
                            match = re.search(r'news\s+(?:about|on|for)\s+([a-zA-Z\s]+?)(?:\?|$|\.|,)', prev_content, re.IGNORECASE)
                            if match:
                                topic = match.group(1).strip()
                                break
                
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