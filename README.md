# Weather & News AI Assistant

A Python-based AI agent application built with Streamlit that answers user questions about current weather conditions and latest news. The application leverages agent orchestration patterns and Model Context Protocol (MCP) servers to integrate external data sources.

## How It Works

Application uses an orchestrator that processes natural language queries and routes them to appropriate data sources. Here's how everything works together:

### Architecture Overview

1. **Streamlit Interface** (`app.py`): Provides the web interface where users interact with the AI assistant through a chat-like interface.

2. **Orchestrator** (`agents/orchestrator.py`): The core logic of the application that:
   - Uses OpenAI's language model to classify user queries (weather, news, both, greeting, conversational, or unrelated)
   - Maintains conversation history using LangChain's ConversationBufferMemory for multi-turn conversations
   - Routes requests to appropriate MCP servers based on detected intent
   - Generates natural, conversational responses using OpenAI
   - Handles errors gracefully with user-friendly messages

3. **MCP Servers** (`mcp_config/`):
   - **Weather MCP** (`weather_mcp.py`): Fetches weather data from Open-Meteo API (free, no API key required)
     - Supports current weather and forecasts (up to 7 days)
     - Handles geocoding to convert location names to coordinates
     - Includes comprehensive error handling with fallback mechanisms
   
   - **News MCP** (`news_mcp.py`): Fetches news articles from GNews.io API
     - Supports topic-based searches and general headlines
     - Returns formatted articles with titles, descriptions, URLs, and publication dates
     - Includes error handling for connection issues and API failures

### Workflow

1. **User Input**: User types a question in the Streamlit interface
2. **Intent Classification**: The orchestrator uses OpenAI to classify the query and extract parameters (location, topic, time references)
3. **Data Fetching**: Based on intent, the orchestrator calls the appropriate MCP server(s)
4. **Response Generation**: OpenAI generates a natural language response based on the fetched data
5. **Display**: The response is displayed in the chat interface with proper formatting

### Key Features

- **Natural Language Processing**: Understands queries like "What's the weather in Paris?" or "Show me news about technology"
- **Multi-turn Conversations**: Remembers previous context, so follow-up questions like "and tomorrow?" work seamlessly
- **Conversational AI**: Handles greetings, farewells, and polite responses naturally
- **Error Handling**: User-friendly error messages, especially for connection problems
- **Time References**: Understands time-based queries like "tomorrow", "in 3 days", etc.
- **Clickable Links**: News articles include clickable links to original sources

## Setup Instructions

### Prerequisites

- Python 3.10 or higher
- pip (Python package manager)

### Installation Steps

1. **Create a virtual environment** (recommended):
   ```bash
   python3 -m venv venv
   ```

2. **Activate the virtual environment**:
   
   On macOS/Linux:
   ```bash
   source venv/bin/activate
   ```
   
   On Windows:
   ```bash
   venv\Scripts\activate
   ```

3. **Install required packages**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**:
   ```bash
   streamlit run app.py
   ```

5. **Access the application**:
   
   The application will automatically open in your default web browser at `http://localhost:8501`
   
   If it doesn't open automatically, you can manually navigate to the URL shown in the terminal.

**Note**: 
- The `.env` file is already included in the project
- Weather API (Open-Meteo) doesn't require an API key

## Project Structure

```
ai_weather_news_agent/
├── app.py                 # Streamlit web interface
├── agents/
│   ├── __init__.py
│   └── orchestrator.py    # Core AI orchestrator logic
├── mcp_config/
│   ├── __init__.py
│   ├── weather_mcp.py     # Weather data MCP server
│   └── news_mcp.py        # News data MCP server
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (included, update with your API keys)
└── README.md             # This file
```

## Usage Examples

- **Weather Queries**:
  - "What's the weather in Paris?"
  - "Will it rain tomorrow in London?"
  - "Show me the weather in Almaty"
  - "What about the day after tomorrow?" (follow-up question)

- **News Queries**:
  - "Show me the latest headlines"
  - "What's the news about technology?"
  - "Tell me news about Russia"

- **Combined Queries**:
  - "What's the weather in New York and show me tech news"

- **Conversational**:
  - "Hello"
  - "Thanks"
  - "Goodbye"

## Technologies Used

- **Streamlit**: Web interface framework
- **LangChain**: LLM application framework with memory management
- **OpenAI API**: For intent classification and response generation
- **Open-Meteo API**: Free weather data service
- **GNews.io API**: News aggregation service
- **Python-dotenv**: Environment variable management

## Notes

- The application uses conversation memory, so it remembers context within a session
- You can clear the conversation history using the "Clear Conversation & Memory" button in the sidebar
- The application requires an active internet connection to fetch weather and news data
- Weather forecasts are available for up to 7 days in advance

