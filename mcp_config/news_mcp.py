"""News MCP server using GNews.io API."""
import requests
import os
from typing import Dict, Any, Optional


class NewsMCP:
    """
    News MCP server that fetches news articles from GNews.io API.
    Requires GNEWS_API_KEY environment variable.
    """
    
    # GNews.io API endpoint
    BASE_URL = "https://gnews.io/api/v4"
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the News MCP server.
        
        Args:
            api_key: Optional GNews.io API key (or use GNEWS_API_KEY env variable)
        """
        self.api_key = api_key or os.getenv("GNEWS_API_KEY")
    
    def get_news(self, topic: Optional[str] = None, max_articles: int = 5) -> Dict[str, Any]:
        """
        Get latest news articles from GNews.io API.
        
        Args:
            topic: Optional topic to search for (e.g., "technology", "sports")
                  If None, returns top headlines
            max_articles: Maximum number of articles to return (default: 5)
            
        Returns:
            Dictionary with news articles or error information
        """
        try:
            # Step 1: Check API key
            if not self.api_key:
                return {
                    "success": False,
                    "error": "GNews API key not configured. Set GNEWS_API_KEY environment variable or pass api_key parameter."
                }
            
            # Step 2: Determine API endpoint and parameters
            print(f"Topic: {topic}")
            if topic:
                # Search for specific topic
                url = f"{self.BASE_URL}/search"
                params = {
                    "q": topic,
                    "token": self.api_key,
                    "lang": "en",
                    "max": max_articles
                }
            else:
                # Get top headlines
                url = f"{self.BASE_URL}/top-headlines"
                params = {
                    "token": self.api_key,
                    "lang": "en",
                    "max": max_articles
                }
            
            # Step 3: Fetch news from API
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            # Step 4: Format articles
            articles = data.get("articles", [])
            print(f"Articles: {articles}")
            
            if not articles:
                return {
                    "success": False,
                    "error": f"No news articles found for topic: {topic}" if topic else "No news articles found"
                }
            
            # Step 5: Format and return result
            formatted_articles = []
            for article in articles[:max_articles]:
                formatted_articles.append({
                    "title": article.get("title", ""),
                    "description": article.get("description", ""),
                    "url": article.get("url", ""),
                    "publishedAt": article.get("publishedAt", ""),
                    "source": article.get("source", {}).get("name", "Unknown")
                })
            
            return {
                "success": True,
                "topic": topic if topic else "headlines",
                "articles": formatted_articles
            }
            
        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "error": f"Failed to fetch news: {str(e)}"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}"
            }
