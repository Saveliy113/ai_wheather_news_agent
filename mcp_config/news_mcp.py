"""News MCP server using RSS feeds (free, no API key required)."""
import requests
import xml.etree.ElementTree as ET
import os
import re
from typing import Dict, Any, Optional, List
from datetime import datetime


class NewsMCP:
    """
    News MCP server that fetches news articles from RSS feeds.
    No API key required - completely free!
    """
    
    # RSS feed sources (free, no API key needed)
    RSS_FEEDS = {
        "general": "http://feeds.bbci.co.uk/news/rss.xml",
        "technology": "http://feeds.bbci.co.uk/news/technology/rss.xml",
        "science": "http://feeds.bbci.co.uk/news/science_and_environment/rss.xml",
        "business": "http://feeds.bbci.co.uk/news/business/rss.xml",
        "sports": "http://feeds.bbci.co.uk/sport/rss.xml",
        "health": "http://feeds.bbci.co.uk/news/health/rss.xml"
    }
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the News MCP server.
        
        Args:
            api_key: Optional (kept for compatibility, but not used - RSS feeds are free)
        """
        # API key is optional - we use RSS feeds which are free
        self.api_key = api_key or os.getenv("GNEWS_API_KEY")
    
    def get_news(self, topic: Optional[str] = None, max_articles: int = 5) -> Dict[str, Any]:
        """
        Get latest news articles from RSS feeds (free, no API key required).
        
        Args:
            topic: Optional topic to search for (e.g., "technology", "sports", "science")
                  If None, returns general headlines
            max_articles: Maximum number of articles to return (default: 5)
            
        Returns:
            Dictionary with news articles or error information
        """
        try:
            # Step 1: Determine which RSS feed to use based on topic
            feed_url = self._get_feed_url(topic)
            
            # Step 2: Fetch RSS feed
            response = requests.get(feed_url, timeout=10)
            response.raise_for_status()
            
            # Step 3: Parse RSS XML
            root = ET.fromstring(response.content)
            
            # RSS 2.0 namespace handling
            items = root.findall('.//item') or root.findall('.//{http://purl.org/rss/1.0/}item')
            
            # Step 4: Extract and format articles
            articles = []
            for item in items[:max_articles]:
                title_elem = item.find('title') or item.find('{http://purl.org/rss/1.0/}title')
                description_elem = item.find('description') or item.find('{http://purl.org/rss/1.0/}description')
                link_elem = item.find('link') or item.find('{http://purl.org/rss/1.0/}link')
                pub_date_elem = item.find('pubDate') or item.find('{http://purl.org/rss/1.0/}date')
                
                title = title_elem.text if title_elem is not None and title_elem.text else "No title"
                description = description_elem.text if description_elem is not None and description_elem.text else ""
                link = link_elem.text if link_elem is not None and link_elem.text else ""
                pub_date = pub_date_elem.text if pub_date_elem is not None and pub_date_elem.text else ""
                
                # Filter by topic if specified and not using topic-specific feed
                if topic and feed_url == self.RSS_FEEDS["general"]:
                    topic_lower = topic.lower()
                    if topic_lower not in title.lower() and topic_lower not in self._clean_html(description).lower():
                        continue
                
                articles.append({
                    "title": title,
                    "description": self._clean_html(description),
                    "url": link,
                    "publishedAt": pub_date,
                    "source": "BBC News"
                })
                
                if len(articles) >= max_articles:
                    break
            
            if not articles:
                return {
                    "success": False,
                    "error": f"No news articles found for topic: {topic}" if topic else "No news articles found"
                }
            
            # Step 5: Return formatted result
            return {
                "success": True,
                "topic": topic if topic else "headlines",
                "articles": articles
            }
            
        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "error": f"Failed to fetch news from RSS feed: {str(e)}"
            }
        except ET.ParseError as e:
            return {
                "success": False,
                "error": f"Failed to parse news feed: {str(e)}"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}"
            }
    
    def _get_feed_url(self, topic: Optional[str]) -> str:
        """
        Get appropriate RSS feed URL based on topic.
        
        Args:
            topic: Topic string (e.g., "technology", "sports")
            
        Returns:
            RSS feed URL
        """
        if not topic:
            return self.RSS_FEEDS["general"]
        
        topic_lower = topic.lower()
        
        # Map common topics to feed categories
        if any(word in topic_lower for word in ["tech", "technology", "software", "computer", "ai", "artificial intelligence"]):
            return self.RSS_FEEDS["technology"]
        elif any(word in topic_lower for word in ["science", "research", "study"]):
            return self.RSS_FEEDS["science"]
        elif any(word in topic_lower for word in ["business", "economy", "finance", "market"]):
            return self.RSS_FEEDS["business"]
        elif any(word in topic_lower for word in ["sport", "sports", "football", "soccer", "basketball"]):
            return self.RSS_FEEDS["sports"]
        elif any(word in topic_lower for word in ["health", "medical", "medicine", "disease"]):
            return self.RSS_FEEDS["health"]
        else:
            # Default to general feed
            return self.RSS_FEEDS["general"]
    
    def _clean_html(self, text: str) -> str:
        """Remove HTML tags from text."""
        if not text:
            return ""
        # Simple HTML tag removal
        text = re.sub(r'<[^>]+>', '', text)
        # Decode common HTML entities
        text = text.replace('&amp;', '&')
        text = text.replace('&lt;', '<')
        text = text.replace('&gt;', '>')
        text = text.replace('&quot;', '"')
        text = text.replace('&#39;', "'")
        text = text.replace('&nbsp;', ' ')
        return text.strip()

