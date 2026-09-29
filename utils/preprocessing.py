"""
Text Preprocessing Pipeline for News Articles
CSC4352 Final Project

This module handles all text cleaning and preprocessing operations
required before generating embeddings.
"""

import re
import html
from typing import List, Optional
import unicodedata


class TextPreprocessor:
    """
    Handles text cleaning and normalization for news articles.
    
    The preprocessing pipeline includes:
    1. HTML tag and entity removal
    2. Special character handling
    3. Whitespace normalization
    4. Unicode normalization
    5. Optional lowercase conversion
    """
    
    def __init__(self, lowercase: bool = False):
        """
        Initialize the preprocessor.
        
        Args:
            lowercase: Whether to convert text to lowercase.
                      Set to False for cased models like BERT-base-cased.
        """
        self.lowercase = lowercase
        
        # Compile regex patterns for efficiency
        self._html_tag_pattern = re.compile(r'<[^>]+>')
        self._url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
        self._email_pattern = re.compile(r'\S+@\S+\.\S+')
        self._multiple_spaces = re.compile(r'\s+')
        self._special_chars = re.compile(r'[^\w\s.,!?;:\'\"-]')
    
    def clean_html(self, text: str) -> str:
        """Remove HTML tags and decode HTML entities."""
        # Decode HTML entities (e.g., &amp; -> &)
        text = html.unescape(text)
        # Remove HTML tags
        text = self._html_tag_pattern.sub(' ', text)
        return text
    
    def remove_urls(self, text: str) -> str:
        """Remove URLs from text."""
        return self._url_pattern.sub(' ', text)
    
    def remove_emails(self, text: str) -> str:
        """Remove email addresses from text."""
        return self._email_pattern.sub(' ', text)
    
    def normalize_unicode(self, text: str) -> str:
        """Normalize Unicode characters to their canonical form."""
        # NFKC normalization: compatibility decomposition, followed by canonical composition
        return unicodedata.normalize('NFKC', text)
    
    def normalize_whitespace(self, text: str) -> str:
        """Collapse multiple whitespaces into single space and strip."""
        text = self._multiple_spaces.sub(' ', text)
        return text.strip()
    
    def remove_special_characters(self, text: str, keep_punctuation: bool = True) -> str:
        """
        Remove special characters while optionally keeping punctuation.
        
        Args:
            text: Input text
            keep_punctuation: If True, keeps common punctuation marks
        """
        if keep_punctuation:
            return self._special_chars.sub(' ', text)
        else:
            # Remove all non-alphanumeric except spaces
            return re.sub(r'[^\w\s]', ' ', text)
    
    def preprocess(self, text: str, keep_punctuation: bool = True) -> str:
        """
        Apply full preprocessing pipeline to text.
        
        Args:
            text: Raw input text
            keep_punctuation: Whether to preserve punctuation marks
            
        Returns:
            Cleaned and normalized text
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Apply preprocessing steps in order
        text = self.clean_html(text)
        text = self.remove_urls(text)
        text = self.remove_emails(text)
        text = self.normalize_unicode(text)
        text = self.remove_special_characters(text, keep_punctuation)
        text = self.normalize_whitespace(text)
        
        if self.lowercase:
            text = text.lower()
        
        return text
    
    def preprocess_batch(self, texts: List[str], keep_punctuation: bool = True) -> List[str]:
        """
        Preprocess a batch of texts.
        
        Args:
            texts: List of raw text strings
            keep_punctuation: Whether to preserve punctuation marks
            
        Returns:
            List of cleaned texts
        """
        return [self.preprocess(text, keep_punctuation) for text in texts]


def combine_title_content(title: str, content: str, separator: str = " [SEP] ") -> str:
    """
    Combine article title and content for embedding generation.
    
    The title is prepended to provide strong semantic signal about the article topic.
    
    Args:
        title: Article headline
        content: Article body text
        separator: String to separate title from content
        
    Returns:
        Combined text suitable for embedding
    """
    title = title.strip() if title else ""
    content = content.strip() if content else ""
    
    if title and content:
        return f"{title}{separator}{content}"
    elif title:
        return title
    else:
        return content


def truncate_text(text: str, max_words: int = 300) -> str:
    """
    Truncate text to approximately max_words.
    
    This is useful for very long articles to ensure they fit within
    the model's context window while keeping meaningful content.
    
    Args:
        text: Input text
        max_words: Maximum number of words to keep
        
    Returns:
        Truncated text
    """
    words = text.split()
    if len(words) <= max_words:
        return text
    return ' '.join(words[:max_words]) + '...'


# Example usage and testing
if __name__ == "__main__":
    preprocessor = TextPreprocessor(lowercase=False)
    
    # Test cases
    test_texts = [
        "Breaking: Tesla's New AI Model Revolutionizes Self-Driving!",
        "<p>This is <b>HTML</b> content &amp; entities</p>",
        "Check out https://example.com for more info!",
        "Contact us at test@example.com  for   details",
        "Special chars: @#$%^&*() should be handled",
    ]
    
    print("=" * 60)
    print("Text Preprocessing Examples")
    print("=" * 60)
    
    for text in test_texts:
        cleaned = preprocessor.preprocess(text)
        print(f"\nOriginal: {text}")
        print(f"Cleaned:  {cleaned}")
