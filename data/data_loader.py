"""
Data Loading and Management for News Articles
CSC4352 Final Project

This module handles loading news datasets from various sources
and converting them to a unified format.

Supported Datasets:
- MIND: Microsoft News Dataset (https://msnews.github.io/)
- AG News: 120,000 articles, 4 categories (HuggingFace)
- BBC News: 2,225 articles, 5 categories (Kaggle)
"""

import json
import os
import zipfile
import csv
from pathlib import Path
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import hashlib

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    def tqdm(iterable, **kwargs):
        return iterable


@dataclass
class NewsArticle:
    """
    Unified data structure for news articles.
    
    This schema matches the project specification and provides
    a consistent interface regardless of the data source.
    
    Attributes:
        article_id: Unique identifier for the article
        title: Article headline
        content: Full article text
        category: Topic category (e.g., Sports, Business, Tech)
        source: Dataset source (MIND, AG_News, BBC_News)
        published_date: Publication date (YYYY-MM-DD format)
        url: Original article URL (if available)
    """
    article_id: str
    title: str
    content: str
    category: str
    source: str
    published_date: str
    url: str = ""
    
    def to_dict(self) -> Dict:
        """Convert article to dictionary."""
        return asdict(self)
    
    def get_text_for_embedding(self, separator: str = " [SEP] ") -> str:
        """Get combined title and content for embedding generation."""
        return f"{self.title}{separator}{self.content}"


class NewsDataset:
    """
    Container for a collection of news articles with utility methods.
    """
    
    def __init__(self, articles: List[NewsArticle] = None):
        """
        Initialize the dataset.
        
        Args:
            articles: Optional list of NewsArticle objects
        """
        self.articles = articles or []
        self._id_to_idx: Dict[str, int] = {}
        self._rebuild_index()
    
    def _rebuild_index(self):
        """Rebuild the article ID to index mapping."""
        self._id_to_idx = {article.article_id: idx for idx, article in enumerate(self.articles)}
    
    def add_article(self, article: NewsArticle):
        """Add a single article to the dataset."""
        self._id_to_idx[article.article_id] = len(self.articles)
        self.articles.append(article)
    
    def get_by_id(self, article_id: str) -> Optional[NewsArticle]:
        """Retrieve an article by its ID."""
        idx = self._id_to_idx.get(article_id)
        if idx is not None:
            return self.articles[idx]
        return None
    
    def get_by_index(self, idx: int) -> Optional[NewsArticle]:
        """Retrieve an article by its index."""
        if 0 <= idx < len(self.articles):
            return self.articles[idx]
        return None
    
    def __len__(self) -> int:
        return len(self.articles)
    
    def __iter__(self):
        return iter(self.articles)
    
    def __getitem__(self, idx: int) -> NewsArticle:
        return self.articles[idx]
    
    def to_dataframe(self):
        """Convert dataset to pandas DataFrame."""
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas required: pip install pandas")
        return pd.DataFrame([article.to_dict() for article in self.articles])
    
    def get_categories(self) -> List[str]:
        """Get list of unique categories."""
        return list(set(article.category for article in self.articles))
    
    def get_sources(self) -> List[str]:
        """Get list of unique sources."""
        return list(set(article.source for article in self.articles))
    
    def filter_by_category(self, category: str) -> 'NewsDataset':
        """Return a new dataset filtered by category."""
        filtered = [a for a in self.articles if a.category == category]
        return NewsDataset(filtered)
    
    def filter_by_source(self, source: str) -> 'NewsDataset':
        """Return a new dataset filtered by source."""
        filtered = [a for a in self.articles if a.source == source]
        return NewsDataset(filtered)
    
    def save_to_json(self, filepath: Union[str, Path]):
        """Save dataset to JSON file."""
        filepath = Path(filepath)
        data = [article.to_dict() for article in self.articles]
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(self.articles)} articles to {filepath}")
    
    @classmethod
    def load_from_json(cls, filepath: Union[str, Path]) -> 'NewsDataset':
        """Load dataset from JSON file."""
        filepath = Path(filepath)
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        articles = [NewsArticle(**item) for item in data]
        print(f"Loaded {len(articles)} articles from {filepath}")
        return cls(articles)
    
    def merge(self, other: 'NewsDataset') -> 'NewsDataset':
        """Merge two datasets together."""
        combined = self.articles + other.articles
        return NewsDataset(combined)


def generate_article_id(title: str, source: str) -> str:
    """
    Generate a unique article ID based on title and source.
    
    Uses MD5 hash to create a consistent, unique identifier.
    """
    content = f"{source}:{title}"
    return hashlib.md5(content.encode()).hexdigest()[:12]


# =============================================================================
# AG NEWS DATASET LOADER
# =============================================================================

def load_ag_news(max_articles: int = 10000, split: str = "train") -> NewsDataset:
    """
    Load the AG News dataset from HuggingFace.
    
    AG News contains 120,000 news articles across 4 categories:
    - World (0)
    - Sports (1)
    - Business (2)
    - Sci/Tech (3)
    
    Args:
        max_articles: Maximum number of articles to load
        split: Dataset split ('train' or 'test')
        
    Returns:
        NewsDataset object containing the articles
        
    Reference: https://huggingface.co/datasets/ag_news
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "HuggingFace datasets library required.\n"
            "Install with: pip install datasets"
        )
    
    print(f"Loading AG News dataset from HuggingFace ({split} split)...")
    dataset = load_dataset("ag_news", split=split)
    
    category_map = {
        0: "World",
        1: "Sports", 
        2: "Business",
        3: "Sci/Tech"
    }
    
    articles = []
    num_articles = min(len(dataset), max_articles)
    
    iterator = range(num_articles)
    if TQDM_AVAILABLE:
        iterator = tqdm(iterator, desc="Processing AG News")
    
    for i in iterator:
        item = dataset[i]
        text = item['text']
        
        # AG News format: title and content are combined
        # Try to split at first sentence for title
        if '. ' in text[:200]:
            first_period = text.index('. ')
            title = text[:first_period + 1]
            content = text[first_period + 2:]
        else:
            title = text[:100] + "..."
            content = text
        
        # Include index in ID to guarantee uniqueness
        article_id = f"ag_{i}_{generate_article_id(title, 'AG_News')}"
        
        article = NewsArticle(
            article_id=article_id,
            title=title.strip(),
            content=content.strip(),
            category=category_map[item['label']],
            source="AG_News",
            published_date="2024-01-01",  # AG News doesn't include dates
            url=""
        )
        articles.append(article)
    
    print(f"Loaded {len(articles)} articles from AG News")
    return NewsDataset(articles)


# =============================================================================
# BBC NEWS DATASET LOADER
# =============================================================================

def load_bbc_news(data_path: Union[str, Path]) -> NewsDataset:
    """
    Load the BBC News dataset from local files.
    
    BBC News contains 2,225 articles across 5 categories:
    - business
    - entertainment
    - politics
    - sport
    - tech
    
    Args:
        data_path: Path to the BBC News dataset directory
                   Should contain subdirectories for each category
                   
    Returns:
        NewsDataset object containing the articles
        
    Dataset structure expected:
        data_path/
            business/
                001.txt
                002.txt
                ...
            entertainment/
                001.txt
                ...
            politics/
                ...
            sport/
                ...
            tech/
                ...
                
    Download from: https://www.kaggle.com/c/learn-ai-bbc/data
    """
    data_path = Path(data_path)
    
    if not data_path.exists():
        raise FileNotFoundError(
            f"BBC News dataset not found at {data_path}\n"
            f"Download from: https://www.kaggle.com/c/learn-ai-bbc/data\n"
            f"Extract to: {data_path}"
        )
    
    categories = ["business", "entertainment", "politics", "sport", "tech"]
    articles = []
    
    print(f"Loading BBC News dataset from {data_path}...")
    
    for category in categories:
        category_path = data_path / category
        
        if not category_path.exists():
            print(f"  Warning: Category folder not found: {category_path}")
            continue
        
        txt_files = list(category_path.glob("*.txt"))
        print(f"  Found {len(txt_files)} articles in {category}")
        
        for txt_file in txt_files:
            try:
                with open(txt_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read().strip()
                
                # BBC format: first line is typically the title
                lines = content.split('\n')
                if len(lines) > 1:
                    title = lines[0].strip()
                    body = '\n'.join(lines[1:]).strip()
                else:
                    title = content[:100]
                    body = content
                
                article = NewsArticle(
                    article_id=generate_article_id(title, "BBC_News"),
                    title=title,
                    content=body,
                    category=category.capitalize(),
                    source="BBC_News",
                    published_date="2024-01-01",
                    url=""
                )
                articles.append(article)
                
            except Exception as e:
                print(f"  Error reading {txt_file}: {e}")
    
    print(f"Loaded {len(articles)} articles from BBC News")
    return NewsDataset(articles)


def load_bbc_news_csv(csv_path: Union[str, Path]) -> NewsDataset:
    """
    Load BBC News from CSV format (alternative format from Kaggle).
    
    Args:
        csv_path: Path to bbc-text.csv file
        
    CSV format expected:
        category,text
        tech,"Article content here..."
        business,"Another article..."
    """
    csv_path = Path(csv_path)
    
    if not csv_path.exists():
        raise FileNotFoundError(f"BBC News CSV not found at {csv_path}")
    
    articles = []
    print(f"Loading BBC News from CSV: {csv_path}...")
    
    with open(csv_path, 'r', encoding='utf-8', errors='ignore') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            text = row.get('text', '')
            category = row.get('category', 'Unknown')
            
            # Extract title from first sentence
            if '. ' in text[:200]:
                first_period = text.index('. ')
                title = text[:first_period + 1]
                content = text[first_period + 2:]
            else:
                title = text[:80] + "..."
                content = text
            
            article = NewsArticle(
                article_id=generate_article_id(title, "BBC_News"),
                title=title.strip(),
                content=content.strip(),
                category=category.capitalize(),
                source="BBC_News",
                published_date="2024-01-01",
                url=""
            )
            articles.append(article)
    
    print(f"Loaded {len(articles)} articles from BBC News CSV")
    return NewsDataset(articles)


# =============================================================================
# MIND DATASET LOADER
# =============================================================================

def load_mind_dataset(
    data_path: Union[str, Path],
    max_articles: int = 50000,
    include_behaviors: bool = False
) -> NewsDataset:
    """
    Load the MIND (Microsoft News Dataset).
    
    MIND is a large-scale dataset for news recommendation research,
    containing news articles and user behavior logs.
    
    Args:
        data_path: Path to extracted MIND dataset directory
        max_articles: Maximum number of articles to load
        include_behaviors: Whether to load user behavior data (advanced)
        
    Returns:
        NewsDataset object containing the articles
        
    Dataset structure expected:
        data_path/
            news.tsv          # News article information
            behaviors.tsv     # User interaction logs (optional)
            
    news.tsv columns:
        News ID, Category, SubCategory, Title, Abstract, URL, 
        Title Entities, Abstract Entities
        
    Download from: https://msnews.github.io/
    
    Note: MIND comes in different sizes:
        - MIND-small: ~65,000 articles
        - MIND-large: ~160,000 articles
    """
    data_path = Path(data_path)
    news_file = data_path / "news.tsv"
    
    if not news_file.exists():
        raise FileNotFoundError(
            f"MIND news.tsv not found at {news_file}\n"
            f"Download from: https://msnews.github.io/\n"
            f"Extract the dataset and provide the path to the extracted folder."
        )
    
    articles = []
    print(f"Loading MIND dataset from {data_path}...")
    
    with open(news_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        
        for i, row in enumerate(reader):
            if i >= max_articles:
                break
            
            if len(row) < 5:
                continue
            
            # MIND TSV format:
            # 0: News ID
            # 1: Category
            # 2: SubCategory
            # 3: Title
            # 4: Abstract
            # 5: URL (optional)
            
            news_id = row[0]
            category = row[1] if len(row) > 1 else "Unknown"
            subcategory = row[2] if len(row) > 2 else ""
            title = row[3] if len(row) > 3 else ""
            abstract = row[4] if len(row) > 4 else ""
            url = row[5] if len(row) > 5 else ""
            
            # Use subcategory for more granular categorization if available
            full_category = f"{category}/{subcategory}" if subcategory else category
            
            article = NewsArticle(
                article_id=news_id,
                title=title,
                content=abstract,
                category=full_category,
                source="MIND",
                published_date="2024-01-01",
                url=url
            )
            articles.append(article)
    
    print(f"Loaded {len(articles)} articles from MIND dataset")
    return NewsDataset(articles)


def download_mind_demo(output_path: Union[str, Path] = "./data/mind_demo") -> Path:
    """
    Provide instructions for downloading MIND dataset.
    
    Note: MIND cannot be auto-downloaded due to licensing.
    This function provides instructions for manual download.
    
    Returns:
        Path where dataset should be extracted
    """
    output_path = Path(output_path)
    
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║                    MIND Dataset Download                          ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║                                                                   ║
    ║  The MIND dataset requires manual download from Microsoft.        ║
    ║                                                                   ║
    ║  Steps:                                                           ║
    ║  1. Go to: https://msnews.github.io/                             ║
    ║  2. Download MIND-small (recommended) or MIND-large               ║
    ║  3. Extract the ZIP file                                          ║
    ║  4. Use load_mind_dataset('/path/to/extracted/folder')           ║
    ║                                                                   ║
    ║  Expected files after extraction:                                 ║
    ║  - news.tsv (news articles)                                       ║
    ║  - behaviors.tsv (user interactions)                              ║
    ║                                                                   ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)
    
    return output_path


# =============================================================================
# COMBINED DATASET LOADER
# =============================================================================

def load_all_datasets(
    ag_news_max: int = 5000,
    bbc_path: Optional[Union[str, Path]] = None,
    mind_path: Optional[Union[str, Path]] = None,
    mind_max: int = 5000
) -> NewsDataset:
    """
    Load and combine multiple news datasets.
    
    This is the recommended function for loading data as it combines
    all available datasets into a unified format.
    
    Args:
        ag_news_max: Maximum articles from AG News
        bbc_path: Path to BBC News dataset (None to skip)
        mind_path: Path to MIND dataset (None to skip)
        mind_max: Maximum articles from MIND
        
    Returns:
        Combined NewsDataset with articles from all sources
    """
    all_articles = []
    
    # Always load AG News (available via HuggingFace)
    try:
        print("\n" + "="*50)
        print("Loading AG News...")
        print("="*50)
        ag_dataset = load_ag_news(max_articles=ag_news_max)
        all_articles.extend(ag_dataset.articles)
    except Exception as e:
        print(f"Could not load AG News: {e}")
    
    # Load BBC News if path provided
    if bbc_path:
        try:
            print("\n" + "="*50)
            print("Loading BBC News...")
            print("="*50)
            bbc_path = Path(bbc_path)
            if bbc_path.suffix == '.csv':
                bbc_dataset = load_bbc_news_csv(bbc_path)
            else:
                bbc_dataset = load_bbc_news(bbc_path)
            all_articles.extend(bbc_dataset.articles)
        except Exception as e:
            print(f"Could not load BBC News: {e}")
    
    # Load MIND if path provided
    if mind_path:
        try:
            print("\n" + "="*50)
            print("Loading MIND Dataset...")
            print("="*50)
            mind_dataset = load_mind_dataset(mind_path, max_articles=mind_max)
            all_articles.extend(mind_dataset.articles)
        except Exception as e:
            print(f"Could not load MIND: {e}")
    
    print("\n" + "="*50)
    print(f"Total articles loaded: {len(all_articles)}")
    print("="*50)
    
    return NewsDataset(all_articles)


# =============================================================================
# SAMPLE DATA (for testing)
# =============================================================================

def load_sample_data() -> NewsDataset:
    """
    Create a sample dataset for testing and demonstration.
    
    This provides a small set of diverse articles without requiring
    external data downloads.
    """
    sample_articles = [
        NewsArticle(
            article_id="sample_001",
            title="Tech Giants Report Record Quarterly Earnings",
            content="Major technology companies including Apple, Microsoft, and Google parent Alphabet reported better-than-expected quarterly earnings, driven by strong cloud computing and AI services growth. Apple's revenue exceeded $100 billion for the quarter, while Microsoft saw Azure cloud revenue increase by 29%. Analysts attribute the growth to increased enterprise AI adoption and consumer spending on premium devices.",
            category="Business",
            source="Sample_News",
            published_date="2024-11-15",
            url="https://example.com/tech-earnings"
        ),
        NewsArticle(
            article_id="sample_002",
            title="New Study Reveals Benefits of Mediterranean Diet",
            content="Researchers at Harvard Medical School published findings showing that adherence to a Mediterranean diet reduces cardiovascular disease risk by 25%. The study followed 25,000 participants over 12 years. Key components include olive oil, nuts, fish, and abundant fruits and vegetables. The findings reinforce previous research on diet and heart health.",
            category="Health",
            source="Sample_News",
            published_date="2024-11-14",
            url="https://example.com/diet-study"
        ),
        NewsArticle(
            article_id="sample_003",
            title="SpaceX Successfully Launches Starship to Orbit",
            content="SpaceX achieved a major milestone with its Starship rocket completing a full orbital flight for the first time. The massive rocket launched from Boca Chica, Texas, and successfully separated its Super Heavy booster before the Starship upper stage reached orbit. CEO Elon Musk called it 'a great day for space exploration' as the company moves closer to its Mars colonization goals.",
            category="Sci/Tech",
            source="Sample_News",
            published_date="2024-11-13",
            url="https://example.com/spacex-launch"
        ),
        NewsArticle(
            article_id="sample_004",
            title="World Cup Qualifiers: Surprising Results Shake Up Rankings",
            content="The latest round of World Cup qualifying matches produced several unexpected results. Traditional powerhouses faced tough challenges from emerging football nations. Germany struggled against North Macedonia, while Brazil needed a late goal to secure a draw. The results have significant implications for the final qualification standings.",
            category="Sports",
            source="Sample_News",
            published_date="2024-11-12",
            url="https://example.com/world-cup"
        ),
        NewsArticle(
            article_id="sample_005",
            title="Climate Summit Reaches Historic Agreement on Emissions",
            content="World leaders at the annual climate summit reached a landmark agreement to reduce global carbon emissions by 50% by 2035. The accord includes binding commitments from the world's largest economies and establishes a $100 billion fund to help developing nations transition to clean energy. Environmental groups cautiously welcomed the deal while calling for stronger enforcement mechanisms.",
            category="World",
            source="Sample_News",
            published_date="2024-11-11",
            url="https://example.com/climate-summit"
        ),
        NewsArticle(
            article_id="sample_006",
            title="AI Breakthrough: New Model Achieves Human-Level Reasoning",
            content="Researchers unveiled a new artificial intelligence system that demonstrates unprecedented reasoning capabilities. The model excels at complex mathematical proofs and scientific problem-solving, matching or exceeding human expert performance on benchmark tests. The development raises both excitement about AI's potential and concerns about safety and job displacement.",
            category="Sci/Tech",
            source="Sample_News",
            published_date="2024-11-10",
            url="https://example.com/ai-breakthrough"
        ),
        NewsArticle(
            article_id="sample_007",
            title="Stock Markets Rally on Positive Economic Data",
            content="Global stock markets surged following the release of stronger-than-expected employment and manufacturing data. The S&P 500 reached a new all-time high, while European and Asian markets also posted significant gains. Economists suggest the data indicates a 'soft landing' for the economy, avoiding recession while inflation continues to moderate.",
            category="Business",
            source="Sample_News",
            published_date="2024-11-09",
            url="https://example.com/market-rally"
        ),
        NewsArticle(
            article_id="sample_008",
            title="Tennis Star Announces Retirement After Historic Career",
            content="One of tennis's greatest players announced retirement from professional competition after a career spanning two decades. With 23 Grand Slam titles and over 100 tournament victories, the athlete leaves an indelible mark on the sport. Tributes poured in from fellow players, fans, and sports organizations worldwide.",
            category="Sports",
            source="Sample_News",
            published_date="2024-11-08",
            url="https://example.com/tennis-retirement"
        ),
        NewsArticle(
            article_id="sample_009",
            title="Revolutionary Battery Technology Promises Week-Long Phone Charge",
            content="A startup has developed a new battery technology using solid-state lithium that could allow smartphones to run for a week on a single charge. The technology also significantly reduces charging time to under 10 minutes. Major phone manufacturers are reportedly in talks to license the technology for next-generation devices.",
            category="Sci/Tech",
            source="Sample_News",
            published_date="2024-11-07",
            url="https://example.com/battery-tech"
        ),
        NewsArticle(
            article_id="sample_010",
            title="Central Banks Signal Coordinated Interest Rate Cuts",
            content="Major central banks including the Federal Reserve, European Central Bank, and Bank of England signaled upcoming interest rate cuts as inflation pressures ease. Markets responded positively to the coordinated dovish stance. Analysts expect the rate cuts to begin in early 2025, providing relief to borrowers and potentially stimulating economic growth.",
            category="Business",
            source="Sample_News",
            published_date="2024-11-06",
            url="https://example.com/rate-cuts"
        ),
        NewsArticle(
            article_id="sample_011",
            title="NASA Discovers New Earth-Like Planet in Habitable Zone",
            content="NASA astronomers announced the discovery of a potentially habitable exoplanet located 40 light-years from Earth. The planet, designated as TOI-700 e, orbits within its star's habitable zone where liquid water could exist. Initial atmospheric analysis suggests possible water vapor signatures, making it a prime candidate for future study.",
            category="Sci/Tech",
            source="Sample_News",
            published_date="2024-11-05",
            url="https://example.com/exoplanet"
        ),
        NewsArticle(
            article_id="sample_012",
            title="Olympics Committee Announces New Host City for 2036 Games",
            content="The International Olympic Committee selected a new host city for the 2036 Summer Olympics following an extensive bidding process. The winning bid promises sustainable, carbon-neutral games with extensive use of existing infrastructure. The announcement ends months of speculation and begins the decade-long preparation process.",
            category="Sports",
            source="Sample_News",
            published_date="2024-11-04",
            url="https://example.com/olympics-2036"
        ),
        NewsArticle(
            article_id="sample_013",
            title="Electric Vehicle Sales Surpass Gasoline Cars for First Time",
            content="For the first time in automotive history, electric vehicle sales exceeded traditional gasoline-powered cars in quarterly sales figures. The milestone reflects rapid consumer adoption driven by improved battery technology, expanded charging infrastructure, and government incentives. Traditional automakers are accelerating their EV transition plans in response.",
            category="Business",
            source="Sample_News",
            published_date="2024-11-03",
            url="https://example.com/ev-sales"
        ),
        NewsArticle(
            article_id="sample_014",
            title="UN Report Warns of Accelerating Biodiversity Loss",
            content="A comprehensive United Nations report documents alarming rates of species extinction and habitat destruction worldwide. The report identifies climate change, pollution, and land use changes as primary drivers. Scientists call for immediate action to preserve remaining ecosystems and prevent catastrophic biodiversity collapse within decades.",
            category="World",
            source="Sample_News",
            published_date="2024-11-02",
            url="https://example.com/biodiversity"
        ),
        NewsArticle(
            article_id="sample_015",
            title="Major Cybersecurity Breach Affects Millions of Users",
            content="A sophisticated cyberattack on a major financial services company exposed personal data of approximately 15 million customers. The breach included names, addresses, and encrypted financial information. The company is offering free credit monitoring services and has engaged federal authorities in the investigation.",
            category="Sci/Tech",
            source="Sample_News",
            published_date="2024-11-01",
            url="https://example.com/cyber-breach"
        ),
    ]
    
    print(f"Created sample dataset with {len(sample_articles)} articles")
    return NewsDataset(sample_articles)


# =============================================================================
# MAIN / TESTING
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("Testing Data Loading")
    print("="*60)
    
    # Test sample data
    dataset = load_sample_data()
    print(f"\nDataset size: {len(dataset)}")
    print(f"Categories: {dataset.get_categories()}")
    print(f"Sources: {dataset.get_sources()}")
    
    print("\nSample article:")
    article = dataset[0]
    print(f"  ID: {article.article_id}")
    print(f"  Title: {article.title}")
    print(f"  Category: {article.category}")
    print(f"  Content preview: {article.content[:100]}...")
    
    # Show download instructions for other datasets
    print("\n" + "="*60)
    print("To load other datasets:")
    print("="*60)
    print("""
    # AG News (auto-download via HuggingFace):
    dataset = load_ag_news(max_articles=10000)
    
    # BBC News (manual download from Kaggle):
    dataset = load_bbc_news('/path/to/bbc-news/')
    # OR
    dataset = load_bbc_news_csv('/path/to/bbc-text.csv')
    
    # MIND (manual download from Microsoft):
    dataset = load_mind_dataset('/path/to/mind/')
    
    # Load all available datasets:
    dataset = load_all_datasets(
        ag_news_max=5000,
        bbc_path='/path/to/bbc/',
        mind_path='/path/to/mind/'
    )
    """)