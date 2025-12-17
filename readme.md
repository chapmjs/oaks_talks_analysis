# President Dallin H. Oaks Complete Talk Collection Analysis

A Python project for analyzing ALL talks by President Dallin H. Oaks, including General Conference, BYU speeches, devotionals, and other addresses. This comprehensive analysis tool fetches talks from the complete collection curated at bencrowder.net.

## Features

- **Complete Talk Collection**: Fetches ALL available talks by President Dallin H. Oaks (not just General Conference)
- **Multi-Source Support**: Handles talks from churchofjesuschrist.org, speeches.byu.edu, and other sources
- **Comprehensive Text Analysis**: Word clouds, sentiment analysis, topic modeling
- **Historical Coverage**: Includes talks spanning multiple decades
- **Advanced NLP**: Topic extraction, sentiment patterns, scripture reference analysis

## Data Source

This project uses the comprehensive collection maintained at:
- https://bencrowder.net/collected-talks/dallin-h-oaks/

This collection includes:
- General Conference talks
- BYU speeches and devotionals
- Other church addresses and talks
- Historical talks spanning several decades

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/oaks-conference-analysis.git
cd oaks-conference-analysis
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Fetch and save talks:
```bash
python fetch_talks.py
```

2. Analyze texts and generate word cloud:
```bash
python analyze_talks.py
```

3. View advanced analysis:
```bash
python advanced_analysis.py
```

## Project Structure

```
oaks-conference-analysis/
├── README.md
├── requirements.txt
├── .gitignore
├── fetch_talks.py          # Script to fetch talks from the website
├── analyze_talks.py        # Basic text analysis and word cloud generation
├── advanced_analysis.py    # Advanced NLP analysis
├── data/                   # Folder for storing fetched talks
│   └── talks/             # Individual talk files
├── output/                 # Generated visualizations
│   ├── wordclouds/        # Word cloud images
│   └── analysis/          # Analysis reports
└── utils/                  # Utility functions
    ├── __init__.py
    ├── text_processing.py  # Text cleaning and processing
    └── custom_stopwords.py # Custom stopwords for religious texts
```

## Key Technologies

- **BeautifulSoup4**: Web scraping
- **Requests**: HTTP requests
- **WordCloud**: Word cloud generation
- **NLTK**: Natural language processing
- **Pandas**: Data manipulation
- **Matplotlib**: Visualization

## Analysis Features

- Word frequency analysis
- Sentiment analysis
- Topic modeling
- Temporal analysis of themes
- Custom stopword filtering for religious context

## License

MIT License

## Author

Your Name

## Acknowledgments

Talks sourced from the official Church of Jesus Christ of Latter-day Saints website.
