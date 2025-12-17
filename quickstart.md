# Quick Start Guide - Oaks Complete Talk Collection Analysis

## Overview
This project now analyzes **ALL talks by President Dallin H. Oaks**, not just General Conference. It pulls from the comprehensive collection at bencrowder.net which includes:
- General Conference talks
- BYU speeches and devotionals
- Other church addresses
- Talks spanning multiple decades

## Setup Instructions

### 1. Clone or Download the Repository
Download the `oaks-conference-analysis` folder to your local machine.

### 2. Install Python Dependencies
```bash
cd oaks-conference-analysis
pip install -r requirements.txt
```

### 3. Fetch ALL Talks
```bash
python fetch_talks.py
```
This downloads the complete collection from bencrowder.net. Features:
- Automatically fetches talks from multiple sources
- Organizes by year and talk type
- Skips already downloaded talks (resumable)
- Creates a download summary report

**Note**: The first run may take 30-60 minutes as it downloads hundreds of talks spanning multiple decades.

### 4. Generate Word Clouds
```bash
python analyze_talks.py
```
This creates multiple visualizations:
- **Main word cloud** - All talks combined
- **Decade-based clouds** - Evolution over time
- **Type-based clouds** - General Conference vs BYU speeches
- **Theme-based clouds** - Faith, Family, Service, etc.
- **Word frequency analysis** - Top terms with statistics

### 5. Run Advanced Analysis
```bash
python advanced_analysis.py
```
This performs:
- Topic modeling (LDA)
- Sentiment analysis
- Scripture reference extraction
- Statistical visualizations

## Output Files

All results are saved in the `output/` directory:
- `output/wordclouds/` - All word cloud images
  - `oaks_all_talks.png` - Complete collection
  - `decade_*.png` - By decade (1970s, 1980s, etc.)
  - `type_*.png` - By talk type
  - `theme_*.png` - By theme
- `output/analysis/` - CSV files and JSON reports
  - `word_frequencies.csv` - Top words with counts
  - `summary.json` - Complete statistics
  - `download_summary.txt` - Collection overview

## Key Improvements from Original

1. **Complete Collection**: Now analyzes ALL talks, not just recent General Conference
2. **Historical Coverage**: Includes talks from multiple decades
3. **Multi-Source Support**: Handles churchofjesuschrist.org, speeches.byu.edu, and other sources
4. **Talk Type Analysis**: Separate analysis for different types of talks
5. **Resumable Downloads**: Skips already downloaded talks

## Customization Options

### Adjusting Stopwords
Edit `utils/custom_stopwords.py` to add/remove terms from word clouds.

### Filtering by Year or Type
Modify `analyze_talks.py` to focus on specific periods or talk types:
```python
# Example: Only analyze General Conference talks
talks = [t for t in talks if 'General_Conference' in t]

# Example: Only analyze talks from 2000 onwards
talks = [t for t in talks if int(t.split('_')[0]) >= 2000]
```

## GitHub Repository Setup

To create a GitHub repository:

```bash
git init
git add .
git commit -m "Complete Dallin H. Oaks talk collection analysis"
git remote add origin https://github.com/yourusername/oaks-conference-analysis.git
git branch -M main
git push -u origin main
```

## Educational Applications for BYU-Idaho

This enhanced project demonstrates:
- **Web Scraping at Scale**: Handling multiple websites and formats
- **Data Organization**: Managing hundreds of documents systematically
- **Text Mining**: Extracting insights from large text corpora
- **Temporal Analysis**: Tracking theme evolution over decades
- **Python Skills**: Advanced file handling, regex, and data structures

Perfect for demonstrating to your Operations Management students:
- How to handle large-scale data collection projects
- Building resilient scrapers that handle multiple sources
- Creating meaningful visualizations from unstructured data
- Developing production-ready Python applications

## Troubleshooting

### If the bencrowder.net site is unavailable:
The fetcher will save what it can download. You can re-run later to get missing talks.

### Memory issues with large word clouds:
Reduce `max_words` parameter in `analyze_talks.py`

### NLTK data errors:
```python
import nltk
nltk.download('all')
```

## Sample Analysis Questions

With this complete collection, you can now answer:
- How have President Oaks' themes evolved over decades?
- What topics are emphasized more in BYU speeches vs General Conference?
- How does his language differ when addressing different audiences?
- What scripture references appear most frequently across all talks?

This comprehensive dataset enables much deeper analysis than just recent Conference talks!
