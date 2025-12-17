"""
Custom stopwords for General Conference text analysis.
Includes common English stopwords plus religious/conference-specific common terms.
"""

# Standard English stopwords
ENGLISH_STOPWORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", 
    "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 
    'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 
    'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 
    'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 
    'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 
    'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 
    'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 
    'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 
    'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 
    'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'both', 
    'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 
    'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 
    'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 
    've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 
    'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 
    'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', 
    "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 
    'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"
}

# Common conference/religious terms that might dominate but aren't distinctive
CONFERENCE_COMMON = {
    'elder', 'president', 'brother', 'sister', 'saint', 'saints',
    'conference', 'general', 'talk', 'spoke', 'speaking', 'said', 'says',
    'today', 'time', 'year', 'years', 'day', 'days',
    'may', 'might', 'must', 'shall', 'would', 'could', 'should',
    'also', 'even', 'well', 'much', 'many', 'every',
    'first', 'second', 'third', 'last', 'next',
    'one', 'two', 'three', 'four', 'five',
    'know', 'known', 'knew', 'think', 'thought',
    'come', 'came', 'go', 'went', 'going',
    'make', 'made', 'making', 'give', 'gave', 'given',
    'see', 'saw', 'seen', 'say', 'said', 'saying',
    'take', 'took', 'taken', 'get', 'got', 'getting',
    'us', 'let', 'way', 'like', 'want', 'need',
    'verse', 'verses', 'chapter', 'chapters'
}

# Terms to optionally keep or remove based on analysis goals
OPTIONAL_RELIGIOUS = {
    # These might be kept for theological analysis but removed for theme analysis
    'amen', 'thee', 'thou', 'thy', 'thine', 'ye',
    'unto', 'thereof', 'therein', 'wherefore'
}

# Names that might be too common (optional removal)
COMMON_NAMES = {
    'joseph', 'smith', 'young', 'brigham'
}

def get_stopwords(include_religious=True, include_names=False, include_optional=False):
    """
    Get customized stopword set based on analysis needs.
    
    Args:
        include_religious: Include common religious/conference terms
        include_names: Include common LDS historical names
        include_optional: Include archaic biblical language
    
    Returns:
        Set of stopwords
    """
    stopwords = ENGLISH_STOPWORDS.copy()
    
    if include_religious:
        stopwords.update(CONFERENCE_COMMON)
    
    if include_names:
        stopwords.update(COMMON_NAMES)
    
    if include_optional:
        stopwords.update(OPTIONAL_RELIGIOUS)
    
    return stopwords

def get_minimal_stopwords():
    """Get only essential English stopwords."""
    return ENGLISH_STOPWORDS.copy()

def get_comprehensive_stopwords():
    """Get all stopwords including religious and optional terms."""
    return get_stopwords(
        include_religious=True,
        include_names=True,
        include_optional=True
    )
