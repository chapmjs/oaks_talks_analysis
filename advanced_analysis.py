#!/usr/bin/env python3
"""
Advanced NLP analysis of President Dallin H. Oaks' General Conference talks.
Includes sentiment analysis, topic modeling, and temporal analysis.
"""

import os
import glob
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from collections import Counter, defaultdict
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.manifold import TSNE

from utils import TextProcessor, combine_talk_texts

class AdvancedAnalyzer:
    def __init__(self):
        self.processor = TextProcessor(use_lemmatization=True)
        self.data_dir = "data/talks"
        self.output_dir = "output/analysis"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def load_talk_metadata(self):
        """Load talks with metadata."""
        talks_data = []
        talk_files = glob.glob(os.path.join(self.data_dir, "*.txt"))
        
        for filepath in talk_files:
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Extract metadata from header
            metadata = {}
            for line in lines[:5]:
                if ':' in line:
                    key, value = line.split(':', 1)
                    metadata[key.strip()] = value.strip()
            
            # Extract content
            content = self.processor.extract_talk_content(filepath)
            
            talks_data.append({
                'file': filepath,
                'title': metadata.get('Title', 'Unknown'),
                'conference': metadata.get('Conference', 'Unknown'),
                'date': metadata.get('Date', 'Unknown'),
                'content': content
            })
        
        return pd.DataFrame(talks_data)
    
    def extract_topics(self, df, n_topics=5, n_words=10):
        """
        Extract topics using LDA (Latent Dirichlet Allocation).
        
        Args:
            df: DataFrame with talk data
            n_topics: Number of topics to extract
            n_words: Number of words per topic
            
        Returns:
            Topic modeling results
        """
        print(f"\nExtracting {n_topics} topics using LDA...")
        
        # Prepare documents
        documents = df['content'].tolist()
        
        # Create TF-IDF matrix
        vectorizer = TfidfVectorizer(
            max_features=100,
            min_df=2,
            max_df=0.8,
            stop_words=list(self.processor.stopwords),
            ngram_range=(1, 2)
        )
        
        doc_term_matrix = vectorizer.fit_transform(documents)
        
        # LDA model
        lda = LatentDirichletAllocation(
            n_components=n_topics,
            max_iter=10,
            learning_method='online',
            random_state=42
        )
        
        lda.fit(doc_term_matrix)
        
        # Get feature names
        feature_names = vectorizer.get_feature_names_out()
        
        # Extract topics
        topics = []
        for topic_idx, topic in enumerate(lda.components_):
            top_words_idx = topic.argsort()[-n_words:][::-1]
            top_words = [feature_names[i] for i in top_words_idx]
            top_scores = [topic[i] for i in top_words_idx]
            
            topics.append({
                'topic_id': topic_idx,
                'words': top_words,
                'scores': top_scores
            })
        
        # Get document-topic distribution
        doc_topics = lda.transform(doc_term_matrix)
        
        # Add dominant topic to dataframe
        df['dominant_topic'] = doc_topics.argmax(axis=1)
        df['topic_score'] = doc_topics.max(axis=1)
        
        return topics, doc_topics, df
    
    def analyze_sentiment_patterns(self, df):
        """
        Analyze sentiment and emotional patterns in talks.
        
        Args:
            df: DataFrame with talk data
            
        Returns:
            Sentiment analysis results
        """
        print("\nAnalyzing sentiment patterns...")
        
        # Define sentiment word lists (simplified)
        positive_words = {
            'joy', 'happy', 'blessed', 'grateful', 'love', 'peace',
            'hope', 'faith', 'wonderful', 'beautiful', 'sacred',
            'precious', 'glorious', 'rejoice', 'comfort', 'strength'
        }
        
        negative_words = {
            'sin', 'evil', 'darkness', 'suffering', 'pain', 'sorrow',
            'fear', 'doubt', 'temptation', 'struggle', 'difficult',
            'trial', 'adversity', 'challenge', 'burden', 'affliction'
        }
        
        encouragement_words = {
            'can', 'will', 'overcome', 'endure', 'persist', 'strive',
            'achieve', 'succeed', 'grow', 'learn', 'improve', 'progress'
        }
        
        warning_words = {
            'beware', 'caution', 'danger', 'avoid', 'must not', 'should not',
            'warning', 'consequence', 'risk', 'threat'
        }
        
        sentiment_data = []
        
        for idx, row in df.iterrows():
            tokens = set(word.lower() for word in self.processor.tokenize(row['content']))
            
            positive_count = len(tokens & positive_words)
            negative_count = len(tokens & negative_words)
            encouragement_count = len(tokens & encouragement_words)
            warning_count = len(tokens & warning_words)
            
            total_words = len(tokens)
            
            sentiment_data.append({
                'title': row['title'],
                'positive_ratio': positive_count / total_words if total_words > 0 else 0,
                'negative_ratio': negative_count / total_words if total_words > 0 else 0,
                'encouragement_ratio': encouragement_count / total_words if total_words > 0 else 0,
                'warning_ratio': warning_count / total_words if total_words > 0 else 0,
                'sentiment_balance': (positive_count - negative_count) / total_words if total_words > 0 else 0
            })
        
        sentiment_df = pd.DataFrame(sentiment_data)
        return sentiment_df
    
    def extract_scripture_references(self, df):
        """
        Extract and analyze scripture references.
        
        Args:
            df: DataFrame with talk data
            
        Returns:
            Scripture reference analysis
        """
        print("\nExtracting scripture references...")
        
        # Patterns for scripture references
        book_patterns = {
            'Old Testament': [
                'Genesis', 'Exodus', 'Leviticus', 'Numbers', 'Deuteronomy',
                'Isaiah', 'Jeremiah', 'Ezekiel', 'Daniel', 'Psalms', 'Proverbs'
            ],
            'New Testament': [
                'Matthew', 'Mark', 'Luke', 'John', 'Acts', 'Romans',
                'Corinthians', 'Galatians', 'Ephesians', 'Philippians',
                'Revelation', 'James', 'Peter'
            ],
            'Book of Mormon': [
                'Nephi', 'Jacob', 'Enos', 'Mosiah', 'Alma', 'Helaman',
                'Mormon', 'Ether', 'Moroni'
            ],
            'Doctrine and Covenants': [
                'D&C', 'Doctrine and Covenants', 'Section'
            ],
            'Pearl of Great Price': [
                'Moses', 'Abraham', 'Joseph Smith'
            ]
        }
        
        scripture_counts = defaultdict(lambda: defaultdict(int))
        
        for idx, row in df.iterrows():
            content = row['content']
            
            for category, books in book_patterns.items():
                for book in books:
                    # Count mentions (case insensitive)
                    count = len(re.findall(rf'\b{book}\b', content, re.IGNORECASE))
                    if count > 0:
                        scripture_counts[row['title']][category] += count
        
        return scripture_counts
    
    def create_visualizations(self, topics, sentiment_df, scripture_counts):
        """
        Create and save visualization plots.
        
        Args:
            topics: Topic modeling results
            sentiment_df: Sentiment analysis DataFrame
            scripture_counts: Scripture reference counts
        """
        print("\nCreating visualizations...")
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # 1. Topic Distribution Visualization
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Topic Analysis - President Dallin H. Oaks', fontsize=16)
        
        for idx, (ax, topic) in enumerate(zip(axes.flat, topics[:6])):
            words = topic['words'][:8]
            scores = topic['scores'][:8]
            
            ax.barh(range(len(words)), scores, color='steelblue')
            ax.set_yticks(range(len(words)))
            ax.set_yticklabels(words)
            ax.set_xlabel('Importance')
            ax.set_title(f'Topic {idx + 1}')
            ax.invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/topics_visualization.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Sentiment Analysis Visualization
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Sentiment Analysis - President Dallin H. Oaks', fontsize=16)
        
        # Sentiment balance distribution
        axes[0, 0].hist(sentiment_df['sentiment_balance'], bins=20, color='skyblue', edgecolor='black')
        axes[0, 0].set_xlabel('Sentiment Balance')
        axes[0, 0].set_ylabel('Number of Talks')
        axes[0, 0].set_title('Overall Sentiment Distribution')
        axes[0, 0].axvline(x=0, color='red', linestyle='--', alpha=0.5)
        
        # Positive vs Negative ratios
        axes[0, 1].scatter(sentiment_df['positive_ratio'], sentiment_df['negative_ratio'], 
                          alpha=0.6, s=50, color='purple')
        axes[0, 1].set_xlabel('Positive Word Ratio')
        axes[0, 1].set_ylabel('Negative Word Ratio')
        axes[0, 1].set_title('Positive vs Negative Language')
        
        # Encouragement vs Warning
        axes[1, 0].scatter(sentiment_df['encouragement_ratio'], sentiment_df['warning_ratio'],
                          alpha=0.6, s=50, color='green')
        axes[1, 0].set_xlabel('Encouragement Ratio')
        axes[1, 0].set_ylabel('Warning Ratio')
        axes[1, 0].set_title('Encouragement vs Warning')
        
        # Average sentiment scores
        sentiment_means = sentiment_df[['positive_ratio', 'negative_ratio', 
                                       'encouragement_ratio', 'warning_ratio']].mean()
        axes[1, 1].bar(range(4), sentiment_means.values, color=['green', 'red', 'blue', 'orange'])
        axes[1, 1].set_xticks(range(4))
        axes[1, 1].set_xticklabels(['Positive', 'Negative', 'Encouragement', 'Warning'], rotation=45)
        axes[1, 1].set_ylabel('Average Ratio')
        axes[1, 1].set_title('Average Sentiment Scores')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/sentiment_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Scripture References Visualization
        if scripture_counts:
            scripture_totals = defaultdict(int)
            for talk_refs in scripture_counts.values():
                for category, count in talk_refs.items():
                    scripture_totals[category] += count
            
            if scripture_totals:
                fig, ax = plt.subplots(figsize=(10, 6))
                categories = list(scripture_totals.keys())
                counts = list(scripture_totals.values())
                
                ax.bar(categories, counts, color='coral')
                ax.set_xlabel('Scripture Category')
                ax.set_ylabel('Total References')
                ax.set_title('Scripture References by Category')
                plt.xticks(rotation=45, ha='right')
                
                plt.tight_layout()
                plt.savefig(f'{self.output_dir}/scripture_references.png', dpi=300, bbox_inches='tight')
                plt.close()
    
    def generate_report(self, df, topics, sentiment_df):
        """
        Generate a comprehensive analysis report.
        
        Args:
            df: DataFrame with talk data
            topics: Topic modeling results
            sentiment_df: Sentiment analysis DataFrame
        """
        print("\nGenerating comprehensive report...")
        
        report = {
            'analysis_date': datetime.now().isoformat(),
            'total_talks': len(df),
            'topics': [],
            'sentiment_summary': {
                'avg_positive_ratio': float(sentiment_df['positive_ratio'].mean()),
                'avg_negative_ratio': float(sentiment_df['negative_ratio'].mean()),
                'avg_encouragement_ratio': float(sentiment_df['encouragement_ratio'].mean()),
                'avg_warning_ratio': float(sentiment_df['warning_ratio'].mean()),
                'most_positive_talk': sentiment_df.loc[sentiment_df['sentiment_balance'].idxmax(), 'title'],
                'most_balanced_talk': sentiment_df.iloc[(sentiment_df['sentiment_balance'].abs()).idxmin()]['title']
            },
            'topic_distribution': df['dominant_topic'].value_counts().to_dict()
        }
        
        # Add topic details
        for topic in topics:
            report['topics'].append({
                'topic_id': topic['topic_id'],
                'top_words': topic['words'][:10],
                'top_scores': [float(s) for s in topic['scores'][:10]]
            })
        
        # Save report
        report_path = f'{self.output_dir}/advanced_analysis_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Report saved to: {report_path}")
        
        # Also save as CSV for easier viewing
        sentiment_df.to_csv(f'{self.output_dir}/sentiment_analysis.csv', index=False)
        
        # Save topic assignments
        df[['title', 'conference', 'dominant_topic', 'topic_score']].to_csv(
            f'{self.output_dir}/topic_assignments.csv', index=False
        )
    
    def run_analysis(self):
        """Run complete advanced analysis."""
        print("="*60)
        print("ADVANCED ANALYSIS - PRESIDENT DALLIN H. OAKS")
        print("="*60)
        
        # Load data
        df = self.load_talk_metadata()
        
        if df.empty:
            print("No talks found. Please run fetch_talks.py first.")
            return
        
        print(f"Loaded {len(df)} talks for analysis")
        
        # Topic modeling
        topics, doc_topics, df = self.extract_topics(df, n_topics=6)
        
        # Sentiment analysis
        sentiment_df = self.analyze_sentiment_patterns(df)
        
        # Scripture references
        scripture_counts = self.extract_scripture_references(df)
        
        # Create visualizations
        self.create_visualizations(topics, sentiment_df, scripture_counts)
        
        # Generate report
        self.generate_report(df, topics, sentiment_df)
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
        print(f"Visualizations saved in: {self.output_dir}/")
        print(f"Reports saved in: {self.output_dir}/")
        
        # Print summary
        print("\nKey Findings:")
        print("-" * 40)
        print(f"Average Positive Language: {sentiment_df['positive_ratio'].mean():.3%}")
        print(f"Average Negative Language: {sentiment_df['negative_ratio'].mean():.3%}")
        print(f"Average Encouragement: {sentiment_df['encouragement_ratio'].mean():.3%}")
        print(f"Average Warning: {sentiment_df['warning_ratio'].mean():.3%}")
        
        print("\nTop Topics Identified:")
        for i, topic in enumerate(topics[:3], 1):
            print(f"  Topic {i}: {', '.join(topic['words'][:5])}")

if __name__ == "__main__":
    analyzer = AdvancedAnalyzer()
    analyzer.run_analysis()
