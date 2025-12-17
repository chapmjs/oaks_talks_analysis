#!/usr/bin/env python3
"""
Analyze President Dallin H. Oaks' General Conference talks and generate word clouds.
"""

import os
import glob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
from datetime import datetime
from utils import TextProcessor, combine_talk_texts, get_comprehensive_stopwords

class TalkAnalyzer:
    def __init__(self):
        self.processor = TextProcessor(use_lemmatization=True)
        self.data_dir = "data/talks"
        self.output_dir = "output"
        self.ensure_directories()
    
    def ensure_directories(self):
        """Create output directories if they don't exist."""
        os.makedirs(f"{self.output_dir}/wordclouds", exist_ok=True)
        os.makedirs(f"{self.output_dir}/analysis", exist_ok=True)
    
    def load_talks(self):
        """Load all talk files."""
        talk_files = glob.glob(os.path.join(self.data_dir, "*.txt"))
        
        if not talk_files:
            print("No talk files found. Please run fetch_talks.py first.")
            return []
        
        print(f"Found {len(talk_files)} talk files")
        return talk_files
    
    def generate_wordcloud(self, text, title="Word Cloud", save_path=None, 
                          max_words=100, colormap='viridis'):
        """
        Generate and save a word cloud.
        
        Args:
            text: Text to generate word cloud from
            title: Title for the word cloud
            save_path: Path to save the image
            max_words: Maximum number of words in cloud
            colormap: Color scheme for the cloud
        """
        # Get comprehensive stopwords
        stopwords = get_comprehensive_stopwords()
        
        # Add more conference-specific stopwords
        additional_stops = {
            'lord', 'god', 'jesus', 'christ', 'church', 'lds',
            'latter', 'day', 'will', 'can', 'also', 'one', 'two',
            'brethren', 'sisters', 'oaks', 'dallin', 'president'
        }
        stopwords.update(additional_stops)
        
        # Generate word cloud
        wordcloud = WordCloud(
            width=1600,
            height=900,
            background_color='white',
            stopwords=stopwords,
            max_words=max_words,
            colormap=colormap,
            relative_scaling=0.5,
            min_font_size=10
        ).generate(text)
        
        # Create figure
        plt.figure(figsize=(16, 9))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.title(title, fontsize=20, pad=20)
        plt.axis('off')
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Word cloud saved to: {save_path}")
        
        plt.show()
        
        return wordcloud
    
    def analyze_word_frequencies(self, talks):
        """
        Analyze word frequencies across all talks.
        
        Args:
            talks: List of talk file paths
            
        Returns:
            DataFrame with word frequency analysis
        """
        all_tokens = []
        
        for talk_file in talks:
            content = self.processor.extract_talk_content(talk_file)
            tokens = self.processor.process_text(content)
            all_tokens.extend(tokens)
        
        # Get word frequencies
        word_freq = Counter(all_tokens)
        
        # Convert to DataFrame
        df = pd.DataFrame(word_freq.most_common(100), columns=['Word', 'Frequency'])
        
        # Add percentage column
        total_words = sum(word_freq.values())
        df['Percentage'] = (df['Frequency'] / total_words * 100).round(2)
        
        return df
    
    def analyze_by_decade(self, talks):
        """
        Analyze talks grouped by decade.
        
        Args:
            talks: List of talk file paths
        """
        decades = {}
        
        for talk_file in talks:
            # Extract year from filename (now in format YYYY_type_title.txt)
            basename = os.path.basename(talk_file)
            
            # Extract year from beginning of filename
            year_match = re.match(r'^(\d{4})', basename)
            
            if year_match:
                year = int(year_match.group(1))
                decade = f"{(year // 10) * 10}s"
                
                if decade not in decades:
                    decades[decade] = []
                
                content = self.processor.extract_talk_content(talk_file)
                decades[decade].append(content)
        
        # Generate word cloud for each decade
        for decade, texts in sorted(decades.items()):
            if texts:
                combined_text = '\n'.join(texts)
                save_path = f"{self.output_dir}/wordclouds/decade_{decade}.png"
                print(f"  Generating word cloud for {decade} ({len(texts)} talks)...")
                self.generate_wordcloud(
                    combined_text,
                    title=f"President Oaks - {decade}",
                    save_path=save_path
                )
    
    def analyze_by_type(self, talks):
        """
        Analyze talks grouped by type (General Conference, BYU, etc.).
        
        Args:
            talks: List of talk file paths
        """
        talk_types = {}
        
        for talk_file in talks:
            # Extract type from filename (format: YYYY_Type_title.txt)
            basename = os.path.basename(talk_file)
            
            # Extract talk type
            parts = basename.split('_', 2)
            if len(parts) >= 2:
                talk_type = parts[1].replace('_', ' ')
                
                if talk_type not in talk_types:
                    talk_types[talk_type] = []
                
                content = self.processor.extract_talk_content(talk_file)
                talk_types[talk_type].append(content)
        
        # Generate word cloud for each talk type
        for talk_type, texts in talk_types.items():
            if texts:
                combined_text = '\n'.join(texts)
                safe_type = talk_type.replace(' ', '_').lower()
                save_path = f"{self.output_dir}/wordclouds/type_{safe_type}.png"
                print(f"  Generating word cloud for {talk_type} ({len(texts)} talks)...")
                self.generate_wordcloud(
                    combined_text,
                    title=f"President Oaks - {talk_type}",
                    save_path=save_path,
                    colormap='coolwarm'
                )
    
    def generate_theme_clouds(self, talks):
        """
        Generate word clouds for specific themes.
        
        Args:
            talks: List of talk file paths
        """
        themes = {
            'Faith & Testimony': ['faith', 'testimony', 'believe', 'witness', 'know', 'truth'],
            'Family & Marriage': ['family', 'marriage', 'children', 'parent', 'father', 'mother'],
            'Service & Love': ['service', 'serve', 'love', 'charity', 'help', 'minister'],
            'Covenant & Temple': ['covenant', 'temple', 'ordinance', 'baptism', 'endowment'],
            'Scripture & Revelation': ['scripture', 'revelation', 'prophet', 'bible', 'book']
        }
        
        # Combine all talks
        all_text = combine_talk_texts(talks)
        
        for theme_name, keywords in themes.items():
            # Extract sentences containing theme keywords
            sentences = self.processor.get_sentences(all_text)
            theme_sentences = []
            
            for sentence in sentences:
                sentence_lower = sentence.lower()
                if any(keyword in sentence_lower for keyword in keywords):
                    theme_sentences.append(sentence)
            
            if theme_sentences:
                theme_text = ' '.join(theme_sentences)
                save_path = f"{self.output_dir}/wordclouds/theme_{theme_name.replace(' & ', '_').lower()}.png"
                
                self.generate_wordcloud(
                    theme_text,
                    title=f"Theme: {theme_name}",
                    save_path=save_path,
                    colormap='plasma'
                )
    
    def generate_comprehensive_analysis(self):
        """Run comprehensive analysis and generate all visualizations."""
        print("="*60)
        print("PRESIDENT DALLIN H. OAKS - COMPLETE TALKS ANALYSIS")
        print("="*60)
        
        # Load talks
        talks = self.load_talks()
        if not talks:
            return
        
        # Combine all talks for main word cloud
        print("\n1. Generating main word cloud...")
        combined_text = combine_talk_texts(talks)
        
        # Generate main word cloud
        main_cloud_path = f"{self.output_dir}/wordclouds/oaks_all_talks.png"
        self.generate_wordcloud(
            combined_text,
            title="President Dallin H. Oaks - Complete Talk Collection",
            save_path=main_cloud_path,
            max_words=150
        )
        
        # Word frequency analysis
        print("\n2. Analyzing word frequencies...")
        freq_df = self.analyze_word_frequencies(talks)
        freq_csv_path = f"{self.output_dir}/analysis/word_frequencies.csv"
        freq_df.to_csv(freq_csv_path, index=False)
        print(f"Word frequencies saved to: {freq_csv_path}")
        
        # Display top 20 words
        print("\nTop 20 Most Frequent Words:")
        print("-" * 40)
        for idx, row in freq_df.head(20).iterrows():
            print(f"{row['Word']:20} {row['Frequency']:6} ({row['Percentage']:.2f}%)")
        
        # Generate decade-based clouds
        print("\n3. Generating decade-based word clouds...")
        self.analyze_by_decade(talks)
        
        # Generate type-based clouds (NEW)
        print("\n4. Generating talk-type word clouds...")
        self.analyze_by_type(talks)
        
        # Generate theme-based clouds
        print("\n5. Generating theme-based word clouds...")
        self.generate_theme_clouds(talks)
        
        # Summary statistics
        print("\n6. Summary Statistics:")
        print("-" * 40)
        
        # Total words
        all_tokens = []
        talk_metadata = {'types': {}, 'years': {}}
        
        for talk_file in talks:
            content = self.processor.extract_talk_content(talk_file)
            tokens = self.processor.process_text(content, remove_stops=False, lemmatize=False)
            all_tokens.extend(tokens)
            
            # Extract metadata from filename
            basename = os.path.basename(talk_file)
            year_match = re.match(r'^(\d{4})', basename)
            if year_match:
                year = year_match.group(1)
                talk_metadata['years'][year] = talk_metadata['years'].get(year, 0) + 1
            
            parts = basename.split('_', 2)
            if len(parts) >= 2:
                talk_type = parts[1].replace('_', ' ')
                talk_metadata['types'][talk_type] = talk_metadata['types'].get(talk_type, 0) + 1
        
        print(f"Total talks analyzed: {len(talks)}")
        print(f"Total words: {len(all_tokens):,}")
        print(f"Unique words: {len(set(all_tokens)):,}")
        print(f"Average words per talk: {len(all_tokens) // len(talks):,}")
        
        print(f"\nTalk Types:")
        for talk_type, count in sorted(talk_metadata['types'].items(), key=lambda x: x[1], reverse=True):
            print(f"  {talk_type}: {count} talks")
        
        print(f"\nYear Range:")
        if talk_metadata['years']:
            years = sorted(talk_metadata['years'].keys())
            print(f"  Earliest: {years[0]}")
            print(f"  Latest: {years[-1]}")
            print(f"  Total years covered: {len(years)}")
        
        # Save summary
        summary = {
            'analysis_date': datetime.now().isoformat(),
            'total_talks': len(talks),
            'total_words': len(all_tokens),
            'unique_words': len(set(all_tokens)),
            'avg_words_per_talk': len(all_tokens) // len(talks),
            'talk_types': talk_metadata['types'],
            'years_analyzed': talk_metadata['years']
        }
        
        import json
        summary_path = f"{self.output_dir}/analysis/summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nAnalysis complete! Check the output folder for results.")
        print(f"Word clouds saved in: {self.output_dir}/wordclouds/")
        print(f"Analysis files saved in: {self.output_dir}/analysis/")

if __name__ == "__main__":
    analyzer = TalkAnalyzer()
    analyzer.generate_comprehensive_analysis()
