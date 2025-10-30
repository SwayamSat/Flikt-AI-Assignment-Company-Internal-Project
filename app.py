import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from transformers import pipeline
import pickle
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Customer Feedback Analysis", layout="wide")

st.title("Intelligent Customer Feedback Analysis System")
st.markdown("AI-powered sentiment analysis, summarization, and insights generation")

@st.cache_resource
def load_sentiment_model():
    try:
        with open('sentiment_model.pkl', 'rb') as f:
            model_info = pickle.load(f)
        return model_info
    except:
        return None

@st.cache_resource
def load_summarizer():
    try:
        summarizer = pipeline("summarization", model="t5-small")
        return summarizer
    except:
        return None

def analyze_sentiment(rating):
    if rating >= 4.0:
        return 'Positive'
    elif rating >= 3.0:
        return 'Neutral'
    else:
        return 'Negative'

def summarize_text(text, summarizer, max_length=100):
    try:
        if len(text) > 50:
            summary = summarizer(text[:1024], max_length=max_length, min_length=30, do_sample=False)
            return summary[0]['summary_text']
        return text
    except:
        return text[:200] + "..."

tabs = st.tabs(["Upload Data", "Sentiment Analysis", "Summaries", "Insights"])

with tabs[0]:
    st.header("Upload Customer Feedback Data")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ File uploaded successfully! {len(df)} records loaded.")
        
        st.subheader("Dataset Preview")
        st.dataframe(df.head(10))
        
        st.subheader("Dataset Statistics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", len(df))
        with col2:
            st.metric("Columns", len(df.columns))
        with col3:
            if 'Rating_Numeric' in df.columns:
                st.metric("Avg Rating", f"{df['Rating_Numeric'].mean():.2f}")
        
        st.session_state['df'] = df
    else:
        st.info("Please upload a CSV file to begin analysis")
        try:
            df = pd.read_csv('cleaned_customer_feedback.csv')
            st.session_state['df'] = df
            st.success(f"✅ Using default dataset: {len(df)} records")
        except:
            pass

with tabs[1]:
    st.header("Sentiment Analysis Results")
    
    if 'df' in st.session_state:
        df = st.session_state['df']
        
        if 'Rating_Numeric' in df.columns:
            df['Sentiment'] = df['Rating_Numeric'].apply(analyze_sentiment)
            
            st.subheader("Sentiment Distribution")
            sentiment_counts = df['Sentiment'].value_counts()
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig, ax = plt.subplots(figsize=(8, 6))
                colors = {'Positive': '#2ecc71', 'Neutral': '#f39c12', 'Negative': '#e74c3c'}
                sentiment_counts.plot(kind='bar', ax=ax, color=[colors[x] for x in sentiment_counts.index])
                ax.set_title('Sentiment Distribution', fontsize=14, fontweight='bold')
                ax.set_xlabel('Sentiment')
                ax.set_ylabel('Count')
                plt.xticks(rotation=0)
                st.pyplot(fig)
            
            with col2:
                fig, ax = plt.subplots(figsize=(8, 6))
                colors_list = [colors[x] for x in sentiment_counts.index]
                ax.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%',
                       colors=colors_list, startangle=90)
                ax.set_title('Sentiment Percentage', fontsize=14, fontweight='bold')
                st.pyplot(fig)
            
            st.subheader("Sentiment Breakdown")
            for sentiment in ['Positive', 'Neutral', 'Negative']:
                count = sentiment_counts.get(sentiment, 0)
                percentage = (count / len(df)) * 100
                st.metric(f"{sentiment}", f"{count} reviews", f"{percentage:.1f}%")
            
            st.subheader("Sample Reviews by Sentiment")
            selected_sentiment = st.selectbox("Select Sentiment", ['Positive', 'Neutral', 'Negative'])
            
            filtered_df = df[df['Sentiment'] == selected_sentiment]
            if len(filtered_df) > 0:
                sample = filtered_df.head(5)
                for idx, row in sample.iterrows():
                    with st.expander(f"{row.get('Review Title', 'Review')} - Rating: {row.get('Rating_Numeric', 'N/A')}"):
                        st.write(row.get('Comments', row.get('Combined_Text', 'No content')))
            else:
                st.info(f"No {selected_sentiment} reviews found")
        else:
            st.warning("Rating column not found in dataset")
    else:
        st.warning("Please upload a dataset first")

with tabs[2]:
    st.header("Text Summarization")
    
    if 'df' in st.session_state:
        df = st.session_state['df']
        
        st.subheader("Generate Summary")
        
        summarizer = load_summarizer()
        
        if summarizer:
            text_column = st.selectbox("Select text column", 
                                      [col for col in df.columns if 'Comment' in col or 'Text' in col or 'Review' in col])
            
            if text_column:
                num_reviews = st.slider("Number of reviews to summarize", 1, min(10, len(df)), 3)
                
                if st.button("Generate Summaries"):
                    with st.spinner("Generating summaries..."):
                        for idx, row in df.head(num_reviews).iterrows():
                            text = str(row[text_column])
                            if len(text) > 20:
                                with st.expander(f"Review #{idx + 1} - {row.get('Review Title', 'Summary')}"):
                                    st.write("**Original:**")
                                    st.write(text[:300] + "..." if len(text) > 300 else text)
                                    st.write("**Summary:**")
                                    summary = summarize_text(text, summarizer)
                                    st.write(summary)
        else:
            st.info("Summarization model not available. Install transformers library.")
            
            if 'Combined_Text' in df.columns:
                st.subheader("Text Samples")
                for idx, row in df.head(5).iterrows():
                    with st.expander(f"Review #{idx + 1}"):
                        st.write(row['Combined_Text'][:500])
    else:
        st.warning("Please upload a dataset first")

with tabs[3]:
    st.header("Insights & Analytics")
    
    if 'df' in st.session_state:
        df = st.session_state['df']
        
        st.subheader("Key Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Reviews", len(df))
        
        with col2:
            if 'Rating_Numeric' in df.columns:
                avg_rating = df['Rating_Numeric'].mean()
                st.metric("Average Rating", f"{avg_rating:.2f}/5.0")
        
        with col3:
            if 'Sentiment' in df.columns:
                positive_pct = (df['Sentiment'] == 'Positive').sum() / len(df) * 100
                st.metric("Positive %", f"{positive_pct:.1f}%")
        
        with col4:
            if 'Category' in df.columns:
                st.metric("Categories", df['Category'].nunique())
        
        if 'Category' in df.columns and 'Rating_Numeric' in df.columns:
            st.subheader("Performance by Category")
            category_stats = df.groupby('Category')['Rating_Numeric'].agg(['mean', 'count']).reset_index()
            category_stats.columns = ['Category', 'Avg Rating', 'Count']
            
            fig, ax = plt.subplots(figsize=(10, 6))
            category_stats_sorted = category_stats.sort_values('Avg Rating')
            ax.barh(category_stats_sorted['Category'], category_stats_sorted['Avg Rating'], color='skyblue')
            ax.set_xlabel('Average Rating')
            ax.set_ylabel('Category')
            ax.set_title('Average Rating by Category', fontweight='bold')
            ax.axvline(x=3.0, color='red', linestyle='--', label='Neutral Threshold')
            ax.legend()
            st.pyplot(fig)
        
        if 'Date_Parsed' in df.columns and 'Rating_Numeric' in df.columns:
            st.subheader("Trend Over Time")
            df['Date_Parsed'] = pd.to_datetime(df['Date_Parsed'])
            df['YearMonth'] = df['Date_Parsed'].dt.to_period('M')
            
            monthly = df.groupby('YearMonth')['Rating_Numeric'].mean().reset_index()
            monthly['YearMonth'] = monthly['YearMonth'].astype(str)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(monthly['YearMonth'], monthly['Rating_Numeric'], marker='o', linewidth=2)
            ax.set_xlabel('Month')
            ax.set_ylabel('Average Rating')
            ax.set_title('Rating Trend Over Time', fontweight='bold')
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            st.pyplot(fig)
        
        st.subheader("Recurring Issues")
        if 'Comments_Processed' in df.columns and 'Sentiment' in df.columns:
            negative_reviews = df[df['Sentiment'] == 'Negative']['Comments_Processed'].dropna()
            
            if len(negative_reviews) > 0:
                negative_list = negative_reviews.astype(str).tolist()
                vectorizer = CountVectorizer(max_features=10, ngram_range=(1, 2), min_df=1)
                word_counts = vectorizer.fit_transform(negative_list)
                feature_names = vectorizer.get_feature_names_out()
                word_freq = dict(zip(feature_names, word_counts.toarray().sum(axis=0)))
                top_issues = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
                
                issues_df = pd.DataFrame(top_issues, columns=['Issue', 'Count'])
                
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.barh(issues_df['Issue'], issues_df['Count'], color='coral')
                ax.set_xlabel('Mentions')
                ax.set_ylabel('Issue')
                ax.set_title('Top Recurring Issues', fontweight='bold')
                st.pyplot(fig)
    else:
        st.warning("Please upload a dataset first")

st.sidebar.title("About")
st.sidebar.info("""
**Customer Feedback Analysis System**

Features:
- Upload CSV data
- Sentiment analysis
- Text summarization
- Visual insights
- Trend analysis

Built with Streamlit & AI
""")

st.sidebar.title("Instructions")
st.sidebar.markdown("""
1. Upload your feedback CSV file
2. View sentiment analysis
3. Generate summaries
4. Explore insights and trends
""")

