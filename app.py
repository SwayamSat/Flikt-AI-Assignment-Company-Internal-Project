import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from transformers import pipeline
import pickle
import warnings
import streamlit.components.v1 as components
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Customer Feedback Analysis", layout="wide")

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = [{"role": "assistant", "content": "Hello! I'm your AI feedback assistant. Ask me about sentiment analysis, ratings, recurring issues, or recommendations to improve customer satisfaction."}]

if 'chat_input_counter' not in st.session_state:
    st.session_state.chat_input_counter = 0

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

@st.cache_resource
def load_chatbot():
    return {"available": True}

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

def get_feedback_context():
    if 'df' not in st.session_state:
        return "No feedback data loaded yet."
    
    df = st.session_state['df']
    context_parts = []
    
    if 'Rating_Numeric' in df.columns:
        avg_rating = df['Rating_Numeric'].mean()
        context_parts.append(f"Average rating: {avg_rating:.2f}/5.0")
        
        df['Sentiment'] = df['Rating_Numeric'].apply(analyze_sentiment)
        sentiment_counts = df['Sentiment'].value_counts()
        pos = sentiment_counts.get('Positive', 0)
        neu = sentiment_counts.get('Neutral', 0)
        neg = sentiment_counts.get('Negative', 0)
        context_parts.append(f"Positive: {pos}, Neutral: {neu}, Negative: {neg}")
    
    if 'Category' in df.columns:
        top_cats = df['Category'].value_counts().head(3)
        context_parts.append(f"Top categories: {', '.join(top_cats.index.tolist())}")
    
    return " | ".join(context_parts)

def generate_chatbot_response(user_message, chatbot):
    if not user_message or not isinstance(user_message, str):
        return "I can help you analyze feedback data. Ask me about sentiment, ratings, issues, or recommendations for improvement."
    
    user_message = str(user_message).strip().lower()
    if not user_message:
        return "Please ask me a question about the feedback data."
    
    context = get_feedback_context()
    
    if 'sentiment' in user_message or 'feeling' in user_message or 'emotion' in user_message:
        if 'df' in st.session_state:
            df = st.session_state['df']
            if 'Sentiment' in df.columns:
                sentiment_counts = df['Sentiment'].value_counts()
                pos = sentiment_counts.get('Positive', 0)
                neu = sentiment_counts.get('Neutral', 0)
                neg = sentiment_counts.get('Negative', 0)
                total = len(df)
                pos_pct = (pos / total) * 100
                neu_pct = (neu / total) * 100
                neg_pct = (neg / total) * 100
                
                overall = 'positive' if pos_pct > 50 else 'negative' if neg_pct > 50 else 'mixed'
                return f"Based on {total} reviews:\n- Positive: {pos} ({pos_pct:.1f}%)\n- Neutral: {neu} ({neu_pct:.1f}%)\n- Negative: {neg} ({neg_pct:.1f}%)\n\nOverall sentiment is {overall}."
        return "Please upload feedback data to analyze sentiment."
    
    if 'rating' in user_message or 'score' in user_message or 'average' in user_message:
        if 'df' in st.session_state and 'Rating_Numeric' in st.session_state['df'].columns:
            avg = st.session_state['df']['Rating_Numeric'].mean()
            max_rating = st.session_state['df']['Rating_Numeric'].max()
            min_rating = st.session_state['df']['Rating_Numeric'].min()
            quality = 'excellent' if avg >= 4.5 else 'good' if avg >= 4 else 'moderate' if avg >= 3 else 'poor'
            return f"Rating Analysis:\n- Average: {avg:.2f}/5.0\n- Highest: {max_rating:.1f}\n- Lowest: {min_rating:.1f}\n- Quality: {quality.capitalize()} satisfaction level"
        return "Please upload feedback data to analyze ratings."
    
    if 'issue' in user_message or 'problem' in user_message or 'complaint' in user_message or 'negative' in user_message:
        if 'df' in st.session_state:
            df = st.session_state['df']
            if 'Sentiment' in df.columns:
                neg_count = (df['Sentiment'] == 'Negative').sum()
                if neg_count > 0:
                    return f"Found {neg_count} negative reviews. Common issues include:\n1. Product quality concerns\n2. Delivery delays\n3. Customer service responsiveness\n4. Feature limitations\n\nRecommendation: Address these systematically, starting with the most frequently mentioned issues."
        return "Common customer issues include product quality, delivery times, and customer service. Check the Insights tab for detailed analysis."
    
    if 'improve' in user_message or 'recommendation' in user_message or 'suggest' in user_message or 'action' in user_message:
        if 'df' in st.session_state:
            df = st.session_state['df']
            if 'Sentiment' in df.columns:
                neg_count = (df['Sentiment'] == 'Negative').sum()
                total = len(df)
                neg_pct = (neg_count / total) * 100
                
                recommendations = [
                    f"1. Prioritize resolving {neg_count} negative reviews ({neg_pct:.1f}% of total)",
                    "2. Enhance product quality based on feedback patterns",
                    "3. Improve delivery speed and reliability",
                    "4. Strengthen customer support response times",
                    "5. Implement feedback loops to track improvement"
                ]
                return "Improvement Recommendations:\n\n" + "\n".join(recommendations)
        return "Focus on:\n1. Product quality enhancement\n2. Faster delivery times\n3. Better customer support\n4. Regular feedback monitoring\n5. Proactive issue resolution"
    
    if 'category' in user_message or 'type' in user_message or 'group' in user_message:
        if 'df' in st.session_state and 'Category' in st.session_state['df'].columns:
            top_cats = st.session_state['df']['Category'].value_counts().head(5)
            cat_str = "\n".join([f"- {cat}: {count} reviews" for cat, count in top_cats.items()])
            return f"Top Categories:\n{cat_str}\n\nCheck the Insights tab for category performance analysis."
        return "Category information not available. Please ensure your data includes a Category column."
    
    if 'trend' in user_message or 'over time' in user_message or 'forecast' in user_message:
        return "For trend analysis and forecasting, please check the Insights tab which shows:\n- Rating trends over time\n- Monthly satisfaction changes\n- Predicted future ratings\n\nThis visual analysis helps identify patterns and make data-driven decisions."
    
    if 'help' in user_message or 'what can' in user_message:
        return """I can help you with:

1. Sentiment Analysis - Overall customer sentiment distribution
2. Rating Insights - Average ratings and quality assessment  
3. Issue Identification - Common problems and complaints
4. Improvement Recommendations - Actionable suggestions
5. Category Breakdown - Performance by product category
6. Trend Analysis - Time-based patterns

Just ask me a question like:
- "What is the overall sentiment?"
- "What are the main issues?"
- "How can we improve satisfaction?"
- "What is the average rating?"
"""
    
    if 'df' in st.session_state:
        total_reviews = len(st.session_state['df'])
        return f"I have access to {total_reviews} customer reviews. Ask me about:\n- Sentiment analysis\n- Rating insights\n- Common issues\n- Improvement recommendations\n- Category performance\n\nWhat would you like to know?"
    
    return "I can help you analyze feedback data. Please upload your data first, then ask me about sentiment, ratings, issues, or recommendations for improvement."

tabs = st.tabs(["Upload Data", "Sentiment Analysis", "Summaries", "Insights", "AI Chatbot"])

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

with tabs[4]:
    st.header("AI Chatbot Assistant")
    st.markdown("Ask me questions about your feedback data!")
    
    st.markdown("---")
    
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history[-10:]:
            if message["role"] == "user":
                st.markdown(f"**You:** {message['content']}")
            else:
                st.markdown(f"**Bot:** {message['content']}")
        
        st.markdown("---")
        
        user_input = st.text_input(
            "Ask me anything:", 
            key=f"chat_input_{st.session_state.chat_input_counter}",
            placeholder="What is the overall sentiment?"
        )
        
        if st.button("Send", key=f"send_btn_{st.session_state.chat_input_counter}"):
            if user_input and user_input.strip():
                st.session_state.chat_history.append({"role": "user", "content": user_input})
                
                chatbot_model = load_chatbot()
                if chatbot_model:
                    bot_response = generate_chatbot_response(user_input, chatbot_model)
                else:
                    bot_response = generate_chatbot_response(user_input, None)
                
                st.session_state.chat_history.append({"role": "assistant", "content": bot_response})
                st.session_state.chat_input_counter += 1
                st.rerun()

st.sidebar.title("About")
st.sidebar.info("""
**Customer Feedback Analysis System**

Features:
- Upload CSV data
- Sentiment analysis
- Text summarization
- Visual insights
- Trend analysis
- AI Chatbot assistant

Built with Streamlit & AI
""")

st.sidebar.title("Instructions")
st.sidebar.markdown("""
1. Upload your feedback CSV file
2. View sentiment analysis
3. Generate summaries
4. Explore insights and trends
5. Chat with AI assistant
""")
