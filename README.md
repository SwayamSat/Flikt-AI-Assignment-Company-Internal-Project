# Intelligent Customer Feedback Analysis System

AI-powered system for analyzing, summarizing, and predicting customer sentiment from feedback data.

## Project Overview

This project implements a complete end-to-end machine learning pipeline for customer feedback analysis, including data preprocessing, sentiment classification, text summarization, predictive insights, and a web-based deployment interface.

## Features

- **Data Processing**: Automated cleaning, tokenization, and preprocessing
- **Sentiment Analysis**: AI-powered classification using DistilBERT
- **Text Summarization**: Transformer-based (T5) and extractive summarization
- **Predictive Insights**: Recurring issue identification and satisfaction forecasting
- **Web Dashboard**: Interactive Streamlit application for real-time analysis

## Project Structure

```
customer-feedback-analysis/
├── data_preprocessing.ipynb          # Part 1: Data handling
├── sentiment_model.ipynb             # Part 2: Sentiment classification
├── text_summarization.ipynb          # Part 3: Text summarization
├── predictive_insights.ipynb         # Part 4: Predictive analytics
├── app.py                            # Part 5: Streamlit web app
├── requirements.txt                  # Python dependencies
├── Customer_Feedback.csv             # Original dataset
├── cleaned_customer_feedback.csv     # Processed dataset
├── run_app.bat                       # Windows run script
├── RUN_APP.md                        # App instructions
└── README.md                         # This file
```

## Installation

### Prerequisites
- Python 3.10 or higher
- 8GB RAM minimum (for transformer models)
- Internet connection for model downloads

### Setup

1. **Clone the repository:**
```bash
git clone https://github.com/YOUR_USERNAME/customer-feedback-analysis.git
cd customer-feedback-analysis
```

2. **Create virtual environment:**
```bash
python -m venv venv
```

3. **Activate virtual environment:**

**Windows:**
```bash
.\venv\Scripts\activate
```

**Linux/Mac:**
```bash
source venv/bin/activate
```

4. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## Usage

### Part 1: Data Preprocessing
```bash
jupyter notebook data_preprocessing.ipynb
```
- Loads and cleans customer feedback data
- Performs tokenization and lemmatization
- Outputs: `cleaned_customer_feedback.csv`

### Part 2: Sentiment Classification
```bash
jupyter notebook sentiment_model.ipynb
```
- Trains DistilBERT model for sentiment analysis
- Evaluates with accuracy, precision, recall, F1 score
- Outputs: `sentiment_model/` directory, `sentiment_model.pkl`

### Part 3: Text Summarization
```bash
jupyter notebook text_summarization.ipynb
```
- Implements T5 transformer summarization
- Provides extractive summarization fallback
- Outputs: `summarized_feedback.csv`, `summarization_results.json`

### Part 4: Predictive Insights
```bash
jupyter notebook predictive_insights.ipynb
```
- Identifies recurring issues and complaints
- Forecasts customer satisfaction trends
- Outputs: `AI_insights_report.pdf`, visualization charts

### Part 5: Web Application
```bash
streamlit run app.py
```
Or use the provided script:
```bash
run_app.bat
```

Access at: http://localhost:8501

## Web Application Features

### Upload Data
- CSV file upload functionality
- Dataset preview and statistics
- Automatic validation

### Sentiment Analysis
- Real-time sentiment classification
- Visual distribution charts (bar & pie)
- Sample review browser by sentiment

### Text Summarization  
- AI-powered summary generation
- Original vs summary comparison
- Batch processing support

### Insights & Analytics
- Key metrics dashboard
- Category performance analysis
- Time-based trend visualization
- Recurring issue identification

## Dataset

**Source**: Customer reviews for smartphone products  
**Size**: 571 records  
**Columns**:
- Review Title
- Customer Name
- Rating (1-5 stars)
- Date
- Category (Display, Camera, Battery, Others, Delivery)
- Comments
- Useful (helpfulness votes)

## Technologies Used

### Machine Learning & NLP
- **Transformers**: DistilBERT, T5
- **scikit-learn**: Text vectorization, regression
- **NLTK**: Tokenization, stopwords

### Data Processing
- **pandas**: Data manipulation
- **numpy**: Numerical computing

### Visualization
- **matplotlib**: Static plots
- **seaborn**: Statistical graphics
- **plotly**: Interactive charts (optional)

### Web Framework
- **Streamlit**: Interactive web application

## Model Performance

### Sentiment Classification
- **Model**: DistilBERT (base-uncased)
- **Classes**: Positive, Neutral, Negative
- **Metrics**: Accuracy, Precision, Recall, F1 Score

### Summarization
- **Model**: T5-small
- **Compression**: 60-70% text reduction
- **Methods**: Abstractive (T5) + Extractive (TF-IDF)

### Forecasting
- **Model**: Linear Regression
- **Output**: Next month satisfaction prediction
- **Visualization**: Trend charts with forecast line

## Requirements

Main dependencies (see `requirements.txt` for full list):
```
pandas
numpy
nltk
scikit-learn
matplotlib
seaborn
transformers
torch
streamlit
```

## Running the Complete Pipeline

1. **Data Preprocessing**: Run Part 1 notebook
2. **Model Training**: Run Parts 2 & 3 notebooks
3. **Analysis**: Run Part 4 notebook  
4. **Deployment**: Launch web app with Part 5

## Troubleshooting

### Common Issues

**Slow model loading:**
- Models download on first run (~250MB for DistilBERT)
- Subsequent runs use cached models

**Memory errors:**
- Reduce batch size in training
- Use CPU instead of GPU if limited VRAM
- Process data in smaller chunks

**Import errors:**
- Ensure virtual environment is activated
- Reinstall requirements: `pip install -r requirements.txt`

## Future Enhancements

- Real-time feedback processing
- Multi-language support
- Advanced forecasting (ARIMA, Prophet)
- Email alert system for negative reviews
- API endpoint for integration

## License

This project is for educational purposes.

## Acknowledgments

- Dataset: Customer product reviews
- Models: Hugging Face Transformers
- Framework: Streamlit

## Contact

For questions or feedback, please open an issue on GitHub.



