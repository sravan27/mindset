#!/usr/bin/env python3
"""
MINDSET Web Application Deployment
Creates a Streamlit web app to visualize the news articles and their transparency metrics
"""

import os
import sys
import json
import argparse
import boto3
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv(".env.aws")

# AWS Configuration
AWS_REGION = os.getenv("AWS_REGION", "eu-north-1")
S3_BUCKET = os.getenv("S3_BUCKET_DATA")
RESOURCE_PREFIX = os.getenv("RESOURCE_PREFIX")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")

def init_aws_clients():
    """Initialize AWS clients"""
    s3 = boto3.client('s3', region_name=AWS_REGION)
    ssm = boto3.client('ssm', region_name=AWS_REGION)
    return s3, ssm

def load_data_from_s3(s3):
    """Load processed data from S3 Gold layer"""
    try:
        # Download gold layer files
        s3.download_file(S3_BUCKET, "gold/news.csv", "gold_news.csv")
        s3.download_file(S3_BUCKET, "gold/behaviors.csv", "gold_behaviors.csv")
        
        # Load into pandas
        news = pd.read_csv("gold_news.csv")
        behaviors = pd.read_csv("gold_behaviors.csv")
        
        return news, behaviors
    except Exception as e:
        st.error(f"Error loading data from S3: {e}")
        return None, None

def create_streamlit_app():
    """Create the Streamlit application"""
    # Create a simple app.py file
    app_code = """
import os
import pandas as pd
import streamlit as st
import boto3
import json
import altair as alt
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(
    page_title="MINDSET - News Analytics Platform",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define AWS variables
AWS_REGION = os.getenv("AWS_REGION", "eu-north-1")
S3_BUCKET = os.getenv("S3_BUCKET_DATA")

# Functions to load data
@st.cache_data
def load_data():
    # For demo, load from local files
    if os.path.exists("gold_news.csv") and os.path.exists("gold_behaviors.csv"):
        news = pd.read_csv("gold_news.csv")
        behaviors = pd.read_csv("gold_behaviors.csv")
        return news, behaviors
    else:
        # Try loading from S3
        try:
            s3 = boto3.client('s3', region_name=AWS_REGION)
            s3.download_file(S3_BUCKET, "gold/news.csv", "gold_news.csv")
            s3.download_file(S3_BUCKET, "gold/behaviors.csv", "gold_behaviors.csv")
            news = pd.read_csv("gold_news.csv")
            behaviors = pd.read_csv("gold_behaviors.csv")
            return news, behaviors
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return None, None

# Load the data
news_df, behaviors_df = load_data()

# Navigation sidebar
st.sidebar.title("MINDSET")
st.sidebar.subheader("News Analytics Platform")
page = st.sidebar.radio("Navigate", ["Dashboard", "News Articles", "Analytics", "About"])

# Show appropriate page
if page == "Dashboard":
    st.title("📊 MINDSET Dashboard")
    st.write("Welcome to MINDSET - your news analytics platform with transparency metrics.")
    
    if news_df is not None:
        col1, col2, col3 = st.columns(3)
        
        # Key metrics
        with col1:
            st.metric("News Articles", len(news_df))
        with col2:
            st.metric("Categories", news_df['category'].nunique())
        with col3:
            st.metric("Average Political Influence", f"{news_df['political_influence'].mean():.1f}/100")
        
        # Distribution of metrics
        st.subheader("Distribution of Transparency Metrics")
        metrics_df = pd.DataFrame({
            'Political Influence': news_df['political_influence'],
            'Rhetoric Intensity': news_df['rhetoric_intensity'],
            'Information Depth': news_df['information_depth']
        })
        
        # Create histogram
        chart = alt.Chart(metrics_df.melt()).mark_bar().encode(
            alt.X('value:Q', bin=True, title='Score (0-100)'),
            alt.Y('count()', title='Count'),
            alt.Color('variable:N', title='Metric'),
            tooltip=['variable', 'value', 'count()']
        ).properties(
            width=700,
            height=400
        )
        st.altair_chart(chart, use_container_width=True)
        
        # News by category
        st.subheader("News by Category")
        cat_counts = news_df['category'].value_counts().reset_index()
        cat_counts.columns = ['category', 'count']
        
        cat_chart = alt.Chart(cat_counts).mark_bar().encode(
            x=alt.X('count:Q', title='Count'),
            y=alt.Y('category:N', sort='-x', title='Category'),
            color=alt.Color('category:N', legend=None),
            tooltip=['category', 'count']
        ).properties(
            width=700,
            height=400
        )
        st.altair_chart(cat_chart, use_container_width=True)
    else:
        st.error("Failed to load data. Please make sure the pipeline has been run.")

elif page == "News Articles":
    st.title("📰 News Articles")
    st.write("Browse and explore news articles with transparency metrics.")
    
    if news_df is not None:
        # Search and filter options
        col1, col2 = st.columns([3, 1])
        with col1:
            search_term = st.text_input("Search articles", "")
        with col2:
            category_filter = st.selectbox("Category", ["All"] + list(news_df['category'].unique()))
        
        # Filter dataframe
        filtered_df = news_df
        if search_term:
            filtered_df = filtered_df[
                filtered_df['title'].str.contains(search_term, case=False) | 
                filtered_df['abstract'].str.contains(search_term, case=False)
            ]
        if category_filter != "All":
            filtered_df = filtered_df[filtered_df['category'] == category_filter]
        
        st.write(f"Showing {len(filtered_df)} of {len(news_df)} articles")
        
        # Display articles
        for i, row in filtered_df.head(10).iterrows():
            with st.expander(f"{row['title']}"):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**{row['title']}**")
                    st.write(row['abstract'])
                    st.caption(f"Category: {row['category']} | Subcategory: {row['subcategory']}")
                    if 'url' in row and row['url']:
                        st.markdown(f"[Read full article]({row['url']})")
                with col2:
                    st.metric("Political Influence", f"{row['political_influence']}/100")
                    st.metric("Rhetoric Intensity", f"{row['rhetoric_intensity']}/100")
                    st.metric("Information Depth", f"{row['information_depth']}/100")
    else:
        st.error("Failed to load data. Please make sure the pipeline has been run.")

elif page == "Analytics":
    st.title("🔍 News Analytics")
    st.write("Advanced analytics of news articles and transparency metrics.")
    
    if news_df is not None:
        # Analysis tabs
        tab1, tab2, tab3 = st.tabs(["Metrics Analysis", "Category Analysis", "Word Clouds"])
        
        with tab1:
            st.subheader("Correlation between metrics")
            
            # Create correlation matrix
            metrics_corr = news_df[['political_influence', 'rhetoric_intensity', 'information_depth']].corr()
            st.dataframe(metrics_corr)
            
            # Scatter plot
            st.subheader("Relationship between metrics")
            x_metric = st.selectbox("X-axis metric", ['political_influence', 'rhetoric_intensity', 'information_depth'])
            y_metric = st.selectbox("Y-axis metric", ['rhetoric_intensity', 'information_depth', 'political_influence'])
            
            scatter = alt.Chart(news_df).mark_circle(size=60).encode(
                x=alt.X(f'{x_metric}:Q', title=x_metric.replace('_', ' ').title()),
                y=alt.Y(f'{y_metric}:Q', title=y_metric.replace('_', ' ').title()),
                color='category:N',
                tooltip=['title', 'category', x_metric, y_metric]
            ).properties(
                width=700,
                height=500
            ).interactive()
            
            st.altair_chart(scatter, use_container_width=True)
        
        with tab2:
            st.subheader("Metrics by Category")
            
            # Calculate average metrics by category
            category_metrics = news_df.groupby('category')[['political_influence', 'rhetoric_intensity', 'information_depth']].mean().reset_index()
            
            # Create a long-form dataframe for plotting
            category_metrics_long = pd.melt(
                category_metrics, 
                id_vars=['category'], 
                value_vars=['political_influence', 'rhetoric_intensity', 'information_depth'],
                var_name='metric', 
                value_name='score'
            )
            
            # Plot metrics by category
            cat_chart = alt.Chart(category_metrics_long).mark_bar().encode(
                x=alt.X('category:N', title='Category'),
                y=alt.Y('score:Q', title='Score'),
                color=alt.Color('metric:N', title='Metric'),
                column=alt.Column('metric:N', title=''),
                tooltip=['category', 'metric', 'score']
            ).properties(
                width=200,
                height=400
            )
            
            st.altair_chart(cat_chart, use_container_width=True)
        
        with tab3:
            st.subheader("Word Clouds by Category")
            selected_category = st.selectbox("Select category", news_df['category'].unique())
            
            # Filter data by category
            category_news = news_df[news_df['category'] == selected_category]
            
            # Combine title and abstract
            text = " ".join(category_news['title'].str.cat(category_news['abstract'], sep=' '))
            
            # Generate word cloud
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
            
            # Display word cloud using matplotlib
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
    else:
        st.error("Failed to load data. Please make sure the pipeline has been run.")

elif page == "About":
    st.title("ℹ️ About MINDSET")
    st.write("""
    MINDSET is an advanced news analytics platform that provides transparency metrics for news articles:
    
    - **Political Influence Level (0-100)**: Measures how politically charged the content is.
    - **Rhetoric Intensity Scale (0-100)**: Quantifies the use of rhetorical devices and emotional language.
    - **Information Depth Score (0-100)**: Evaluates the depth and substance of the information provided.
    
    ### Architecture
    
    MINDSET uses a medallion architecture for data processing:
    
    1. **Raw Layer**: Original data from news APIs and datasets
    2. **Bronze Layer**: Cleaned and standardized data
    3. **Silver Layer**: Feature-rich data with engineered features
    4. **Gold Layer**: Analytics-ready data with calculated metrics
    
    ### Technologies
    
    - AWS S3 for data storage
    - Python data science stack (Pandas, Scikit-learn)
    - ML models for metric calculations
    - Streamlit for web interface
    
    ### Datasets
    
    MINDSET uses the MINDLarge dataset from Microsoft for news recommendation.
    """)
    
    st.subheader("Contact")
    st.write("For more information, contact the MINDSET team.")

# Footer
st.sidebar.markdown("---")
st.sidebar.info("MINDSET v1.0 - News Analytics Platform")
"""
    
    # Write to file
    with open("app.py", "w") as f:
        f.write(app_code)
    
    # Create requirements.txt for Streamlit
    requirements = """
streamlit==1.35.0
pandas==2.2.0
boto3==1.34.51
numpy==1.26.0
altair==5.2.0
matplotlib==3.8.0
wordcloud==1.9.2
python-dotenv==1.0.1
"""
    with open("streamlit_requirements.txt", "w") as f:
        f.write(requirements)
    
    print("Streamlit application created!")
    print("Run with: streamlit run app.py")

def run_streamlit():
    """Run the Streamlit application"""
    # Install Streamlit if needed
    try:
        import streamlit
    except ImportError:
        print("Installing Streamlit and dependencies...")
        os.system(f"{sys.executable} -m pip install -r streamlit_requirements.txt")
    
    # Run the app
    os.system(f"{sys.executable} -m streamlit run app.py")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="MINDSET Web Application Deployment")
    parser.add_argument("--create-only", action="store_true", help="Only create the app files, don't run")
    parser.add_argument("--run", action="store_true", help="Run the Streamlit app")
    args = parser.parse_args()
    
    # Initialize AWS clients
    s3, ssm = init_aws_clients()
    
    # Load data from S3
    news, behaviors = load_data_from_s3(s3)
    
    if news is None:
        print("Error: Failed to load data from S3. Make sure you've run the pipeline first.")
        return False
    
    # Create the Streamlit app
    create_streamlit_app()
    
    if args.run or not args.create_only:
        # Run the Streamlit app
        run_streamlit()
    
    return True

if __name__ == "__main__":
    main()