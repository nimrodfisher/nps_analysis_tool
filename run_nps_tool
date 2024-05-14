import streamlit as st
import pandas as pd
import plotly.express as px
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter

# NLTK resources download
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Initialize NLTK's sentiment analyzer
sia = SentimentIntensityAnalyzer()

def load_data(uploaded_file):
    """Load data from a CSV file and convert date column to datetime."""
    data = pd.read_csv(uploaded_file)
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce').dt.date
    return data

def calculate_nps(df):
    """Assign NPS categories based on scores and add as a new 'Category' column."""
    def nps_category(score):
        if score >= 9:
            return 'Promoter'
        elif score >= 7:
            return 'Passive'
        else:
            return 'Detractor'
    df['Category'] = df['Score'].apply(nps_category)  # Correcting to 'Category'
    return df


def perform_sentiment_analysis(df):
    """Perform sentiment analysis and add sentiment labels to the dataframe."""
    df['Sentiment'] = df['Feedback'].apply(lambda x: sia.polarity_scores(x)['compound']).apply(
        lambda score: 'Positive' if score >= 0.05 else 'Negative' if score <= -0.05 else 'Neutral'
    )
    return df


def plot_nps_trend(df, interval):
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    nps_trend = df.groupby([pd.Grouper(freq=interval), 'Category']).size().unstack(fill_value=0)

    # Calculate percentages
    total_responses_per_period = nps_trend.sum(axis=1)
    nps_trend_percentage = nps_trend.div(total_responses_per_period, axis=0) * 100

    # Ensure both Promoter and Detractor columns exist
    if 'Promoter' not in nps_trend_percentage.columns:
        nps_trend_percentage['Promoter'] = 0
    if 'Detractor' not in nps_trend_percentage.columns:
        nps_trend_percentage['Detractor'] = 0

    # Calculate NPS
    nps_trend['NPS'] = nps_trend_percentage['Promoter'] - nps_trend_percentage['Detractor']

    # Plot NPS trend
    fig = px.line(nps_trend, y='NPS', title='NPS Trend Over Time',
                  labels={'index': 'Date', 'value': 'Net Promoter Score'})
    st.plotly_chart(fig, use_container_width=True)


def extract_keywords(df, category):
    """Extract keywords from feedback based on the selected category and calculate their frequency."""
    feedback_text = df[df['Category'] == category]['Feedback'].tolist()
    words = []
    for text in feedback_text:
        word_tokens = word_tokenize(text.lower())
        filtered_words = [word for word in word_tokens if word.isalpha() and word not in stop_words]
        words.extend(filtered_words)
    return Counter(words)

def plot_keywords_bubble_chart(keyword_counts, title):
    """Display keywords frequency as a bubble chart."""
    if not keyword_counts:
        st.write("No keywords found for this category.")
        return
    keywords_df = pd.DataFrame(keyword_counts.most_common(20), columns=['Word', 'Frequency'])
    fig = px.scatter(keywords_df, x='Word', y='Frequency', size='Frequency', color='Word', title=title, size_max=60)
    st.plotly_chart(fig, use_container_width=True)

def main():
    """Main function to run the Streamlit app."""
    st.title('NPS Analyzer Tool')  # App header

    st.sidebar.title('Filter Options')
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type="csv")

    if uploaded_file is not None:
        global data
        data = load_data(uploaded_file)

        # data['Date'] = pd.to_datetime(data['Date'], errors='coerce')

        if 'Date' in data.columns and not data['Date'].isna().all():
            # Filter out NaT values if present and get unique dates
            unique_dates = sorted(data.dropna(subset=['Date'])['Date'].unique())
            if unique_dates:
                # Use the sorted unique dates for the slider options
                min_date, max_date = st.sidebar.select_slider(
                    'Select the date range',
                    options=unique_dates,
                    value=(unique_dates[0], unique_dates[-1])
                )
                # Filter data based on selected date range
                data = data[(data['Date'] >= min_date) & (data['Date'] <= max_date)]
            else:
                st.error("No valid dates available for selection.")
        else:
            st.error("Date column is missing or contains invalid data.")

        selected_products = st.sidebar.multiselect('Select Product(s)', options=data['Product'].unique())

        interval = st.sidebar.selectbox('Select aggregation interval', ['D', 'W', 'M'])
        category_for_keywords = st.sidebar.selectbox('Select a category for keywords analysis', ['Detractor', 'Passive', 'Promoter'])

        if st.sidebar.button('Analyze Data'):
            if min_date and max_date:
                data = data[(data['Date'] >= min_date) & (data['Date'] <= max_date)]
            if selected_products:
                data = data[data['Product'].isin(selected_products)]


            data = calculate_nps(data)
            data = perform_sentiment_analysis(data)

            st.subheader('NPS Trend Over Time')
            st.info("NPS Trend Over Time: This chart shows the net promoter score trend calculated as the percentage of promoters minus the percentage of detractors over time.")
            plot_nps_trend(data, interval)

            st.subheader('Sentiment Analysis Results for All Feedback')
            st.info("Sentiment Analysis Results: This section visualizes the count and proportion of feedback classified as Positive, Neutral, or Negative based on sentiment analysis.")
            sentiment_counts = data['Sentiment'].value_counts()
            total_feedback = sentiment_counts.sum()
            sentiment_props = sentiment_counts


            # Calculate and display the proportion of each sentiment category
            sentiment_props = (sentiment_counts / total_feedback * 100).round(2)
            sentiment_data = pd.DataFrame({'Sentiment': sentiment_counts.index, 'Counts': sentiment_counts.values,
                                           'Proportion': sentiment_props.values})
            fig = px.bar(sentiment_data, x='Sentiment', y='Counts', text='Proportion',
                         hover_data=['Proportion'], labels={'Proportion': '% of Total Feedback'},
                         title="Overall Sentiment Analysis")
            fig.update_traces(texttemplate='%{text}%', textposition='outside')
            st.plotly_chart(fig)

            st.subheader('Keyword Analysis for Selected Category')
            st.info("Keyword Analysis: Displays the most frequent words used in the feedback from a selected product, helping to pinpoint common themes or issues.")
            # Select category for keyword analysis
            selected_category = category_for_keywords
            # data['Category'] = data['Score'].apply(calculate_nps(data))
            print (selected_category)
            keyword_counts = extract_keywords(data, selected_category)
            # Filter data based on selected category
            # filtered_keywords = data[data['Category'] == selected_category]['Keywords']

            # Extract keywords based on selected category
            keyword_counts = extract_keywords(data, selected_category)

            # Plot if there are keywords to display
            if keyword_counts:
                plot_keywords_bubble_chart(keyword_counts, f'Keyword Frequency for Category {selected_category}')
            else:
                st.write("No keywords found for this category.")

            # Explanation bubbles for each section

if __name__ == '__main__':
    main()
