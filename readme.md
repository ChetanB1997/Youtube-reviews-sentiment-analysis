# YouTube Comment Sentiment Analysis

This Python project performs sentiment analysis on YouTube video comments using natural language processing techniques. It analyzes the sentiment (positive, negative, or neutral) expressed in the comments to gain insights about the audience's reactions to the video.

## Features

- Scrapes comments from a YouTube video using selenium.
- Preprocesses the comments using NLP by removing noise, such as special characters and stopwords.
- Performs sentiment analysis on the preprocessed comments using a machine learning model.
- Generates visualizations to illustrate the sentiment distribution of the comments.
- Provides summary statistics and insights about the sentiment of the comments.

## Prerequisites

- Python 3.7 or above
- selenium webdriver
- Required Python packages listed in `requirements.txt`

## Installation

1. Clone the repository:

   ```shell
   git clone https://github.com/ChetanB1997/Youtube-reviews-sentiment-analysis.git

## Usage
1.Run the script :
-python main.py

2.Provide the YouTube video URL when prompted.

3.The script will scrape the comments, perform sentiment analysis, and generate visualizations and summary statistics.

4.The sentiment analysis results will be displayed in the console and saved in the output directory.