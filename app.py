
from flask import Flask, request, jsonify
import os
import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModel
import csv
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import requests
from bs4 import BeautifulSoup

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)

# Load the sentiment analysis model
model = pipeline('sentiment-analysis')
tokenizer = AutoTokenizer.from_pretrained("Hammad358/distill-bert-revised")
Model = AutoModel.from_pretrained("Hammad358/distill-bert-revised")

# Stop words
stop_words = set(stopwords.words('english'))

# Aspects for analysis
aspects = [
    'Usability', 'Performance', 'Bug', 'Security', 'Community',
    'Compatibility', 'Documentation', 'Legal', 'Portability',
    'OnlySentiment', 'Others'
]

# Preprocess the text for analysis
def preprocess_text(text):
    if len(text.split()) <= 2:  # Skip preprocessing for very short texts
        return text
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    tokens = word_tokenize(text)
    tokens = [word.lower() for word in tokens if word.lower() not in stop_words]
    return ' '.join(tokens)


# Fetch data from StackOverflow threads
def fetch_stackoverflow_thread(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        threads = []
        # Locate all the search result elements
        search_results = soup.find_all('div', class_='s-post-summary')

        for result in search_results:
            # Extract title and link
            title_element = result.find('a', class_='s-link')
            if title_element:
                title = title_element.get_text(strip=True)
                link = f"https://stackoverflow.com{title_element['href']}"
            else:
                title, link = None, None
            
            # Extract excerpt (optional)
            excerpt_element = result.find('div', class_='s-post-summary--content-excerpt')
            excerpt = excerpt_element.get_text(strip=True) if excerpt_element else None

            # Append the extracted data
            threads.append({
                'title': title,
                'url': link,
                'excerpt': excerpt
            })

        if not threads:
            raise Exception("No threads found. The HTML structure might have changed.")

        return threads
    except Exception as e:
        raise Exception(f"Error while fetching thread data: {str(e)}")

def save_to_csv(data, filename):
    try:
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Title', 'URL', 'Excerpt'])
            for item in data:
                writer.writerow([item['title'], item['url'], item['excerpt']])
    except Exception as e:
        raise Exception(f"Error while saving to CSV: {str(e)}")

# Fetch data from a single StackOverflow question
def fetch_stackoverflow_data(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract the question and answers
        question_body = soup.find('div', class_='js-post-body').get_text().strip()
        answers = soup.find_all('div', class_='answer')
        answer_texts = [answer.find('div', class_='js-post-body').get_text().strip() for answer in answers]

        # Save the data to a CSV file
        csv_filename = 'question_data.csv'
        with open(csv_filename, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['Answer'])
            writer.writerow([question_body])  # Include question as an answer
            for answer_text in answer_texts:
                writer.writerow([answer_text])

        return csv_filename
    except Exception as e:
        raise Exception(f"Error scraping the URL: {str(e)}")

# Process CSV data for aspect analysis
def process_data_from_file(csv_filename):
    aspect_sentiments = {aspect: {'positive': 0, 'negative': 0} for aspect in aspects}
    try:
        data = pd.read_csv(csv_filename)

        for _, row in data.iterrows():
            text = row.get('Excerpt') or row.get('Text') or row.get('Answer')
            if text:
                preprocessed_text = preprocess_text(text)
                sentiment = model(preprocessed_text[:512])  # Limit to 512 tokens
                sentiment_label = sentiment[0]['label']

                for aspect in aspects:
                    if aspect.lower() in preprocessed_text:
                        if sentiment_label == 'POSITIVE':
                            aspect_sentiments[aspect]['positive'] += 1
                        else:
                            aspect_sentiments[aspect]['negative'] += 1

        # Convert sentiment counts to percentages
        aspect_percentages = {
            aspect: {
                'positive': (counts['positive'] / max(counts['positive'] + counts['negative'], 1)) * 100,
                'negative': (counts['negative'] / max(counts['positive'] + counts['negative'], 1)) * 100,
            }
            for aspect, counts in aspect_sentiments.items()
        }

        return jsonify(aspect_percentages)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# API route to handle thread input
@app.route('/fetch-threads', methods=['POST'])
def handle_thread_input():
    try:
        data = request.get_json()
        url = data.get('url')
        if not url:
            return jsonify({'error': 'URL is required'}), 400

        csv_filename = fetch_stackoverflow_thread(url)
        return process_data_from_file(csv_filename)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# API route to handle question URLs
@app.route('/fetch-data', methods=['POST'])
def handle_question_input():
    try:
        data = request.get_json()
        url = data.get('url')
        if not url:
            return jsonify({'error': 'URL is required'}), 400

        csv_filename = fetch_stackoverflow_data(url)
        return process_data_from_file(csv_filename)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/fetch-thread-data', methods=['POST'])
def handle_fetch_thread_data():
    try:
        data = request.get_json()
        print("Received payload:", data)
        url = data.get('url')

        if not url:
            return jsonify({'error': 'URL is required'}), 400

        if not url.startswith('https://stackoverflow.com/search?q='):
            return jsonify({'error': 'Invalid URL format'}), 400

        extracted_data = fetch_stackoverflow_thread(url)
        print("Extracted data:", extracted_data)

        csv_filename = 'stackoverflow_threads.csv'
        save_to_csv(extracted_data, csv_filename)
        return jsonify({'message': f'Data saved to {csv_filename}', 'data': extracted_data}), 200
    except Exception as e:
        print("Error:", str(e))
        return jsonify({'error': str(e)}), 500

# API route to handle text input
@app.route('/process-text', methods=['POST'])
def handle_text_input():
    try:
        data = request.get_json()
        text = data.get('text')
        if not text:
            return jsonify({'error': 'Text is required'}), 400

        # Save the input text to a CSV
        csv_filename = 'text_data.csv'
        with open(csv_filename, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['Text'])
            writer.writerow([text])

        return process_data_from_file(csv_filename)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5004, debug=True)
