import re
import math
import numpy as np
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
import os
# Ensure nltk resources are downloaded
nltk.download('punkt_tab', quiet=True)

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

class TextSummarizer:
    def __init__(self, method='tfidf'):
        # Initialize for TF-IDF or Embedding summarizer
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        self.method = method
        if self.method == 'embedding':
            self.model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        else:
            self.stop_words = set(stopwords.words('english'))
            self.lemmatizer = WordNetLemmatizer()
    
    # Preprocess text for TF-IDF method
    def preprocess_text(self, text):
        return re.sub(r'[^\w\s.]', '', text.lower())
    
    # Tokenize and Lemmatize text for TF-IDF method
    def tokenize_and_lemmatize(self, text):
        words = word_tokenize(text)
        return [
            self.lemmatizer.lemmatize(word) 
            for word in words if word.isalnum() and word not in self.stop_words
        ]
    
    # Calculate TF-IDF for TF-IDF method
    def calculate_tf_idf(self, text):
        sentences = sent_tokenize(text)
        words = self.tokenize_and_lemmatize(text)
        word_freq = Counter(words)
        
        max_freq = max(word_freq.values())
        tf = {word: freq / max_freq for word, freq in word_freq.items()}
        
        num_sentences = len(sentences)
        idf = {}
        
        for word in word_freq.keys():
            sentence_count = sum(1 for sentence in sentences if word in self.tokenize_and_lemmatize(sentence))
            idf[word] = math.log((num_sentences + 1) / (1 + sentence_count))  # Smoothed IDF
            
        tf_idf = {word: tf[word] * idf[word] for word in word_freq.keys()}
        return tf_idf
    
    # Score sentences for TF-IDF method
    def calculate_sentence_scores(self, sentences, tf_idf):
        sentence_scores = {}
        
        for sentence in sentences:
            words = self.tokenize_and_lemmatize(sentence)
            word_count = len(words)
            
            if word_count <= 3:  # Skip very short sentences
                continue
                
            score = sum(tf_idf.get(word, 0) for word in words)
            sentence_scores[sentence] = score / word_count  # Normalize by sentence length
            
        return sentence_scores
    
    # Method for TF-IDF summarization
    def summarize_tfidf(self, text, percentage=0.3):
        processed_text = self.preprocess_text(text)
        sentences = sent_tokenize(text)
        total_sentences = len(sentences)
        
        if total_sentences == 0:
            return ""
        
        num_sentences = max(1, math.ceil(len(sentences) * percentage))
        
        tf_idf = self.calculate_tf_idf(processed_text)
        sentence_scores = self.calculate_sentence_scores(sentences, tf_idf)
        
        ranked_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)
        top_sentences = ranked_sentences[:num_sentences]
        
        selected_sentences = [sentence for sentence, score in sorted(
            top_sentences, key=lambda x: sentences.index(x[0])
        )]
        
        return ' '.join(selected_sentences)
    
    # Method for Embedding-based summarization
    def summarize_embedding(self, text, percentage=0.3):
        sentences = sent_tokenize(text)
        embeddings = self.model.encode(sentences)
        
        similarity_matrix = cosine_similarity(embeddings)
        sentence_scores = similarity_matrix.sum(axis=1)
        
        num_sentences = max(1, math.ceil(len(sentences) * percentage))
        print("Total no of sentences in text are ",len(sentences))
        print("The number of sentences present in the summary are ",num_sentences)
        
        top_sentence_indices = np.argsort(sentence_scores)[-num_sentences:]
        selected_sentences = [sentences[i] for i in sorted(top_sentence_indices)]
        
        return ' '.join(selected_sentences)
    
    # General summarizer method based on selected summarization method
    def summarize(self, text, percentage=0.3):
        if self.method == 'embedding':
            return self.summarize_embedding(text, percentage)
        else:
            return self.summarize_tfidf(text, percentage)


class AbstractiveSummarizer:
    def __init__(self):
        try:
            model_path = "./saved_model"
            if os.path.exists(model_path):
                print("Loading model from local path...")
                self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                self.summarizer = pipeline("summarization", model=self.model, tokenizer=self.tokenizer)
            else:
                print("Local model not found. Using default BART model...")
                self.summarizer = pipeline(
                    "summarization", 
                    model="facebook/bart-large-cnn",
                    early_stopping=True,
                    do_sample=False,
                )
                print("Saving the model to local path for future use...")
                self.model = self.summarizer.model
                self.tokenizer = self.summarizer.tokenizer
                self.save_model(model_path)
        except ImportError:
            print("Warning: transformers package not installed. Falling back to extractive summarization.")
            self.summarizer = None

    def save_model(self, model_path):
        try:
            print(f"Saving model and tokenizer to {model_path}...")
            self.model.save_pretrained(model_path)
            self.tokenizer.save_pretrained(model_path)
            print("Model and tokenizer saved successfully!")
        except Exception as e:
            print(f"Error while saving model: {e}")

    def chunk_text(self, text, chunk_size=1000):
        # Tokenize the text into tokens and split into chunks of max chunk_size
        tokens = self.tokenizer.encode(text, truncation=True, max_length=chunk_size)
        chunks = []
        chunk = []
        
        for token in tokens:
            chunk.append(token)
            if len(chunk) >= chunk_size:
                chunks.append(self.tokenizer.decode(chunk))
                chunk = []
        
        if chunk:
            chunks.append(self.tokenizer.decode(chunk))
        
        # Ensure chunks are not empty and respect the model's token limit
        chunks = [chunk for chunk in chunks if len(self.tokenizer.encode(chunk)) > 0]
        return chunks

    def summarize(self, text):
        if not isinstance(text, str):
            raise ValueError("Input text must be a string")

        # Token count of input text
        token_count = len(self.tokenizer.encode(text))
        print("The number of tokens in the input text:", token_count)

        if token_count < 1000:
            max_length = max(1, int(token_count * 0.35))
            min_length = max(1, int(token_count * 0.25))

            try:
                summary = self.summarizer(
                    text, max_length=max_length, min_length=min_length, do_sample=False, early_stopping=True
                )
                return summary[0]['summary_text']
            except Exception as e:
                print(f"Error during summarization: {e}")
                return "Error during summarization."
        else:
            chunks = self.chunk_text(text)  # Create smaller chunks of text
            summaries = []
            print("The number of chunks formed are ",len(chunks))

            for chunk in chunks:
                token_count_chunk = len(self.tokenizer.encode(chunk))
                max_length = max(1, int(token_count_chunk * 0.35))
                min_length = max(1, int(token_count_chunk * 0.25))

                try:
                    summary = self.summarizer(
                        chunk, max_length=max_length, min_length=min_length, do_sample=False, early_stopping=True
                    )
                    summaries.append(summary[0]['summary_text'])
                except Exception as e:
                    print(f"Error during summarization of chunk: {e}")
                    summaries.append("Error summarizing chunk.")

            return " ".join(summaries)
