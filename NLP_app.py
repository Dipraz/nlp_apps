import streamlit as st
import spacy
from textblob import TextBlob
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from transformers import pipeline
from collections import Counter
import pandas as pd
import nltk
nltk.download('punkt')



def get_sentiment_distribution(text):
    """Get the distribution of sentiment in a text."""
    blob = TextBlob(text)
    sentiment_distribution = blob.sentiment.polarity
    return sentiment_distribution

def get_most_common_positive_words(text, n=5):
    """Get the most common positive words in a text."""
    blob = TextBlob(text)
    positive_words = [word.lower() for word, tag in blob.tags if tag.startswith('JJ') and TextBlob(word).sentiment.polarity > 0]
    most_common_positive_words = Counter(positive_words).most_common(n)
    return most_common_positive_words

def get_emotion_distribution(text):
    """Get the distribution of emotions in a text."""
    blob = TextBlob(text)
    emotion_distribution = Counter()
    for sentence in blob.sentences:
        emotion = sentence.sentiment.polarity
        emotion_distribution[emotion] += 1
    return emotion_distribution

def bart_summarizer(text):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summary = summarizer(text, max_length=1000, min_length=30, do_sample=False)
    return summary[0]['summary_text']

def sumy_summarizer(docx):
    parser = PlaintextParser.from_string(docx, Tokenizer("english"))
    lex_summarizer = LexRankSummarizer()
    summary = lex_summarizer(parser.document, 3)
    summary_list = [str(sentence) for sentence in summary]
    result = " ".join(summary_list)
    return result

def text_analyzer(my_text):
    nlp = spacy.load('en_core_web_sm')
    docx = nlp(my_text)
    tokens = [token.text for token in docx]
    allData = [{'Tokens': token.text, 'Lemma': token.lemma_} for token in docx]
    return allData


def entity_analyzer(my_text):
    nlp = spacy.load('en_core_web_sm')
    docx = nlp(my_text)
    tokens = [token.text for token in docx]
    entities = [(entity.text, entity.label_) for entity in docx.ents]
    allData = {'Tokens': tokens, 'Entities': entities}
    return allData


def main():
    """NLP app with Streamlit"""
    st.title("NLP app That Makes You More Productive")
    st.image("iStock-1195731173.jpg", width=400, caption="NLPiffy")
    st.subheader("Natural Language Processing on the Go, Anytime...")

    # Tokenization
    if st.checkbox("Show Tokens and Lemma"):
        st.subheader("Tokenize Your Text")
        st.image("IMG_0079.jpg", caption="Tokenization")
        message = st.text_area("Enter your text", "Type here", key="text_area_token")
        if st.button("Analyze Tokens", key="analyze_tokens"):
            nlp_result = text_analyzer(message)
            st.json(nlp_result)
        else:
            st.warning("Please enter your text")

    # Named Entity Recognition
    if st.checkbox("Show Named Entity"):
        st.subheader("Extract Entities from Your Text")
        st.image("header-img-event-extraction@2x.png", caption="Named Entity Recognition")
        message = st.text_area("Enter your text", "Type here", key="text_area_entity")
        if st.button("Extract Entities", key="extract_entities"):
            if message:
                nlp_result = entity_analyzer(message)
                st.json(nlp_result)
            else:
                st.warning("Please enter your text")

    # Choose the type of sentiment analysis
    if st.checkbox("Show sentiment analysis"):
        # Get the message from the user.
        message = st.text_area("Enter your message", "Type here", key="text_area")

        st.subheader("Choose the type of sentiment analysis")
        st.image("Sentiment-Analysis-Using-Python-i2tutorials.jpg", caption="Tokenization")
        sentiment_type = st.selectbox("Select the type of sentiment analysis", ["Positive", "Negative", "Emotion"])

        # Show the distribution of sentiment
        if sentiment_type == "Positive" or sentiment_type == "Negative":
            st.subheader("Distribution of Sentiment")
            sentiment_distribution = get_sentiment_distribution(message)
            sentiment_data = pd.DataFrame({"Sentiment": [sentiment_distribution]})
            st.bar_chart(sentiment_data)
        elif sentiment_type == "Emotion":
            st.subheader("Distribution of Emotions")
            message = st.text_area("Enter your text", "Type here", key="text_area_emotion")
            emotion_distribution = get_emotion_distribution(message)
            st.bar_chart(emotion_distribution)
        else:
            st.write("The sentiment type is not supported.")

    # Show the most common positive words
    if st.checkbox("Show common words"):
        st.subheader("Most Common Positive Words")
        message = st.text_area("Enter your text", "Type here", key="text_area_common_words")
        most_common_positive_words = get_most_common_positive_words(message)
        st.write(most_common_positive_words)

    # Text Summarization
    if st.checkbox("Show Summarization"):
        st.subheader("Summarize Your Text")
        st.image("python-text-summarization.jpg", caption="Text Summarization")
        message = st.text_area("Enter your text", "Type here", key="text_area_summarize")
        summary_options = st.selectbox("Choose your summarizer", ["Summarizer", "Sumy", "BART"])
        if st.button("Summarize Text", key="summarize_text"):
            if summary_options == "Summarizer":
                st.text("Using Summarizer...")
                summary_result = sumy_summarizer(message)
                st.success(summary_result)
            elif summary_options == "Sumy":
                st.text("Using Sumy Summarizer...")
                summary_result = sumy_summarizer(message)
                st.success(summary_result)
            elif summary_options == "BART":
                st.text("Using BART Summarizer...")
                summary_result = bart_summarizer(message)
                st.success(summary_result)
            else:
                st.warning("Please select a summarization method")

    # About the App
    st.markdown("Developed by [DipRaj](https://github.com/Dipraz)")

    st.sidebar.subheader("About the App")
    st.sidebar.text("The NLP App provides a user-friendly interface that makes it easy to perform various NLP tasks.")
    st.sidebar.info("Credits to the Streamlit team")

    st.sidebar.subheader("Thanks for using NLP_APP")


if __name__ == "__main__":
    main()