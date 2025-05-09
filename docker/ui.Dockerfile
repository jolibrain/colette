FROM python:3.10-slim

WORKDIR /app
COPY src/ ./src/
RUN pip install gradio-i18n gradio-log nltk && python -m nltk.downloader punkt stopwords punkt_tab averaged_perceptron_tagger_eng
EXPOSE 7860
# Run the Gradio app
CMD ["python", "src/colette/app.py"]
