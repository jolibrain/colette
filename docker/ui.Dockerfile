FROM python:3.12-slim

WORKDIR /app
COPY src/ ./src/

COPY src/colette/ui_requirements.txt .
RUN pip install -r ui_requirements.txt

RUN python -m nltk.downloader punkt stopwords punkt_tab averaged_perceptron_tagger_eng 
EXPOSE 7860
# Run the Gradio app
CMD ["python", "src/colette/app.py"]