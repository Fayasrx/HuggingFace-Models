# HuggingFace Sentiment Analysis

A sentiment analysis project using HuggingFace Transformers, featuring both a pre-trained inference pipeline and a BERT fine-tuning script on the IMDB dataset. Includes a Flask web app for interactive sentiment prediction.

## Features

- **Web-based sentiment analysis** — Enter any text and get instant sentiment predictions via a Flask UI
- **Pre-trained pipeline** — Uses HuggingFace's `sentiment-analysis` pipeline for zero-setup inference
- **BERT fine-tuning** — Fine-tunes `bert-base-uncased` on the IMDB movie review dataset for custom model training
- **Model export** — Saves the fine-tuned model locally for reuse

## Project Structure

```
├── app.py       # Flask web app for sentiment analysis (pre-trained pipeline)
├── model.py     # BERT fine-tuning script on IMDB dataset
├── .gitignore
└── README.md
```

## Requirements

- Python 3.8+
- Flask
- Transformers (HuggingFace)
- Datasets (HuggingFace)
- PyTorch

### Install Dependencies

```bash
pip install flask transformers datasets torch
```

## Usage

### 1. Run the Web App (Pre-trained Model)

```bash
python app.py
```

Open your browser and navigate to `http://127.0.0.1:5000`. Enter any text in the textarea and click **Analyze** to get the sentiment prediction (POSITIVE/NEGATIVE).

### 2. Fine-tune BERT on IMDB Dataset

```bash
python model.py
```

This script will:
1. Load the IMDB movie review dataset from HuggingFace
2. Tokenize the data using `BertTokenizer`
3. Fine-tune `bert-base-uncased` for binary sentiment classification
4. Train on 2,000 samples and evaluate on 500 samples (configurable)
5. Save the fine-tuned model to `./sentiment_model/`

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Base Model | `bert-base-uncased` |
| Dataset | IMDB (2K train / 500 eval) |
| Epochs | 1 |
| Batch Size | 8 |
| Evaluation | Per epoch |
| Output | `./results/` and `./sentiment_model/` |

## Technologies Used

- **HuggingFace Transformers** — Pre-trained models and fine-tuning
- **HuggingFace Datasets** — IMDB dataset loading
- **Flask** — Lightweight web framework for the inference UI
- **PyTorch** — Deep learning backend
- **BERT** — Bidirectional Encoder Representations from Transformers

## License

This project is open source and available for educational and research purposes.
