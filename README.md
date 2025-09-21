🛡️ AI-Powered Misinformation Detection & Literacy Assistant — Starter

An advanced AI system that tackles misinformation across text, images, and videos.

Text: Predicts whether content is fake or true with probability scores.

Images: Uses Error Level Analysis (ELA) and metadata to detect tampering.

Videos: Extracts previews and keyframes to highlight possible manipulation.

Designed with a multilingual, India-first approach, the tool empowers newsrooms, fact-checkers, educators, banks, and public agencies to combat scams, fraud, and digital misinformation with greater confidence.

🧱 Project Structure
/
├─ README.md
├─ requirements.txt
├─ LICENSE
├─ app.py                      # Streamlit demo (Text/Image/Video + explanations)
├─ architecture.md             # High-level design (local + GCP-ready)
├─ gcp_setup.md                # Optional: Google Cloud Vertex AI setup
├─ config/
│  └─ .env.example             # Put API keys here (copy to .env)
├─ data/
│  ├─ news_dataset.csv         # Small labeled dataset for text classification
│  └─ README.md
├─ notebooks/
│  ├─ 01_training.ipynb        # Train the text classifier (PyTorch DistilBERT)
│  └─ 02_testing.ipynb         # Load & test the trained model
├─ models/                     # Saved models after training
├─ src/
│  ├─ text_model.py            # Train/save/load PyTorch DistilBERT utilities
│  ├─ utils.py                 # Preprocessing, language detection, translation helpers
│  ├─ image_checks.py          # Image integrity checks (EXIF, recompression, noise)
│  └─ tips.py                  # Media literacy tips and explanations
└─ run_app.sh                   # One-line helper to launch the Streamlit app

🧒 Quick Start: 3 Steps

You will do: (A) Setup, (B) Train, (C) Test & Play

A) Setup

Install Python 3.10+.

Open terminal/Command Prompt.

Navigate to project folder:

cd "AI-Powered Misinformation Detection & Literacy Assistant"


Create a virtual environment:

python -m venv .venv


Activate it:

Windows: .\.venv\Scripts\activate

Mac/Linux: source .venv/bin/activate

Install dependencies:

pip install -r requirements.txt

B) Train the Text Classifier

Open Jupyter Notebook:

python -m notebook


Open notebooks/01_training.ipynb

Run all cells (top to bottom).

After completion, your models/ folder will contain:

distilbert_fake_news/ (PyTorch saved model)

C) Test & Play

Option 1: Notebook

Open notebooks/02_testing.ipynb

Run all cells. Test your own text to see prediction, confidence, and explanation.

Option 2: Streamlit App

Launch via terminal:

./run_app.sh


(Use chmod +x run_app.sh if needed)

Browser opens Streamlit app.

Use the Text, Image, and Video tabs:

Text → Get FAKE / TRUE prediction with confidence

Image → Run ELA + metadata analysis

Video → Extract keyframes and detect anomalies

🌏 India-first & Multilingual Support

Automatically detects language.

Translates non-English text to English for the classifier.

Supports simple output in 10 Indian languages (Hindi, Marathi, Bengali, Tamil, Telugu, Kannada, Malayalam, Gujarati, Punjabi, Urdu).

Optional: For enhanced explanations, connect Google Cloud Vertex AI via .env (see gcp_setup.md).

🧪 Features

Included:

PyTorch DistilBERT text classifier

Transparent explanations for text predictions

Image integrity checks (EXIF, recompression, noise heuristics)

Video keyframe extraction for possible manipulation

Media literacy tips after each prediction

Not included:

Web evidence gathering / reverse image search

Full video forensics

Automated human judgment replacement

🧯 Safety & Ethics

Educational purpose only.

Always review predictions with a human editor/fact-checker.

Respect privacy; do not use to profile or target individuals.

🛠️ Developer Notes

Python 3.10+ recommended

Lightweight starter for prototyping

Modular code allows swapping models, datasets, or adding GCP integration

🚀 Start exploring and building media literacy tools!