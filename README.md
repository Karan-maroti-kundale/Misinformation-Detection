# 🛡️ AI-Powered Misinformation Detection & Literacy Assistant

An advanced, multilingual AI system designed to **detect and explain misinformation** across **text, images, and videos** — with an **India-first** approach to empower newsrooms, fact-checkers, educators, banks, and public agencies.

---

## 🌟 Overview

This tool helps identify potential fake news, tampered images, or manipulated videos.   
It combines transformer-based NLP models (DistilBERT), media integrity checks, and explainable AI insights for literacy and transparency.

### Core Capabilities

- **Text:** Predicts whether content is *FAKE* or *TRUE* with probability scores.  
- **Images:** Uses **Error Level Analysis (ELA)**, **EXIF metadata**, and **noise** detection for tampering analysis.  
- **Videos:** Extracts keyframes and previews to highlight possible manipulations.

---

## 🧱 Project Structure

/
├─ README.md
├─ requirements.txt
├─ LICENSE
├─ app.py # Streamlit demo (Text/Image/Video + explanations)
├─ architecture.md # High-level design (local + GCP-ready)
├─ gcp_setup.md # Optional: Google Cloud Vertex AI setup
├─ config/
│ └─ .env.example # Place API keys here (copy to .env)
├─ data/
│ ├─ news_dataset.csv # Small labeled dataset for text classification
│ └─ README.md
├─ notebooks/
│ ├─ 01_training.ipynb # Train the text classifier (PyTorch DistilBERT)
│ └─ 02_testing.ipynb # Load & test the trained model
├─ models/ # Saved models after training
├─ src/
│ ├─ text_model.py # Train/save/load PyTorch DistilBERT utilities
│ ├─ utils.py # Preprocessing, language detection, translation helpers
│ ├─ image_checks.py # Image integrity checks (EXIF, recompression, noise)
│ └─ tips.py # Media literacy tips and explanations
└─ run_app.sh # One-line helper to launch the Streamlit app

text

---

## 🚀 Quick Start (3 Steps)

### A) Setup

1. **Install Python 3.10+**
2. Open terminal or CMD and navigate to the project folder:
cd "AI-Powered Misinformation Detection & Literacy Assistant"

3. **Create a virtual environment:**

python -m venv .venv

4. **Activate it:**

- **Windows:**  
  ```
  .\.venv\Scripts\activate
  ```
  
5. **Install dependencies:**
pip install -r requirements.txt

---

### B) Train the Text Classifier

1. **Open Jupyter Notebook:**
python -m notebook
2. Open `notebooks/01_training.ipynb`  
3. Run all cells (top to bottom)
After training, your `models/` folder will contain:
distilbert_fake_news/ # PyTorch saved model

---

### C) Test & Play

You have two ways to test your model.

#### Option 1: Jupyter Notebook
1. Open `notebooks/02_testing.ipynb`
2. Run all cells and test your text samples  
   → See prediction, confidence, and reasoning.

#### Option 2: Streamlit App
1. Launch via terminal:
./run_app.sh
2. Browser will launch automatically.

3. Explore tabs:
- **Text:** Get *FAKE / TRUE* prediction with confidence  
- **Image:** Run ELA and metadata analysis  
- **Video:** Extract keyframes & detect anomalies

---

## 🛠 Developer Notes

- Recommended Python: **3.10+**  
- Lightweight, modular prototype for AI-driven misinformation detection  
- Compatible with local systems and scalable to **Google Cloud (Vertex AI)**  
- Easily extendable with custom datasets, multilingual models, or new features  

---

## 📜 License

This project is released under the **MIT License** — see [LICENSE](./LICENSE) for details.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!  
Feel free to fork, modify, and submit pull requests.

---

### 🚀 Start exploring and building media literacy tools today!
