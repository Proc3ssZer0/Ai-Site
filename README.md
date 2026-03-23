# 🎬 Ai-Site – AI Movie Recommender by Mood

A local web application built with Flask that uses a neural network to analyze your mood (via text or emojis) and suggests movies that match your emotional state. All processing is done locally – no external APIs.

## ✨ Features

- Mood detection from text or emotion selection
- Movie recommendations from a local database
- Fully offline after the initial model download
- Simple web interface

## 📋 Requirements

- Python 3.8 or higher
- Dependencies listed in `requirements.txt`
- (Optional) GPU with CUDA for faster inference

## 🛠 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your_username/Ai-Site.git
   cd Ai-Site

1. Create a virtual environment (recommended):
    ```bash
     python -m venv venv
   source venv/bin/activate   # Linux/macOS
   venv\Scripts\activate      # Windows
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
4. The first time you run the app, the neural network model will be downloaded automatically (about 200–500 MB). An internet connection is required only at that moment.

▶️ Running the App

Start the Flask server with:
python main.py

You will see output like:
 * Running on http://127.0.0.1:5000

Open that address in your browser.


⚠️ Notes

· The first launch may take extra time to download the model.
· For faster inference, use a GPU with CUDA support (if available).
· The default movie database contains about 500 entries. You can extend it by adding your own JSON or CSV files.

📄 License

This project is licensed under the GNU General Public License v3.0 – see the LICENSE file for details.

---

🐛 If you encounter any issues, please open an issue on GitHub.
