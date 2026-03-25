from flask import Flask, render_template, request, session
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import re

sentiment_analyzer = pipeline("sentiment-analysis", model="blanchefort/rubert-base-cased-sentiment")

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'

tokenizer = AutoTokenizer.from_pretrained("sberbank-ai/rugpt3medium_based_on_gpt2")
model = AutoModelForCausalLM.from_pretrained("sberbank-ai/rugpt3medium_based_on_gpt2")

# Тексты интерфейса и mood
texts = {
    'en': {
        'title': 'My first AI site',
        'label': 'Write your feelings',
        'placeholder': 'e.g., I\'m fine!',
        'submit': 'Submit',
        'result_title': 'Recommendation:',
        'lang_switch': 'Switch to Русский',
        'lang_target': 'ru',
        'moods': {
            'POSITIVE': 'good',
            'NEGATIVE': 'bad',
            'NEUTRAL': 'neutral'
        }
    },
    'ru': {
        'title': 'Мой первый AI-сайт',
        'label': 'Напишите свои чувства',
        'placeholder': 'Например, мне хорошо!',
        'submit': 'Отправить',
        'result_title': 'Рекомендация:',
        'lang_switch': 'Switch to English',
        'lang_target': 'en',
        'moods': {
            'POSITIVE': 'хорошее',
            'NEGATIVE': 'плохое',
            'NEUTRAL': 'нейтральное'
        }
    }
}

def extract_film_title(generated_text: str) -> str:
    """Extract movie title from generated text."""
    if not generated_text:
        return ""
    # Поиск в кавычках (русских или английских)
    match = re.search(r'["«]([^"»]{2,100})["»]', generated_text)
    if match:
        return match.group(1).strip()
    # Если нет кавычек, берем первую строку
    return generated_text.strip().split('\n')[0][:100].strip()

def generate_recommendation(mood_word: str, lang: str) -> str:
    """Generate movie recommendation based on mood and language."""
    if lang == 'ru':
        prompt = (
            f"Посоветуй 1 популярный фильм для человека, у которого {mood_word} настроение. "
            "Напиши только название фильма на русском языке, ничего больше."
        )
    else:
        prompt = (
            f"Recommend 1 popular movie for a person who is in a {mood_word} mood. "
            "Output only the movie title in English, nothing else."
        )

    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_length=55,
        do_sample=True,
        top_p=0.9,
        temperature=0.8,
    )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text[len(prompt):].strip()

@app.route("/", methods=["GET", "POST"])
def index():
    lang = request.args.get('lang')
    if lang in ('en', 'ru'):
        session['lang'] = lang
    elif 'lang' not in session:
        session['lang'] = 'en'

    current_lang = session['lang']
    t = texts[current_lang]

    recommendation = ""
    user_text = ""

    if request.method == "POST":
        user_text = request.form["message"]
        result = sentiment_analyzer(user_text)[0]
        label = result["label"]  # POSITIVE / NEGATIVE / NEUTRAL

        mood_word = t['moods'].get(label, 'neutral')  # на нужном языке

        # Генерируем название фильма на выбранном языке
        ai_text = extract_film_title(generate_recommendation(mood_word, current_lang))
        recommendation = f"{t['result_title']} {ai_text}."

    return render_template("index.html",
                           texts=t,
                           recommendation=recommendation,
                           user_text=user_text,
                           lang=current_lang)

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
