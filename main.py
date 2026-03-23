from flask import Flask, render_template, request
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import re

sentiment_analyzer = pipeline("sentiment-analysis", model="blanchefort/rubert-base-cased-sentiment")

app = Flask(__name__)

tokenizer = AutoTokenizer.from_pretrained("sberbank-ai/rugpt3medium_based_on_gpt2")
model = AutoModelForCausalLM.from_pretrained("sberbank-ai/rugpt3medium_based_on_gpt2")

def extract_film_title(generated_text: str) -> str:
    if not generated_text:
        return ""
    m = re.search(r'«([^»]{2,100})»', generated_text)
    if m:
        return m.group(1).strip()
    return generated_text.strip().splitlines()[0][:80].strip()


def generate_recomendation(mood):
    prompt = (f"Посоветуй 1 популярный фильм для человека, у которого {mood} настроение. Напиши только название фильма на английском ничего больше.")
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_length=50,
        do_sample=True,
        top_p=0.9,
        temperature=0.9,
    )
    text= tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text[len(prompt):].strip()

@app.route("/", methods=["GET","POST"])
def index():
    recommendation = ""
    user_text = ""

    if request.method == "POST":
        user_text = request.form["message"]

        result = sentiment_analyzer(user_text)[0]
        label = result["label"]

        if label == "POSITIVE":
            mood = "хорошее"
        elif label == "NEGATIVE":
            mood = "плохое"
        else:
            mood = "нейтральное"
        ai_text = extract_film_title(generate_recomendation(mood))
        recommendation = f"Mood: {mood}. Recommendation: {ai_text}."

    return render_template("index.html", recommendation=recommendation, user_text=user_text)

if __name__ == "__main__":
    app.run(debug=True)
