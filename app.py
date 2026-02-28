from flask import Flask, request, render_template_string
from transformers import pipeline
import sys
import subprocess

# Install Flask if not already installed
try:
    import flask
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "flask"])

app = Flask(__name__)
sentiment = pipeline("sentiment-analysis")

html = """
<!doctype html>
<title>Sentiment Analysis</title>
<h2>Enter text to analyze sentiment:</h2>
<form method=post>
  <textarea name=text rows=4 cols=50></textarea><br><br>
  <input type=submit value="Analyze">
</form>
{% if result %}
  <h3>Result: {{ result }}</h3>
{% endif %}
"""

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    if request.method == "POST":
        text = request.form["text"]
        result = sentiment(text)[0]["label"]
    return render_template_string(html, result=result)

if __name__ == "__main__":
    app.run(debug=True)
