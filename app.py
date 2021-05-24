from flask import Flask, render_template, request
from chatbot_dep import chatbot_response

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("home.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return (chatbot_response(userText))

if __name__ == "__main__":
    app.run()
