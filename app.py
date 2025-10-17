from flask import Flask, request

app = Flask(__name__)

@app.route("/callback", methods=["GET", "POST"])
def callback():
    print("📩 Incoming webhook or OAuth callback:")
    print("Headers:", dict(request.headers))
    print("Body:", request.get_data(as_text=True))
    print("Args:", request.args)
    return "Callback received successfully!", 200

if __name__ == "__main__":
    app.run(port=5010)
