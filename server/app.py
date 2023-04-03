
from flask import Flask, render_template
from flask_socketio import SocketIO

app = Flask(__name__)
socketio = SocketIO(app,cors_allowed_origins="*")

@app.route("/")
def index():
    return render_template("index.html")

@socketio.on("update")
def update(data):
    print('Current Value', data['value'])

if __name__ == "__main__":
    socketio.run(app)