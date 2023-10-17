from flask import Flask
from flask_cors import CORS


if __name__ == "__main__":
    app = Flask(__name__)
    CORS(app)

    with app.app_context():
        from endpoints import blueprint
        app.register_blueprint(blueprint)

    app.run(host='0.0.0.0', port=5000)