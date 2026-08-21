from flask import Flask, render_template, request, send_from_directory, jsonify
import os
import glob

from chatbot import ask_groq
from werkzeug.utils import secure_filename
from predict import predict_image


app = Flask(__name__)


# =========================================================
# CONFIGURATION
# =========================================================

UPLOAD_FOLDER = "uploads"
DATASET_FOLDER = "dataset"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

ALLOWED_EXTENSIONS = {
    "jpg",
    "jpeg",
    "png"
}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def allowed_file(filename):

    return (
        "." in filename
        and
        filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


def get_images(folder, count=8):

    extensions = [
        "*.jpg",
        "*.jpeg",
        "*.png",
        "*.JPG",
        "*.JPEG",
        "*.PNG"
    ]

    images = []

    for extension in extensions:

        images.extend(
            glob.glob(
                os.path.join(folder, extension)
            )
        )

    images = list(dict.fromkeys(images))

    images = sorted(images)

    return images[:count]


# =========================================================
# HOME PAGE
# =========================================================

@app.route("/")
def home():

    fire = get_images(
        os.path.join(
            DATASET_FOLDER,
            "test",
            "fire"
        )
    )

    smoke = get_images(
        os.path.join(
            DATASET_FOLDER,
            "test",
            "Smoke"
        )
    )

    nonfire = get_images(
        os.path.join(
            DATASET_FOLDER,
            "test",
            "nonfire"
        )
    )

    fire = [
        x.replace("\\", "/")
        for x in fire
    ]

    smoke = [
        x.replace("\\", "/")
        for x in smoke
    ]

    nonfire = [
        x.replace("\\", "/")
        for x in nonfire
    ]

    return render_template(
        "index.html",
        fire=fire,
        smoke=smoke,
        nonfire=nonfire
    )


# =========================================================
# SERVE DATASET IMAGES
# =========================================================

@app.route("/dataset/<path:filename>")
def dataset_file(filename):

    return send_from_directory(
        DATASET_FOLDER,
        filename
    )


# =========================================================
# PREDICT GALLERY IMAGE
# =========================================================

@app.route(
    "/predict_gallery",
    methods=["POST"]
)
def predict_gallery():

    image_path = request.form.get(
        "image_path"
    )

    if not image_path:

        return (
            "No image selected",
            400
        )

    if not os.path.exists(image_path):

        return (
            "Selected image not found",
            404
        )

    try:

        prediction, confidence = predict_image(
            image_path
        )

        filename = image_path.replace(
            "\\",
            "/"
        )

        return render_template(
            "result.html",

            prediction=prediction,

            confidence=round(
                confidence * 100,
                2
            ),

            image=filename
        )

    except Exception as e:

        return (
            f"Prediction error: {str(e)}",
            500
        )


# =========================================================
# SERVE UPLOADED IMAGES
# =========================================================

@app.route("/uploads/<filename>")
def uploaded_file(filename):

    return send_from_directory(
        app.config["UPLOAD_FOLDER"],
        filename
    )


# =========================================================
# USER IMAGE PREDICTION
# =========================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    if "image" not in request.files:

        return (
            "No file uploaded",
            400
        )

    file = request.files["image"]

    if file.filename == "":

        return (
            "No file selected",
            400
        )

    if not allowed_file(file.filename):

        return (
            "Only JPG, JPEG and PNG images are allowed",
            400
        )

    filename = secure_filename(
        file.filename
    )

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    try:

        file.save(filepath)

        prediction, confidence = predict_image(
            filepath
        )

        return render_template(
            "result.html",

            prediction=prediction,

            confidence=round(
                confidence * 100,
                2
            ),

            image=filename
        )

    except Exception as e:

        return (
            f"Prediction error: {str(e)}",
            500
        )


# =========================================================
# FILE TOO LARGE ERROR
# =========================================================

@app.errorhandler(413)
def request_entity_too_large(error):

    return (
        "File is too large. Maximum allowed size is 10 MB.",
        413
    )


# =========================================================
# PERFORMANCE PAGE
# =========================================================

@app.route("/performance")
def performance():

    return render_template(
        "performance.html"
    )


# =========================================================
# CHATBOT PAGE
# =========================================================

@app.route(
    "/chat",
    methods=["GET", "POST"]
)
def chat():

    question = ""
    answer = ""

    if request.method == "POST":

        question = request.form.get(
            "question",
            ""
        ).strip()

        if question:

            try:

                answer = ask_groq(
                    question
                )

            except Exception as e:

                answer = (
                    f"❌ Chatbot Error: {str(e)}"
                )

    return render_template(
        "chat.html",
        question=question,
        answer=answer
    )


# =========================================================
# CHATBOT API
# =========================================================

@app.route(
    "/chat_api",
    methods=["POST"]
)
def chat_api():

    try:

        data = request.get_json()

        if not data:

            return jsonify({
                "error": "No data received."
            }), 400

        message = data.get(
            "message",
            ""
        ).strip()

        if not message:

            return jsonify({
                "error": "Please enter a question."
            }), 400

        answer = ask_groq(
            message
        )

        return jsonify({
            "answer": answer
        })


    except Exception as e:

        print(
            "Chatbot API Error:",
            str(e)
        )

        return jsonify({
            "error": str(e)
        }), 500


# =========================================================
# RUN APPLICATION
# =========================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )