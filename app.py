import os
import requests
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/send', methods=['POST', 'GET'])
def submit():
    try:
        # Retrieve the 'url' and 'q' query parameters
        img_url = request.args.get('url')
        question = request.args.get('q')

        if not img_url:
            return jsonify({"error": "URL parameter is missing"}), 400
        if not question:
            return jsonify({"error": "Question parameter is missing"}), 400

        # Load the pre-trained processor and model
        processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
        model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")

        # Fetch and process the image
        raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

        # Optional: Resize the image to reduce memory usage
        raw_image.thumbnail((512, 512))

        inputs = processor(raw_image, question, return_tensors="pt")

        # Generate and decode the answer
        out = model.generate(**inputs)
        answer = processor.decode(out[0], skip_special_tokens=True)

        return jsonify({"Answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
