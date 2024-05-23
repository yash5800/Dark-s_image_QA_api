import requests
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering
from flask import Flask,jsonify,request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/send', methods=['POST', 'GET'])
def submit():
      img_url = request.args.get('url')
      question = request.args.get('q')  
      if not img_url:
          return jsonify({"error": "URL parameter is missing"})
      if not question:
          return jsonify({"error": "Question parameter is missing"})

      processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
      model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
      
      raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

      inputs = processor(raw_image, question, return_tensors="pt")
      
      out = model.generate(**inputs)
      print(processor.decode(out[0], skip_special_tokens=True))
      re = processor.decode(out[0], skip_special_tokens=True)
      return jsonify({"Answer": str(re)})
      
if __name__ == "__main__":
    app.run(debug=True)
