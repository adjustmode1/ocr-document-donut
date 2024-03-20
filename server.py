import re
import torch
import cv2
import numpy as np
import io
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel

from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

model_path = "naver-clova-ix/donut-base-finetuned-cord-v2"

print('loading processor')
processor = DonutProcessor.from_pretrained(model_path)
print('loading model')
model = VisionEncoderDecoderModel.from_pretrained(model_path)

decoder_input_ids = processor.tokenizer("<s_cord-v2>", add_special_tokens=False, return_tensors="pt").input_ids

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


def readImage(imagePath):
  image = cv2.imread(imagePath)

  pixel_values = processor(
      image, return_tensors="pt"
  ).pixel_values

  return pixel_values

@app.route("/",methods=['GET'])
def main():
  return render_template('index.html')

@app.route("/predict", methods=['POST'])
def renderIndex():
    print(request.files)
    if 'file' not in request.files:
        return 'No file part'

    file = request.files['file']
    if file.filename == '':
        return 'No selected file '

    # Read image from file object using OpenCV
    image_stream = io.BytesIO(file.read())
    image_stream.seek(0)
    image = cv2.imdecode(np.frombuffer(image_stream.read(), np.uint8), cv2.IMREAD_COLOR)

    pixel_values = processor(
      image, return_tensors="pt"
    ).pixel_values

    outputs = model.generate(
        pixel_values.to(device),
        decoder_input_ids=decoder_input_ids.to(device),
        max_length=model.decoder.config.max_position_embeddings,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        use_cache=True,
        bad_words_ids=[[processor.tokenizer.unk_token_id]],
        return_dict_in_generate=True,
    )

    sequence = processor.batch_decode(outputs.sequences)[0]
    sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
    sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()  # remove first task start token

    return jsonify(processor.token2json(sequence))

if __name__ == '__main__':
    app.run(debug=False)