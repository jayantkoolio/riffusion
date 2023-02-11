from diffusers import DiffusionPipeline
from riffusion.spectrogram_image_converter import SpectrogramImageConverter
from riffusion.spectrogram_params import SpectrogramParams
from io import BytesIO
from IPython.display import Audio
import boto3
import json

pipe = DiffusionPipeline.from_pretrained("riffusion/riffusion-model-v1")
pipe = pipe.to("cuda")

params = SpectrogramParams()
converter = SpectrogramImageConverter(params)

#s3 = boto3.client("s3")
BUCKET = "app.koolio.ai"
EXPORT_PATH = "dev_file/"

sqs = boto3.resource("sqs")
queue = sqs.get_queue_by_name(QueueName="your_queue_name")

def predict(prompt, negative_prompt):
   spec = pipe(
       prompt,
       negative_prompt=negative_prompt,
       width=768,
   ).images[0]

   wav = converter.audio_from_spectrogram_image(image=spec)
   wav.export('output2.wav', format='wav')
   filename = "output2.wav"
   saved_filename_with_format = EXPORT_PATH + filename
#   s3.upload_file(filename, BUCKET, saved_filename_with_format)
   return 'output2.wav', spec

def process_message(body):
    message = json.loads(body)
    prompt = message["prompt"]
    negative_prompt = message["negative_prompt"]
    path, spec = predict(prompt, negative_prompt)

while True:
    messages = queue.receive_messages(MaxNumberOfMessages=1)
    if messages:
        for message in messages:
            process_message(message.body)
            message.delete()
    else:
        break
