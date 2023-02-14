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

s3 = boto3.resource('s3')
BUCKET = "ns.riffusion.test"
FOLDER = "dev_riffusion/"

sqs = boto3.resource("sqs")
queue = sqs.get_queue_by_name(QueueName="your_queue_name")

def predict(prompt, negative_prompt, filename):
   spec = pipe(
       prompt,
       negative_prompt=negative_prompt,
       width=768,
   ).images[0]
   wav = converter.audio_from_spectrogram_image(image=spec)
   wav.export(filename , format='wav')
   saved_filename_with_format = FOLDER + filename
   s3.Bucket(BUCKET).upload_file(filename, saved_filename_with_format)

def process_message(body):
    message = json.loads(body)
    prompt = message["positive_text"]
    negative_prompt = message["negative_text"]
    username = message["username"]
    filename = username + "_" + prompt + ".wav"
    predict(prompt, negative_prompt, filename)

# while True:
#     messages = queue.receive_messages(MaxNumberOfMessages=1)
#     if messages:
#         for message in messages:
#             process_message(message.body)
#             message.delete()
#     else:
#        break
        
prompt = "solo piano piece, classical"
negative_prompt = "drums"
username = "ecs"
predict(prompt, negative_prompt, username)
