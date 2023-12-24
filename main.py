import discord
from discord.ext import commands
from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np

def get_class(model_path, labels_path, image_path):
  np.set_printoptions(suppress=True)
  model = load_model("keras_model.h5", compile=False)
  class_names = open(labels_path, "r", encoding="utf-8").readlines
  data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
  image=Image.open(image_path).convert("RGB")

  size = (224, 224)
  image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

  image_array = np.asarray(image)

  normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

  data[0] = normalized_image_array

  prediction = model.predict(data)
  index = np.argmax(prediction)
  class_name = class_names[index]
  confidence_score = prediction[0][index]
  
  return(class_name[2:], confidence_score)

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix='#', intents = discord.Intents.default())

@bot.event
async def on_ready():
    print(f'I {bot.user}')

@bot.command()
async def check(ctx):
    if ctx.message.attachements:
        for attachmnent in ctx.message.attachments:
            file_name = attachmnent.filename
            file_url = attachmnent.url
            await attachmnent.save(f'./{attachmnent.filename}')
            await ctx.send(f'Сохрани картинку в ./{attachmnent.filenamee}')
            await ctx.send(get_class(model_path="./keras_model.h5", labels_path="labels.txt", image_path=f"./{attachmnent.filename}"))
    else:
        await ctx.send('Вы забыли загрузить картинку')
bot.run('MTE4NTg3NTcwMDM3MzI2NjQzMg.GmP-6J.WB0wOckDF5lAMGIXMkkwJSGBB-yFTIo7_fTtj4')