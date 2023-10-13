import csv
import sys, os


import interlab
import time
import requests
import discord
from discord.ext import tasks
import base64
import random
import operator
import math
import threading
from getpass import getpass
import openai



os.environ["OPENAI_API_KEY"] = input("your key ")
openai.api_key = os.getenv('OPENAI_API_KEY')

if not openai.api_key:
    raise ValueError("API Key not found in environment variables.")


def ask_gpt1(prompt, gpt="gpt-3.5-turbo", temperature=0.5):
  messages = [{"role":"user", "content": prompt}]
  assert gpt in ["gpt-3.5-turbo", "gpt-4"], "Invalid GPT model"
  response = openai.ChatCompletion.create(
      model=gpt,
      messages=messages,
      temperature=temperature,
      max_tokens=500
  )
  return response.choices[0].message.content

def ask_gpt(prompt):
    """Query GPT with a prompt and get the response."""
    response = openai.Completion.create(
      model="text-davinci-003",   # Switch back to the GPT-3 model identifier
      prompt=prompt,
      max_tokens=700
    )
    return response.choices[0].text.strip()


def get_description(song_name):
    prompt = f"""
    give the best description to Midjourney bot so that it draws an image for {song_name} music. I want to capture emotions. Do not type anything else except the description. Use at least 5 sentences. Do not use the words related to music or video. Do not use singers name. Do not use the word midjourney and bot
    """
    return ask_gpt(prompt)

def retell_description(song_name):
    prompt = f"""

retell {song_name}.  Do not use the words related to music or video, only give visual description for the picture. Do not use singers name.  Give the description of emotions as vividly as possible. Do not use the word midjourney and bot
    """
    return ask_gpt(prompt)

def first_hand_description(song_name):
    prompt = f"""

Describe the feelings from the hero of the {song_name} music from their perspective. Give the first-person description What do they feel? What do they see? What do they hear? Do not use the words related to music or video, only give visual description for the picture. Do not use singers name. 
    """
    return ask_gpt(prompt)

#e4 = langchain.chat_models.ChatOpenAI(model_name='gpt-4')



print("Starting...")
# API variables
engine_id = "stable-diffusion-xl-1024-v1-0"
api_host = os.getenv('API_HOST', 'https://api.stability.ai')
api_key = #stablediffkey

# Bot token
token = #your token

# Initialize list to store state inputs
state_inputs = []


def generate_image(prompt_inputs):
  """Generates an image using the Stability API.

    Args:
    prompt_inputs (list): A list of inputs where each input is a list 
    containing a string (state) and a timestamp.

    Returns:
    str: The filename of the generated image.
    """
  p = []

  prompt_options = []
  if len(prompt_inputs) < 10:
    p = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  else:
    while len(p) < 10:
      n = random.randint(0, len(prompt_inputs))
      if not n in p:
        p.append(n)
  print("pset:", p)
  for i in range(len(prompt_inputs)):
    if i in p:
      prompt_options += [{
        "text":
        prompt_inputs[i][0],
        "weight":
        round((max(
          0, 3000 * random.random() +
          (10000 - (time.time() - prompt_inputs[i][1])))) / 1000, 1)
      }]
  # Check if API key is present
  if api_key is None:
    raise Exception("Missing Stability API key.")

  # Construct the payload for the request
  payload = {
    "text_prompts": prompt_options[0:min(7, len(prompt_options))], # [{"text":"monochrome","weight":-1}] + 
   # "samples": 4,
    #"steps": 15,
   # "cfg_scale": 25,
  }
  print(payload)
  # Make the request to the Stability API
  response = requests.post(
    f"{api_host}/v1/generation/{engine_id}/text-to-image",
    headers={
      "Content-Type": "application/json",
      "Accept": "application/json",
      "Authorization": f"Bearer {api_key}"
    },
    json=payload,
  )

  # If the request was unsuccessful, raise an exception
  if response.status_code != 200:
    raise Exception("Non-200 response: " + str(response.text))

  # If the request was successful, save the image locally
  basic_data = response.json()
  filename = "StableImage" + str(time.time())[0:6] + str(
    random.randint(1000, 9999))
  with open(filename + ".png", "wb") as file:
    file.write(base64.b64decode(basic_data["artifacts"][0]["base64"]))
  return filename


# Initialize the Discord client
intents = discord.Intents.default()
intents.message_content = True
discord_client = discord.Client(intents=intents)

@discord_client.event
async def on_ready():
  print(f'We have logged in as {discord_client.user}')


@discord_client.event

async def on_message(message):
  if message.content.startswith('/create'):
    song_name = message.content[8:]
    await message.channel.send('Creating the best description for your song')
    new_user = True
    
  #  description = get_description(song_name)
  #  await message.channel.send(description)
  #  state_input = [description, time.time(), message.author]
  #  state_inputs = [state_input]  # Make a list of lists
  #  image_filename1 = generate_image(state_inputs)
    
  #  description1 = retell_description(song_name)
  #  await message.channel.send(description1)
 #   state_input1 = [description1, time.time(), message.author]
  #  state_inputs1 = [state_input1]  # Make a list of lists
  #  image_filename2 = generate_image(state_inputs1)

    description_joined = first_hand_description(song_name)
    await message.channel.send(description_joined)
    state_input_joined = [description_joined, time.time(), message.author]
    state_inputs_joined = [state_input_joined]  # Make a list of lists
    image_filename3 = generate_image(state_inputs_joined)
    
    #image_filename = generate_image(get_description(song))
 #   await message.channel.send(file=discord.File(image_filename1 + ".png"))
  #  await message.channel.send(file=discord.File(image_filename2 + ".png"))
    await message.channel.send(file=discord.File(image_filename3 + ".png"))
discord_client.run(token)