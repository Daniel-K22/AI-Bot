import openai
import asyncio
import os
from aiogram import Bot, types, Dispatcher
from aiogram.types import InputFile
from dotenv import load_dotenv
import aiohttp
import ffmpeg

# Downloading API Keys
load_dotenv()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Setting up the bot
bot = Bot(token=TELEGRAM_BOT_TOKEN)
dp = Dispatcher(bot)
openai.api_key = OPENAI_API_KEY


# Text message processing function (GPT-4)
async def chatgpt_response(message: types.Message):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": message.text}],
        temperature=0.7
    )
    await message.reply(response["choices"][0]["message"]["content"])


# Image generation function (DALL·E 3)
async def generate_image(message: types.Message):
    response = openai.Image.create(
        model="dall-e-3",
        prompt=message.text,
        n=1,
        size="1024x1024"
    )
    image_url = response["data"][0]["url"]
    await message.reply_photo(photo=image_url, caption="Here is your image!")


# Voice message processing function (Whisper API)
async def transcribe_voice(message: types.Message):
    voice = await message.voice.get_file()
    file_path = f"voice_{message.from_user.id}.ogg"

    # Download voice messages
    await bot.download_file(voice.file_path, file_path)

    # Convertion to MP3 (Whisper does not accept OGG)
    mp3_path = file_path.replace(".ogg", ".mp3")
    ffmpeg.input(file_path).output(mp3_path).run(overwrite_output=True)

    # We send to Whisper API
    with open(mp3_path, "rb") as audio_file:
        transcript = openai.Audio.transcribe(
            model="whisper-1",
            file=audio_file,
            language="en"  # Auto detect launguage, but default English
        )

    os.remove(file_path)
    os.remove(mp3_path)

    await message.reply(f"🎤 You said: {transcript['text']}")


# Command handler
@dp.message_handler(commands=['start'])
async def send_welcome(message: types.Message):
    await message.reply(
        "Hello! Send me a text message and I'll reply using GPT-4.\n"
        "Or describe an image, and I'll generate it using DALL·E 3.\n"
        "You can also send a voice message, and I'll transcribe it using Whisper API.")


# Обработчик голосовых сообщений
@dp.message_handler(content_types=types.ContentType.VOICE)
async def process_voice_message(message: types.Message):
    await transcribe_voice(message)


# Voice message handler
@dp.message_handler()
async def process_text_message(message: types.Message):
    if message.text.lower().startswith("generate"):
        await generate_image(message)
    else:
        await chatgpt_response(message)

# Launch bot
async def main():
    await dp.start_polling()


if __name__ == "__main__":
    asyncio.run(main())
