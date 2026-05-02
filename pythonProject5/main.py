import telebot
from config import TG_TOKEN
from ai_helper import ai_classification
bot = telebot.TeleBot(TG_TOKEN)

@bot.message_handlers(commands=['start'])
def start_commands(message):
    text = (
        f'Привет,{message.from_username}!\n'
        'я бот, который распознаёт птиц на фото. Отправь мне фотографию и я определю какая на ней птица'
    )
    bot.send_message(message.chat.id, text)

@bot.message_handlers(content_types=['photo'])
def hand_photo(message):
    file_info = bot.get_file(message.photo[-1].file_id)
    file_name = file_info.file_path.split('/')[-1]
    file_path = file_info.file_path

    downloaded_file = bot.download_file(file_path)
    with open(file_name, 'wb') as photo:
        photo.write(downloaded_file)

    results = ai_classification(file_name)
    if results:
        text = 'вот что я думаю'
        bot.send_message(message.chat.id, text)
    else:
        text = 'увы, у меня не вышло определить кто это'
        return bot.send_message(message.chat.id, text)

    for photo_path, result in results:
      with open(photo_path, 'rb') as photo:
        image_caption = f'с вероятностью {result[1]}% на фото {result[0].lower()}'
        bot.send_photo(message.chat.id, photo=photo, caption=image_caption)

bot.infinity_polling()