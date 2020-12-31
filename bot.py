import os
import pathlib

import telebot
from telebot.apihelper import ApiException

from do_image_stuff import do_magic, do_me_out

proxy = "http://localhost:8123"

os.environ["http_proxy"] = proxy
os.environ["HTTP_PROXY"] = proxy
os.environ["https_proxy"] = proxy
os.environ["HTTPS_PROXY"] = proxy

BOT = telebot.TeleBot("1140263991:AAEr3RIGdO5TN7QCnZYcNAKK9dr7xCHiJe8")

got_image = dict()



@BOT.message_handler(commands=["start"])
def start(message):
    BOT.send_message(
        message.chat.id,
        "Yo!, send me nudes. with face.. or just send selfies :(, when you finished sending nudes hit /done",
    )
    pathlib.Path(f"bot/{message.chat.id}").mkdir(parents=True, exist_ok=True)


@BOT.message_handler(commands=["reset"])
def reset(message):
    reset_user_dir(message)
    BOT.send_message(message.chat.id, "Ok...")


@BOT.message_handler(commands=["laymeout"])
def lay_me_out(message):

    message = BOT.send_message(
        message.chat.id,
        "Give me a sec!",
    )
    do_me_out(message.chat.id)
    BOT.edit_message_text(
        message_id=message.message_id,
        chat_id=message.chat.id,
        text=f"Uploading...",
    )
    BOT.send_photo(message.chat.id, photo=open(f"gifsbot/{message.chat.id}.jpg", "rb"))


def reset_user_dir(message):
    got_image.pop(message.chat.id, None)
    pathlib.Path(f"archive/{message.chat.id}/").mkdir(parents=True, exist_ok=True)
    os.system(f"mv bot/{message.chat.id}/* archive/{message.chat.id}/")


@BOT.message_handler(commands=["done"])
def done(message):
    got_image.pop(message.chat.id, None)
    BOT.send_message(
        message.chat.id,
        "Alright. imma make you a gif now :)",
    )
    do_magic(
        chat_id=message.chat.id,
        bot=BOT,
    )


@BOT.message_handler(commands=["resend"])
def resend(message):
    files = os.listdir(f"bot/{message.chat.id}/")
    files_count = len(files)
    if files_count <= 0:
        BOT.send_message(message.chat.id, f"I didn't have any images.. fuck off.")
        return

    is_video = files_count == 1
    file_format = "gif"
    if is_video:
        file_format = "mp4"
        BOT.send_message(message.chat.id, f"Alright resending the last video")
    else:
        BOT.send_message(message.chat.id, f"Alright resending the last gif")
    BOT.send_animation(
        message.chat.id, animation=open(f"{message.chat.id}.{file_format}", "rb")
    )


@BOT.message_handler(content_types=["photo"])
def upload(message):
    file_id = message.photo[-1].file_id
    file_info = BOT.get_file(file_id)

    user_files = os.listdir(f"bot/{message.chat.id}")
    for file in user_files:
        if file.split(".")[0].split("_bitchBITCH_")[1] == file_id:
            BOT.send_message(message.chat.id, f"skipping {file_id}")
            return

    downloaded_file = BOT.download_file(file_info.file_path)
    curr = len(os.listdir(f"bot/{message.chat.id}"))
    working_file = f"bot/{message.chat.id}/{curr}_bitchBITCH_{file_id}.jpg"
    with open(working_file, "wb") as new_file:
        new_file.write(downloaded_file)

    print("received", working_file)
    if message.chat.id in got_image.keys():
        BOT.edit_message_text(
            message_id=got_image[message.chat.id],
            chat_id=message.chat.id,
            text=f"got {curr + 1}. if that's not all of them wait for me to catch them all :( then hit /done",
        )
    else:
        message = BOT.send_message(
            message.chat.id,
            f"got {curr + 1}. if that's not all of them wait for me to catch them all :( then hit /done",
        )
        got_image[message.chat.id] = message.message_id


@BOT.message_handler(content_types=["video"])
def upload(message):
    file_id = message.video.file_id
    file_info = BOT.get_file(file_id)

    reset_user_dir(message)

    downloaded_file = BOT.download_file(file_info.file_path)
    working_file = f"bot/{message.chat.id}/vid_bitchBITCH_{file_id}.mp4"
    with open(working_file, "wb") as new_file:
        new_file.write(downloaded_file)

    BOT.send_message(
        message.chat.id,
        f"got a video. i hope it's not longer than one minute cause then you'll be in for a long wait",
    )
    done(message)


while True:
    try:
        BOT.polling()
    except ApiException:
        pass
