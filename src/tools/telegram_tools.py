from datetime import datetime

import requests

import tools



logger = tools.get_logger(__name__, do_file_logging = True)

_base_url = "https://api.telegram.org/bot{}:{}/{}"


def telegram_escape(message):
    must_escape = ['_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!']
    for char in must_escape:
        message = message.replace(char, '\\' + char)
    return message


def telegram_sendMessage(user, token, chat_id, message, message_title):
    url = _base_url.format(user, token, "sendMessage")

    composed_message = "*{title}*\n\n{message_body}\n\n_{timestamp}_".format(
            title = telegram_escape(message_title),
            message_body = telegram_escape(message),
            timestamp = telegram_escape(str(datetime.now()))
            )

    data = { 'chat_id'   : chat_id,
             'text'      : composed_message,
             'parse_mode': "MarkdownV2",
             }

    try:
        r = requests.post(url = url, json = data)
    except Exception as ex:
        logger.exception(ex)


def telegram_sendPhoto(user, token, chat_id, photo, caption):
    url = _base_url.format(user, token, "sendPhoto")

    data = {
            "chat_id": chat_id,
            "caption": telegram_escape(caption)
            }

    try:
        r = requests.post(url = url, data = data, files = { "photo": photo })
    except Exception as ex:
        logger.exception(ex)
