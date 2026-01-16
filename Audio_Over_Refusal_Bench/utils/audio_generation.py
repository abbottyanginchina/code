import websocket
import datetime
import hashlib
import base64
import hmac
import json
from urllib.parse import urlencode
import time
import ssl
from wsgiref.handlers import format_date_time
from datetime import datetime
from time import mktime
import _thread as thread
import os


STATUS_FIRST_FRAME = 0  # 第一帧的标识
STATUS_CONTINUE_FRAME = 1  # 中间帧标识
STATUS_LAST_FRAME = 2  # 最后一帧的标识


class Ws_Param(object):
    def __init__(self, APPID, APIKey, APISecret, Text):
        self.APPID = APPID
        self.APIKey = APIKey
        self.APISecret = APISecret
        self.Text = Text

        self.CommonArgs = {"app_id": self.APPID}
        self.BusinessArgs = {
            "aue": "lame",                 # mp3
            "sfl": 1,                      # 必传
            "auf": "audio/L16;rate=16000",
            "vcn": "x4_yezi",
            "tte": "utf8"
        }

        self.Data = {
            "status": 2,
            "text": str(base64.b64encode(self.Text.encode("utf-8")), "UTF8")
        }

    def create_url(self):
        url = 'wss://tts-api.xfyun.cn/v2/tts'
        now = datetime.now()
        date = format_date_time(mktime(now.timetuple()))

        signature_origin = (
            "host: ws-api.xfyun.cn\n"
            f"date: {date}\n"
            "GET /v2/tts HTTP/1.1"
        )

        signature_sha = hmac.new(
            self.APISecret.encode("utf-8"),
            signature_origin.encode("utf-8"),
            hashlib.sha256
        ).digest()

        signature_sha = base64.b64encode(signature_sha).decode("utf-8")

        authorization_origin = (
            f'api_key="{self.APIKey}", algorithm="hmac-sha256", '
            f'headers="host date request-line", signature="{signature_sha}"'
        )

        authorization = base64.b64encode(
            authorization_origin.encode("utf-8")
        ).decode("utf-8")

        params = {
            "authorization": authorization,
            "date": date,
            "host": "ws-api.xfyun.cn"
        }

        return url + "?" + urlencode(params)

def generation_audio(text, save_path):
    """
    text: 要合成的文本
    save_path: mp3 保存路径
    """

    wsParam = Ws_Param(
        APPID="ee24b9ca",
        APIKey="96ecd8e548bd81373d83f3c3b00876c1",
        APISecret="N2Y4NGRmMDAxZDhhMDZkZTlmOGFkN2Rk",
        Text=text
    )

    def on_message(ws, message):
        try:
            message = json.loads(message)
            code = message["code"]
            status = message["data"]["status"]

            if code != 0:
                print("TTS error:", message["message"])
                ws.close()
                return

            audio = base64.b64decode(message["data"]["audio"])
            with open(save_path, "ab") as f:
                f.write(audio)

            if status == 2:
                print("TTS finished, closing ws")
                ws.close()

        except Exception as e:
            print("on_message exception:", e)

    def on_error(ws, error):
        print("### ws error:", error)

    def on_close(ws, *args):
        # print("### ws closed ###")
        pass

    def on_open(ws):
        def run():
            payload = {
                "common": wsParam.CommonArgs,
                "business": wsParam.BusinessArgs,
                "data": wsParam.Data
            }
            ws.send(json.dumps(payload))
            
            if os.path.exists(save_path):
                os.remove(save_path)

        thread.start_new_thread(run, ())

    ws = websocket.WebSocketApp(
        wsParam.create_url(),
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )

    ws.on_open = on_open
    ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})

if __name__ == "__main__":
    text = "Hello, I am Xiaomi, nice to meet you."
    generation_audio(text, "/Users/abbottyang/Downloads/demo.mp3")
