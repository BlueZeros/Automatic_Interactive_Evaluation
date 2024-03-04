import pdb
import json
import requests
from .base_model import API_Model

class YiYan_Model(API_Model):
    def __init__(self, 
                 api_key="m8ULmlwjjg3bAn8LzG7HqOyC",
                 secret_key="DmHc4PC9En478FGoLRCNVPVMXdpDdIVT",
                 version="ERNIE-Bot-4.0",
                 stop_ids=[]):
        super().__init__(api_key, stop_ids)
        self.api_key = api_key
        self.secret_key = secret_key
        if version == "ERNIE-Bot":
            self.url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions?access_token="
        elif version == "ERNIE-Bot-turbo":
            self.url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/eb-instant?access_token="
        elif version == "ERNIE-Bot-4.0":
            self.url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions_pro?access_token="
        else:
            f"version=={version} is not available in YiYan Model!"
            raise NotImplementedError
    
    def get_access_token(self):
        """
        使用 AK，SK 生成鉴权签名（Access Token）
        :return: access_token，或是None(如果错误)
        """
        url = "https://aip.baidubce.com/oauth/2.0/token"
        params = {"grant_type": "client_credentials", "client_id": self.api_key, "client_secret": self.secret_key}
        return str(requests.post(url, params=params).json().get("access_token"))
    
    def generate(self, inputs):
        while True:
            try:
                url = self.url + self.get_access_token()
                payload = json.dumps({
                    "messages": [
                        {
                            "role": "user",
                            "content": inputs
                        },
                    ],
                    "temperature": 0.01,
                    "stop": self.stop_ids

                })
                headers = {
                    'Content-Type': 'application/json'
                }
                response = requests.request("POST", url, headers=headers, data=payload)
                outputs = json.loads(response.text)["result"]
                break
            except:
                print("Yiyan Retrying...")

        return outputs
