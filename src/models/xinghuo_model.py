from .base_model import API_Model
import models.xinghuo_utils as SparkApi
import pdb

class XingHuo_Model(API_Model):
    def __init__(self, 
                 api_key="7207a892f04e5d1815784b2d5aac568b",
                 secret_key="NmNiMzUzMTYwMjc4MjY1NTYyMzZkNjU5",
                 appid="a9bfb41f",
                #  api_key="200b4e4427de58d0c2971ea6d7c12609",
                #  secret_key="MDU4ZDVkYTEwMDhiMGExMjE0ODIzZWE3",
                #  appid="ceeb0642",
                 version="v3",
                 stop_ids=[]):
        super().__init__(api_key, stop_ids)

        self.api_key = [
            # "62b8f6f042a94c84e178f1a25983875f",
            # "d9b145adc522ff9448509f4a0e6d9877",
            # "6a3aeb87d81e04d0153afdfd98631503",
            # "2f326fda83b38a1de3fe25f62154866f",
            # "baa5fd40a78037e029f5b1488dcb1483",
            "88eaf2be606c1f2b3311364df8cc1799"
        ]

        self.secret_key = [
            # "ZWNhNzk1Mzc2OTIwZDA3ZTllMzc3ZWJl",
            # "NzEzYmYzYWJkMGYyM2E3MTg5MmY3MWZj",
            # "MTllM2IzY2JjMGVlMDFmODE4ZjdlNDBj",
            # "YTRiODhiNDU3ZTQ1OWVlOGZlYmQ0MjE1",
            # "ZTM0OTJiYTEzODg2ZDkzNTZiZGJjMjAy",
            "MTg1YzJjOTAwNjFkNjk5MTc3OGVjOWE2",
        ]
        self.appid = [
            # "a1a55ad1",
            # "1e613418",
            # "e9324b80",
            # "45cd164a",
            # "8e7c39ff",
            "00ff09ff",
        ]

        if version == "v1":
            self.dmain = "general" # v1.5版本
            self.Spark_url = "ws://spark-api.xf-yun.com/v1.1/chat" # v1.5环境的地址
        elif version == "v2":
            self.domain = "generalv2"    # v2.0版本
            self.Spark_url = "ws://spark-api.xf-yun.com/v2.1/chat"  # v2.0环境的地址
        elif version == "v3":
            self.domain = "generalv3"    # v3.0版本
            self.Spark_url = "ws://spark-api.xf-yun.com/v3.1/chat"  # v3.0环境的地址
        else:
            f"version=={version} is not available in XingHuo Model!"
            raise NotImplementedError
    
    def generate(self, inputs):
        SparkApi.answer =""
        output = ""
        inputs = inputs.replace("sex", "").replace("nipple", "chest").replace("anti", "").replace("性","")
        inputs = inputs[-7999:]
        prompt = checklen(getText("user", inputs))
        times = 0
        l = len(self.api_key)
        while output == "":
            if times > 0:
                # pdb.set_trace()
                print("Retrying...")
            # pdb.set_trace()
            SparkApi.main(self.appid[times%l], self.api_key[times%l], self.secret_key[times%l], self.Spark_url, self.domain, prompt)
            times += 1
            output = SparkApi.answer
            for stop_id in self.stop_ids:
                output = output.strip((stop_id+": ")).split(stop_id)[0]

            
        return output

def getText(role,content):
    text =[]
    jsoncon = {}
    jsoncon["role"] = role
    jsoncon["content"] = content
    text.append(jsoncon)
    return text

def getlength(text):
    length = 0
    for content in text:
        temp = content["content"]
        leng = len(temp)
        length += leng
    return length

def checklen(text):
    while (getlength(text) > 8000):
        del text[0]
        # text[0]["content"] = text[0]["content"][-8000:]
    return text
    
