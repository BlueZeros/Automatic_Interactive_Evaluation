from .base_model import API_Model
import models.xinghuo_utils as SparkApi
import pdb

class XingHuo_Model(API_Model):
    def __init__(self, 
                 api_key="",
                 secret_key="",
                 appid="",
                 version="v3",
                 stop_ids=[]):
        super().__init__(api_key, stop_ids)

        self.api_key = api_key
        self.secret_key = secret_key
        self.appid = appid

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
    
