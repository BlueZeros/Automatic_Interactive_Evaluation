import dashscope
from .base_model import API_Model

class QianWen_Model(API_Model):
    def __init__(self, 
                 api_key="sk-51e0740ed82b4fe79983d1d5e18d0f77",
                 version="qwen_max",
                 stop_ids=[]):
        super().__init__(api_key, stop_ids)
        dashscope.api_key = api_key
        if version == "qwen_turbo":
            self.version = dashscope.Generation.Models.qwen_turbo
        elif version == "qwen_max":
            self.version = dashscope.Generation.Models.qwen_max
        elif version == "qwen_max_longcontext":
            self.version = dashscope.Generation.Models.qwen_max_longcontext
        else:
            print(f"version=={version} is not available in qianwen model!")
            raise NotImplementedError
    
    def generate(self, inputs):
        message = [{"role": "user", "content": inputs}]

        while True:
            try:
                response = dashscope.Generation.call(
                    self.version,
                    messages=message,
                    # set the random seed, optional, default to 1234 if not set
                    seed=0,
                    stop=self.stop_ids,
                    result_format='message',  # set the result to be "message" format.
                    temperature=0.01,
                )

                outputs = response["output"]["choices"][0]["message"]["content"]
                break
            except:
                print("Error, Retrying...")

        return outputs