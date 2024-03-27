from transformers import AutoConfig, AutoModel, AutoTokenizer
model_name = "chatglm-6b"

# 载入Tokenizer，本地的话，修改为本地地址
tokenizer = AutoTokenizer.from_pretrained("../model/{0}".format(model_name), trust_remote_code=True)
CHECKPOINT_PATH = "output/adgen-{0}-pt-128-2e-2/checkpoint-3000".format(model_name)

import torch
import os

config = AutoConfig.from_pretrained("../model/{0}".format(model_name), trust_remote_code=True, pre_seq_len=128)
model = AutoModel.from_pretrained("../model/{0}".format(model_name), config=config, trust_remote_code=True)
prefix_state_dict = torch.load(os.path.join(CHECKPOINT_PATH, "pytorch_model.bin"))
new_prefix_state_dict = {}
for k, v in prefix_state_dict.items():
    if k.startswith("transformer.prefix_encoder."):
        new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)

# Comment out the following line if you don't use quantization

model = model.quantize(4)
model = model.half().cuda()
model.transformer.prefix_encoder.float().cuda()
model = model.eval()


history = []
while True:
    print("输入对话：")
    content = input()
    #response, history = model.chat(tokenizer, content, history=history)
    response, history = model.chat(tokenizer, content, history=[])
    print("回复：\n", response)
