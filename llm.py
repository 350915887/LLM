from transformers import AutoTokenizer, AutoModel, AutoConfig
import torch
import os

tokenizer = AutoTokenizer.from_pretrained("model/chatglm-6b", trust_remote_code=True)

CHECKPOINT_PATH = "ptuning/output/adgen-chatglm-6b-pt-128-2e-2/checkpoint-3000"
config = AutoConfig.from_pretrained("model/chatglm-6b", trust_remote_code=True, pre_seq_len=128)

model = AutoModel.from_pretrained("model/chatglm-6b", config=config, trust_remote_code=True)

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


#eval模式
model = model.eval()

def nlq2er(content):
  response, history = model.chat(tokenizer, content, history=[])
  return response
  
if __name__ == "__main__":
  print(nlq2er("泰勒斯威夫特是干什么的'"))