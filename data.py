import json
path = "data/dev.json"
with open(path, 'r') as f:
    for line in f:
        dict = json.loads(line)
        q = dict["question"]
        spo = dict["answer"].split(" ||| ")
        print(q)
        print(spo)