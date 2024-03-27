import torch
from scipy.spatial.distance import cosine
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
import re

class simcse:
    def __init__(self):
        self.model_path = "model/sup-simcse-bert-base-uncased"
        # 从本地加载simcse模型
        self.tokenizer_simcse = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModel.from_pretrained(self.model_path)

    def rank(self, list, target, k, t):
        """
        sort list by their distance to target
        :param list:[string1,string2...]
        :param target:string
        :param k:top-k threshold
        :param t:similarity threshold
        :return:[[score, string], [score', string']...] that is sorted
        """
        list.append(target)
        rank = []
        inputs_simcse = self.tokenizer_simcse(list, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            embeddings = self.model(**inputs_simcse, output_hidden_states=True, return_dict=True).pooler_output
        for i in range(len(list) - 1):
            score = 1 - cosine(embeddings[i], embeddings[len(list) - 1])  # elist中第i个和entity比较
            if score >= t:
                rank.append([score, list[i]])
        rank.sort(reverse=True)
        return rank[:k]

class bge:
    def __init__(self):
        self.model_path = "model/bge-large-zh-v1.5"
        self.model = SentenceTransformer(self.model_path)
        
    def count(self, list1, list2):
        embedding1 = self.model.encode(list1, normalize_embeddings=True)
        embedding2 = self.model.encode(list2, normalize_embeddings=True)
        score = embedding1 @ embedding2.T
        return score
        
    def rank(self, list, target, k, t):
        """
        compare every element of list to target, the element is supposed to be string or list of strings, which finaly transform to a merged string
        """
        rank = []
        target_embedding = self.model.encode([target], normalize_embeddings=True)
        for i in list:
            pattern = ""
            for ii in i:
                pattern += ii
            embedding = self.model.encode([re.sub(r'\(.*?\)|\[.*?]|\{.*?}|（.*?）', '', pattern)], normalize_embeddings=True)
            score = target_embedding @ embedding.T
            if score >= t:
                rank.append([score, i])
        rank.sort(reverse=True)
        return rank[:k]
    
        
