import jieba
import re
import similarity
import json
import llm


def entity_search(dict, entity):
    """
    :param dict:知识库
    :param entity:待搜索的实体
    :return:list, 搜索的实体结果
    """
    # 模糊查询实体

    wordpiece = jieba.cut(entity)  # 分词，再重新拼接
    sub_entity = [re.escape(i) for i in wordpiece]  # 给歧义符号加转义符
    entity = ""
    for s in sub_entity:
        if entity == "":
            entity = s
        else:
            entity += ".*" + s

    e_pattern = re.compile(r'.*' + entity + '.*', re.IGNORECASE)  # 不区分大小写
    e_list = []
    for key in dict.keys():
        result = e_pattern.search(key)
        if result:
            e_list.append(result.group())
    return e_list


def relation_search(sub_dict, relation):
    """
    :param sub_dict:对应某个实体的属性字典
    :param relation:待搜索的关系
    :return:list, 搜索的关系结果
    """
    r_pattern = re.compile(r'.*' + re.escape(relation) + '.*')
    r_list = []
    for key in sub_dict.keys():
        result = r_pattern.search(key)
        if result:
            r_list.append(result.group())
    return r_list


if __name__ == '__main__':
    # 总共20559652行
    # dict = {}
    # with open(kg_path, 'r') as file:
    #     for line in file:
    #         line = line.strip('\n').split("\t")
    #         if line[0] in dict:
    #             dict[line[0]][line[1]] = line[2]
    #         else:
    #             dict[line[0]] = {line[1]: line[2]}
    # print(len(dict))

    knowledge_base = json.load(open('KB/Knowledge.json', 'r'))
    sim_model = similarity.bge()

    path = "data/test.json"
    try:
        check = json.load(open("checkpoint.json", "r"))
    except FileNotFoundError:
        check = {"right": 0, "wrong": 0, "acc-e": 0, "acc-r": 0}

    index = -1

    with open(path, 'r') as f:
        for line in f:
            index += 1
            if index < check["right"] + check["wrong"]:
                continue

            dict = json.loads(line)
            # retrive data

            question = dict["question"]
            # print("question:", question)
            # print("answer:", answer)
            response = []
            for i in range(3):  # 大模型三次输出机会
                origin_response = llm.nlq2er(question)
                # print("response:", response)
                response = origin_response.split(" ||| ")
                if len(response) == 3:
                    break
                else:
                    continue
            try:
                origin_entity, relation = response[0], response[1]
            except:
                check["wrong"] += 1
                print("can't analyse to triple")
                print(origin_response)
                print(response)
                print("accuracy:", check["right"], "/", index + 1,
                      check["right"] / (index + 1))
                json.dump(check, open("checkpoint.json", "w"))
                continue

            # entity去括号处理, relation分词处理
            entity = re.sub(r'\(.*?\)|\[.*?]|\{.*?}|（.*?）', '', origin_entity)
            wordpiece = jieba.cut(relation)
            sub_relation = [i for i in wordpiece]

            candidate = []
            # search entity from kg
            e_list = entity_search(knowledge_base, entity)
            e_rank = sim_model.rank(e_list, origin_entity, 2, 0.7)  # 加注释的匹配，可能因为长度相似而得分较高，此时应当十分严格；
            e_rank.extend(sim_model.rank(e_list, entity, 15, 0.5))  # 相应地，不加注释的匹配宽容一些

            # search relation from kg-entity
            """
            for e in e_rank:
                print("e:", knowledge_base[e[1]])
                r_list = []
                for sub in sub_relation:
                    r_list.extend(relation_search(knowledge_base[e[1]], sub))
                r_rank = sim_model.rank(r_list, relation, 5, 0.5)
                print("r_rank:", r_rank)
                for r in r_rank:
                    candidate.append([e[0]*r[0], e[1], r[1]])
                    candidate.sort(reverse=True)
            """

            # choose all relations for entity
            for e in e_rank:
                # print("e:", knowledge_base[e[1]])
                r_list = []
                for key in knowledge_base[e[1]].keys():
                    r_list.append(key)
                r_rank = sim_model.rank(r_list, relation, 3, 0.3)  # 关系稍微严格些，因为实体对应关系的集合较小
                # print("r_rank:", r_rank)
                for r in r_rank:
                    score = e[0] * r[0]  # 定义实体关系的综合相似度
                    candidate.append([score, [e[1], r[1]]])
                    candidate.sort(reverse=True)
                    candidate = candidate[:5]

            # c_rank = candidate
            # 二次重排
            c_list = [c[1] for c in candidate]
            c_rank = sim_model.rank(c_list, question, 1, 0.0)

            if c_rank == []:
                result = None
                output = {"id": index, "answer":""}
            else:
                result = knowledge_base[c_rank[0][1][0]][c_rank[0][1][1]]
                output = {"id": index, "answer": c_rank[0][1][0] + " ||| " + c_rank[0][1][1] + " ||| " + result}
            # print("result:", result)

            try: # 评估模块，如果有标答则评估
                answer = dict["answer"].split(" ||| ")[2]
                squestion = dict["answer"].split(" ||| ")[:1]
                llm_sim = sim_model.count(squestion, [entity, relation])[0]
                check["acc-e"] += llm_sim[0]
                check["acc-r"] += llm_sim[1]
    
                print("llm accuracy:", llm_sim[0], llm_sim[1])
                if result == answer:
                    check["right"] += 1
                else:
                    check["wrong"] += 1
    
                    print("-------------------------------------------------------")
                    # print("wrong")
                    print("question:", question)
                    print("response:", response)
                    print("entity:", entity)
                    print("e_rank:", e_rank)
                    for e in e_rank:
                        print("e:", knowledge_base[e[1]])
                    print(candidate)
                    print(c_rank)
                    print("answer:", dict["answer"])
                    print("result:", result)
                    print("-----------------------------------------------------\n")
            except:
                check["wrong"] += 1
                output = json.dumps(output, ensure_ascii=False)
                f1 = open('output/output.json', 'a+', encoding='utf-8')
                f1.write(output + '\n')

            json.dump(check, open("checkpoint.json", "w"))
            print("accuracy:", check["right"], "/", index + 1,
                  check["right"] / (index + 1))

    print("total llm accuracy:", check["acc-e"] / (index + 1),
          check["acc-r"] / (index + 1))
    print("total accuracy:", check["right"], "/", index + 1,
          check["right"] / (index + 1))



