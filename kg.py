from py2neo import Graph, Node, Relationship, NodeMatcher, Subgraph, RelationshipMatcher
URI = "neo4j+ssc://d6aa953a.databases.neo4j.io"
AUTH = ("neo4j", "41824144")  # 用户名、密码
graph = Graph(uri=URI, auth=AUTH)
def create_KG(dict):
    """
    创建图谱
    :param dict: 字典
    :return:
    """
    graph.delete_all()
    print("deleted old")

    for key in dict:
        a = Node('Node', name=key)
        for attribute in dict[key]:
            a[attribute] = dict[key][attribute]
        graph.create(a)
    print("created new")
    return

def entity_search(entity):
    """
    模糊查询
    :param entity:
    :return:
    """
    # 创建查询对象
    node_matcher = NodeMatcher(graph)
    # node = node_matcher.match("Node").where(name="...Baby One More Time（布兰妮·斯皮尔斯个人单曲）").first()
    node_list = list(node_matcher.match("Node").where("_.name=~'.*"+entity+".*'"))  # 模糊查询
    return node_list

def relation_search(relation):
