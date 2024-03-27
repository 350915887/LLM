import json
if __name__ == '__main__':
    check = {"right": 0, "wrong": 0, "acc-e": 0, "acc-r": 0}
    json.dump(check, open("checkpoint.json", "w"))
    print("reset checkpoint.json")