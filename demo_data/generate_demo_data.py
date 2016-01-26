import sys, json, random

fd = open(sys.argv[1], 'wb')
for i in range(2):
    data = {}
    data["doc_id"] = random.randint(1, 100)
    data["sentences"] = []
    for j in range(3):
        sentence = {}
        sentence["sent_id"] = random.randint(1, 20)
        sentence["words"] = []
        for k in range(4):
            sentence["words"].append(random.randint(1, 20))
        data["sentences"].append(sentence)
    fd.write("%s\n" % (json.dumps(data)))
fd.close()
