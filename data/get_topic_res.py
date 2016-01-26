#encoding=utf-8
import sys
id2txt_file = "txt2id.table.processed" #sys.argv[1]
#raw_phi_file = "demo_local_phi.iter100" #sys.argv[2]
raw_phi_file = "demo_gl_phi.iter100" #sys.argv[2]
res_file = "global_res.dat" #sys.argv[3]

fd1 = open(id2txt_file, 'r')    #dict
fd2 = open(raw_phi_file, 'r')    #raw res
fd3 = open(res_file, 'wb')   #res

id2txt_dict = dict()
for line in fd1:
    tokens = line.strip().split(' ')
    if len(tokens) < 3:
        continue
    id2txt_dict[int(tokens[2])] = tokens[0]


for line in fd2:
    tokens = line.strip().split(' ')
    topic = tokens[0]
    kvs = dict()
    for tk in tokens[1:]:
        its = tk.split(':')
        kvs[int(its[0])] = float(its[1])
    sorted_kvs = sorted(kvs.items(), key=lambda x:x[1], reverse=True)
    res = ["%s:%s" % (id2txt_dict.get(id, "unknown"), value) for (id, value) in sorted_kvs[:20]]
    fd3.write("%s\t%s\n" % (topic, ' '.join(res)))

fd1.close()
fd2.close()
fd3.close()
        


