import csv
import json
from itertools import chain
import os
from collections import defaultdict
import random
import tqdm
from tqdm import tqdm
from queue import Queue

def load_txt(input_dir, merge_list=False):
    ans = [line for line in csv.reader(open(input_dir, 'r', encoding='UTF-8'), delimiter='\t') if len(line)]
    if merge_list:
        ans = list(chain.from_iterable(ans))
    return ans

def write_txt(info_list, out_dir):
    with open(out_dir, 'w', encoding='UTF-8') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(info_list)

def load_json(input_dir, serial_key=False):
    ret_dict = json.load(open(input_dir,'r', encoding='utf-8'))

    if serial_key:
        ret_dict = {tuple([int(i) for i in k.split('_')]):[tuple(l) for l in v] for k,v in ret_dict.items()}
    return ret_dict

def write_json(info_dict, out_dir, serial_key=False):
    if serial_key:
        info_dict = {'_'.join([str(i) for i in k]):v for k,v in info_dict.items()}
    with open(out_dir, "w") as f:
        json.dump(info_dict, f)

def parse_triple(triples, head_mode=True):
    if head_mode:
        if len(triples)==1 :
            h, r, t = triples[0].split()
        else:
            h, r, t = triples
    else:
        t, r, h = triples
    return h, r, t

def load_dict(root_dir,filename):
    with open(os.path.join(root_dir, filename), encoding="utf-8") as fin:
        entity2id = dict()
        for line in fin:
            eid, entity = line.strip().split('\t')
            entity2id[entity] = int(eid)
        return entity2id

def load_examples(path):
    file = open(path,"r",encoding="utf-8")
    example={}
    data=file.readlines()
    for line in data:
        if line in example.keys():
            example[line] +=1
        else:
            example[line] =1

    example = sorted(example.items(), key=lambda item: item[1],reverse=True)
    example=dict(example)

    example_final=[]
    for line in example.keys():
        tmp= {}
        rule_head,rule_body=line.strip().split("<---")
        tmp["rule_head"] = rule_head
        tmp["rule_body"] = rule_body
        example_final.append(tmp)

    return example_final,len(example_final)

def load_type(path):
    with open(path,"r", encoding="utf-8") as file:
        data=file.readlines()
        ent2id={};ent2type=defaultdict(list);id=0
        for line in data:
            ent,type=tuple(line.strip().split("\t"))
            if ent not in ent2type.keys():
                ent2type[ent] = [type]
            else:
                ent2type[ent].append(type)
            if ent not in ent2id.keys():
                ent2id[ent]=id
                id+=1
    return ent2id,ent2type

def get_type_dict(ent2type,type2idx,ent2idx):
    type_dict=defaultdict(list)
    for key, value in ent2type.items():
        for i in value:
            if type2idx[i] not in type_dict.keys():
                type_dict[type2idx[i]] = [ent2idx[key]]
            else:
                type_dict[type2idx[i]].append(ent2idx[key])
    return type_dict


class node:
    def __init__(self, ent, rel=None, pre=None, next=None, level=1):
        self.ent=ent; self.rel=rel
        self.pre=pre; self.next=next
        self.level=level

    def retrace(self):
        path = [(self.rel, self.ent)]; t = self.pre
        while t != None:
            path.append((t.rel, t.ent)); t = t.pre
        return path[::-1]

def gen_filtered_query(graph, query, path_length):
    filter_query = []
    for e1, e2 in tqdm(query):
        if BFS_bool(graph, e1, e2, path_length):
            filter_query.append([e1,e2])
    return filter_query

def BFS_bool(graph, e1, e2, path_length, attempt_thres=5e4):
    attempt = 0; q = Queue(); q.put(node(e1, level=1))
    while not q.empty() and attempt <= attempt_thres:
        cur = q.get(); attempt += 1
        if cur.ent == e2: return True
        if cur.ent in graph and cur.level <= path_length:
            if len(graph[cur.ent])>0:
                for i in graph[cur.ent]:
                    q.put(node(i[1], pre=cur, level=cur.level+1, rel=i[0]))
    return False

def negative_sampling(graph, e1, entity_type, type_e2s, max_len=7, lowest_level=1, max_size=3):
    negative_pairs = []
    q = Queue(); q.put(node(e1, level=1)); l = 1
    while not q.empty() and len(negative_pairs) <= max_size:
        cur = q.get()
        if len([i for i in entity_type[cur.ent] if i in type_e2s])!=0 and cur.level>=lowest_level:
            negative_pairs.append((e1, cur.ent))
        if cur.ent in graph and cur.level <= max_len:
            if len(graph[cur.ent]) > 0:
                l+=1
                for i in graph[cur.ent]:
                    q.put(node(i[1], pre=cur, level=l, rel=i[0]))
    negative_pairs = list(set(negative_pairs))
    if len(negative_pairs) >= max_size:
        random.shuffle(negative_pairs); negative_pairs = negative_pairs[:max_size]
    return negative_pairs


def get_model_path(model_name, model_size=None):
    prefix = f"/home/{os.getenv('USER')}/.cache/huggingface/hub/models--"
    if "llama" in model_name.lower():
        model_ver = model_name.split('-')[-1]
        if model_ver == '2':
            model_size = "7b" if model_size not in ["7b", "13b", "70b"] else model_size
            model_id = "meta-llama/Llama-2-{}-chat-hf".format(model_size)
        if model_ver == '3':
            model_size = "8b" if model_size not in ["8b", "70b"] else model_size
            model_id = "meta-llama/Llama-3-{}-hf".format(model_size)
    model_dir = os.path.join(prefix+model_id.replace('/', '--'), 'snapshots')
    if os.listdir(model_dir):
        return os.path.join(model_dir, os.listdir(model_dir)[0])
    else:
        raise Exception("No model found in {}".format(model_dir))

if __name__ == "__main__":
    data_dir="dataset/nell/"
    with open(data_dir+"NELL_ent2type_DONE.json", "r") as f:
        lines = json.load(f)

    with open(os.path.join(data_dir+"entities.txt"), "w") as f:
        for key,values in lines.items():
            for value in values:
                f.write(key+"\t"+str(value)+"\n")