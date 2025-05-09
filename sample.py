import json
import os
from utils import load_txt, parse_triple,load_json,load_type,write_json,write_txt,get_model_path
from collections import defaultdict
import random
from copy import deepcopy
from tqdm import tqdm
from itertools import product
from typetree import typetree
from difflib import get_close_matches
import torch 
from transformers import LlamaTokenizer, LlamaForCausalLM


def load_paths(paths):
    res=[]
    with open(paths,'r') as fin:
        for i in fin.readlines():
            res.append(i.strip().split())
    ent_ser=[]
    rel_ser=[]
    for path in res:
        ent=[]
        rel=[]
        for k,i in enumerate(path):
            if k%2==0:
                ent.append(i)
            else:
                rel.append(i)
        ent_ser.append(ent)
        rel_ser.append(rel)
    return ent_ser,rel_ser


class RelationDict:
    def __init__(self, data_dir, inv=True):
        self.rel2idx, self.idx2rel = {}, {}
        self.idx = 0; self.noninv_idx = 0
        for rel in load_txt(data_dir):
            if len(rel[0]) < 3:
                continue
            self.add_relation(rel[0])
            if "_inv" not in rel[0]:
                self.noninv_idx += 1

    def add_relation(self, rel):
        if rel not in self.rel2idx:
            self.rel2idx[rel] = self.idx
            self.idx2rel[self.idx] = rel
            self.idx += 1

class Dataset(object):
    def __init__(self, args, sparsity=1, inv=True):
        data_dir, self.max_length = args.path_data, args.max_length
        self.generation_rounds = args.generation_rounds
        self.logger = args.logger
        self.data_set=data_dir.split('/')[-2]
        # Entity Dict and Type
        self.ent2idx,self.ent2type = load_type(data_dir+'entities.txt')
        self.idx2ent = {i:j for j, i in self.ent2idx.items()}
        self.type2idx = {i:j for j,i in enumerate(load_json(data_dir+'types.json'))}

        # Relation Dict
        self.rdict = RelationDict(data_dir+"relations.txt", inv=False)
        self.head_rdict = deepcopy(self.rdict)
        self.head_rdict.add_relation("None")

        root=f"dataset/{self.data_set}/processed_data/"
        write_json(self.ent2idx,os.path.join(root, "entities.json"))
        write_json(self.type2idx,os.path.join(root, "types.json"))
        write_json(self.rdict.rel2idx,os.path.join(root, "relations.json"))

        # Fact Triples
        fact_file = "fact.txt"
        fact = load_txt(data_dir+fact_file)
        self.fact_rdf = random.sample(fact, int(len(fact)*sparsity)) #采样的密度
        # Graph Triples
        self.train_rdf = load_txt(data_dir+'train.txt')
        self.valid_rdf = load_txt(data_dir+'valid.txt')
        self.test_rdf = load_txt(data_dir+'test.txt')
        self.all_rdf = self.fact_rdf + self.train_rdf + self.valid_rdf
        if "lp" in args.task:
            test_dir = args.path_root + f"{args.target_heads}-testset"
            if os.path.exists(test_dir):
                test_set_ = load_txt(test_dir)
                self.all_rdf = list(set([tuple(i) for i in self.all_rdf])-set([tuple(item) for item in test_set_]))

        #about triples
        self.rel2rdf = defaultdict(list)
        self.graph_h2rt = defaultdict(list); self.graph_ht2r = defaultdict(list);self.graph_rt2h=defaultdict(list)
        self.graph_r2ht = defaultdict(list);self.graph_r2ht_idx=defaultdict(list);self.graph_r2ht_idx_fact=defaultdict(list)
        
        for rdf in self.all_rdf:
            h, r, t = parse_triple(rdf)
            self.rel2rdf[r].append((h, r, t))
            self.graph_h2rt[h].append((r, t))
            self.graph_ht2r[(h, t)].append(r)
            self.graph_r2ht[r].append([h,t])
            self.graph_rt2h[(r, t)].append(h)
            r_id=self.rdict.rel2idx[r];h_id,t_id=self.ent2idx[h],self.ent2idx[t]
            self.graph_r2ht_idx[r_id].append([h_id,t_id])
        
        for rdf in self.fact_rdf:
            h, r, t = parse_triple(rdf)
            r_id = self.rdict.rel2idx[r];h_id, t_id = self.ent2idx[h], self.ent2idx[t]
            self.graph_r2ht_idx_fact[r_id].append(([h_id,t_id]))

        self.graph_ht2r = dict(self.graph_ht2r)
        self.meta_path = defaultdict(list)
        self.model_name = 'llama-2'
        self.device=args.device

        #only head
        heads=[]
        for value in self.rdict.idx2rel.values():
            if "_inv" not in value and value != "None":
                heads.append(value)
        self.heads=heads
        
        self.relation=self.head_rdict.idx2rel.values()
        self.type=set(tuple([i[0] for i in self.ent2type.values()]))

    def bfs_sample(self,max_length,num,path_example,target_head):
        self.max_length = max_length
        self.target_head = target_head
        instance_paths=[];save_paths=[]
        target_head =[target_head] if type(target_head) == str else target_head
        for head in target_head:
            sampled_rdf=self.rel2rdf[head] if len(self.rel2rdf[head]) < num else random.sample(self.rel2rdf[head], num)
            for rdf in tqdm(sampled_rdf, desc="Sampling instance_paths"):
                e1,_,e2 = parse_triple(rdf)
                attempt = 0
                stack=[(e1,"nan","nan")] 
                meta_seq, expended_node= [], []
                while len(stack) > 0 and attempt < 100000:
                    cur_h,cur_r,cur_t = stack.pop(-1)
                    
                    if cur_t == "nan":
                        rt_list = self.graph_h2rt[cur_h] if cur_h in self.graph_h2rt else []
                        if len(cur_r.split('|')) < self.max_length and len(rt_list) > 0:
                             for r_, t_ in rt_list:
                                stack.append((cur_h, r_, t_));attempt+=1
                        expended_node.append(cur_h)
                    
                    else:
                        rt_list = self.graph_h2rt[cur_t] if cur_t in self.graph_h2rt else [] 
                        if len(cur_h.split('|')) < (2*(self.max_length-1)) and len(rt_list) > 0 and cur_t not in expended_node:
                            for r_, t_ in rt_list:
                                stack.append((cur_h+"|"+cur_r , cur_t+'|' + r_, t_))
                                attempt += 1 
                        expended_node.append(cur_t)
                  
                    if  cur_t == e2 and cur_r != head:
                        path=[];tmp=[cur_h.split("|"),cur_r.split("|"),cur_t.split("|")]
                        for rth in tmp:
                            for _ in rth:
                                path.append(_)
                        meta_seq.append(path)
                if len(meta_seq)==0:
                    continue
                self.meta_path[head].extend(meta_seq)
            write_txt(self.meta_path[head],path_example +head+f'-{self.max_length}.txt')  
            instance_paths.append(path_example +head+f'-{self.max_length}.txt')
            save_paths.append(f"results/{self.data_set}/metapath/{head}-{self.max_length}.txt")
            self.logger.info(f'Sample {len(self.meta_path[head])} instance paths for {head}') 
        
        for instance_path,save_path in zip(instance_paths,save_paths):
            self.generate_metapath(paths=instance_path,
                                   ent2type=self.ent2type,
                                   save_path=save_path)  
                   
        return self.meta_path
    
    def generate_metapath(self,paths,ent2type,save_path,MAX_TYPE=2,mode='NELL'):
        if mode!='NELL':
            tree=typetree()
        ent_ser,rel_ser=load_paths(paths)

        type_num={}
        for ent in ent_ser:
            for e in ent:
                typelist=ent2type[e]

                for t in typelist:
                    if t in type_num:
                        type_num[t]+=1
                    else:
                        type_num[t]=1

        typeinfo=[]
        for ent in ent_ser:
            typeinfo_unit=[]
            for e in ent:
                typelist=ent2type[e]
                if mode!='NELL':
                    res=tree.filter_typelist(typelist)
                    candidate_set=set([])
                    for pp in res:
                        candidate_set.add(pp[-1])
                else:
                    candidate_set=set(typelist.copy())
                if len(candidate_set)>MAX_TYPE:
                    sorted_ty=[(i,type_num[i]) for i in candidate_set]
                    sorted_ty=sorted(sorted_ty,key=lambda x:x[1],reverse=True)
                    candidate_set=[ge[0] for ge in sorted_ty[:MAX_TYPE]]
                assert len(candidate_set)>0
                typeinfo_unit.append(candidate_set)
            typeinfo.append(typeinfo_unit)
        metapath={}
        for u,v in zip(typeinfo,rel_ser):
            path_l=len(v)+len(u)
            
            metapath_u=[0]*path_l
            for k,r in enumerate(v):
                metapath_u[k*2+1]=r
            for ty_u in product(*u):
                for k,t in enumerate(ty_u):
                    metapath_u[k*2]=t 
                try:
                    sk=tuple(metapath_u)
                    if sk in metapath:
                        metapath[sk]+=1 
                    else:
                        metapath[sk]=1
                except TypeError:
                    print(tuple(metapath_u))
        sort_metapath=sorted(metapath.items(),key=lambda x:x[1],reverse=True)
        with open(save_path,'w') as fin:
            wrt_str=['\t'.join(k)+'\t'+str(v) for k,v in sort_metapath]
            wrt_str='\n'.join(wrt_str)
            fin.write(wrt_str)
  
    
    def generate_with_llama(self,target_heads): 
        self.load_llama()
        self.load_metapaths(target_heads)

        generate_metapaths=[]
        for _ in range(self.generation_rounds):
            examples=random.choices(self.meta_path,weights=self.prob,k=30)
            try:
                metapaths=self.metapath_llama(target_heads,examples)
                generate_metapaths.extend(metapaths.split("\n"))
            except:
                continue
        return generate_metapaths

    def metapath_llama(self, target_heads,examples):
        template_background=("In the context of Heterogeneous Information Networks (HINs), a meta-path is a formalized definition of a composite relation between multi types of nodes in the network. "
                             "A meta-path can explain a certain relation below are three example meta-paths.Each meta-path should start with an type, involve interactions between types and relations, and end with an type. "
                             f"You need to generate meta-paths with {self.max_length*2+1} words in total.")
        
        query=f"Do not give me any explanation or any other information. Just give me ten more meta-paths with high score to explain relation {target_heads}."
        inputs =[{'system': template_background},
                {'system': f"Here are some example meta-paths for {target_heads}.Each meta-paths is followed with its score:{examples}"},#规则库不能太大了
                {'user': query}]
        inputs = self.format_llama(inputs)

        input_id = self.tokenizer(inputs, return_tensors="pt").input_ids.to(f"cuda:{self.device}")
        output = self.model.generate(input_ids=input_id,max_new_tokens=800, temperature=0.3)
        text = self.tokenizer.decode((output[0]), skip_special_tokens=True)
        
        meta_path_final=self.clean_metapath(text)  
        return meta_path_final
    
    def load_llama(self):
        model_dir = get_model_path(self.model_name, '7B')
        self.model=LlamaForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float16,
                                                    device_map={"model": int(self.device), "lm_head": int(self.device)})
        self.tokenizer = LlamaTokenizer.from_pretrained(model_dir)

        
    def format_llama(self,texts):
        B_INST, E_INST = "<s>[INST]", "[/INST]"
        B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n"
        out = []
        for entry in texts:
            role, text = list(entry.items())[0]
            out.append(f'{B_SYS}{ text }{E_SYS}' if 'sys' in role.lower() else text)
        return B_INST + ''.join(out) + E_INST
    
    def load_metapaths(self,target_heads):
        meta_paths=load_json(f"results/{self.data_set}/Rank/{target_heads}-{self.max_length}.json")
        score_=[(value[0]+value[1]) for _,value in meta_paths.items()] 
        total=sum(score_)
        prob=[score/total for score in score_] #
        self.meta_path=[i for i in meta_paths.keys()]
        self.prob=prob

    def clean_metapath(self,text):
          text=text.split("[/INST]")[1].split("\n\n")[1].split("\n")
          meta_path_final=[]
          for metapath in text:
            tmp=metapath.split(".")[1].replace("[","").replace("]","").replace("'","").replace("\"","").replace(" ","").split(",")
            flag=True
            for i,item in enumerate(tmp):
                if i%2==0:
                    if item not in self.type:
                        match_list=get_close_matches(item,self.type,n=1)
                        if len(match_list)==0: 
                            flag =False
                            continue 
                        tmp[i]=match_list[0]
                        
                else: 
                    if item not in self.relation:
                        match_list=get_close_matches(item,self.relation,n=1)
                        if len(match_list)==0: 
                            flag =False
                            continue
                        tmp[i]=match_list[0]
            if flag: 
                meta_path_final.append(tmp)