import torch,utils
from transformers import LlamaTokenizer, LlamaForCausalLM
from langchain.prompts.example_selector import LengthBasedExampleSelector
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
import math
from tqdm import tqdm
from scipy.sparse import identity,csr_matrix
from collections import  defaultdict,OrderedDict
from copy import deepcopy
torch.cuda.empty_cache()


class RuleValidator:
    #这个类接受相关数据，读取规则对所有规则按照规则头计算置信度，main函数返回排序的结果
    def __init__(self,fact_dict,rel_dict,N,heads,
                 rel2idx,type2idx,ent2idx,
                 ent2type,path_rank,path_example):
        self.fact_dict = fact_dict
        self.rel_dict =rel_dict
        self.N = N
        self.heads =heads
        self.rel2idx = rel2idx;self.type2idx= type2idx;self.ent2idx=ent2idx
        self.ent2type = ent2type
        self.metapaths={} #将不同规则头对应的规则分别加载
        self.path_rank=path_rank
        self.path_example=path_example

        self.type_dict=defaultdict(list)
        for key,value in self.ent2type.items():
            for i in value:
                if self.type2idx[i] not in self.type_dict.keys():
                    self.type_dict[self.type2idx[i]] = [self.ent2idx[key]]
                else:
                    self.type_dict[self.type2idx[i]].append(self.ent2idx[key])

    def RuleScore(self,rule_head,rule_body):
        """
        计算置信度和支持度 需要将字符串转化为id train_rules里面有
        :param rule_head: 对应all_rule中的所有规则头
        :param rule_body:  前面rule_head对应的规则体
        :return: conf 置信度, coverage支持度, pca-conf pca置信度
        """
        # PCA conf 计算需要将rel_dict 替换为fact里面的dict，N转化为fact里面的N

        conf,coverage =self.Conf(rule_body,rule_head,self.rel_dict)
        pca_conf,coverage =self.Conf(rule_body,rule_head,self.fact_dict)
        return pca_conf,conf,coverage

    def Conf(self,rule_body,rule_head,rel_dict):
            #调用可以放入不同的dict计算pca_conf和conf
            graph_dict = defaultdict(csr_matrix)
            #构建当前规则的可达矩阵
            for i in range(len(rule_body)):
                row, col, data = [], [], []
                if i %2 ==1: #关系的可达矩阵
                    for [src, tgt] in rel_dict[int(rule_body[i])]: #需要检查前后实体是否满足关系
                        if src in self.type_dict[rule_body[i-1]] and tgt in self.type_dict[rule_body[i+1]]:
                            row.append(src);col.append(tgt);data.append(1.0)
                graph_dict[rule_body[i]] = csr_matrix((deepcopy(data), (deepcopy(row), deepcopy(col))), shape=(self.N, self.N))

            row, col, data = [], [], []
            for [src, tgt] in rel_dict[int(rule_head)]:
                row.append(src);col.append(tgt);data.append(1.0)
            graph_dict[rule_head] = csr_matrix((deepcopy(data), (deepcopy(row), deepcopy(col))), shape=(self.N, self.N))

            result = identity(self.N)
            for i in range(len(rule_body)):
                if i %2 ==0: 
                    continue
                result *= graph_dict[int(rule_body[i])]
            result1 = result.multiply(graph_dict[rule_head])

            body = result.nnz  # number of pairs that have the meta-path
            if body ==0:
                return 0,0
            A = result1.nnz # number of pairs that have both query relation and meta-path  the support of rule \rho
            (conf, coverage) = (0, 0) if body == 0 else (float(A / body), float(A / len(self.rel_dict[rule_head])))
            return conf, coverage

    def Metapath2id(self,metapath):
        idx = []
        for i in range(len(metapath)):
            if i %2==0: idx.append(self.type2idx[metapath[i]])
            else: idx.append(self.rel2idx[metapath[i]])
        return idx

    def RankMetapath(self,length):
        heads=[self.heads] if type(self.heads)==str else self.heads
        for head in heads:
            with open(self.path_example+ f"{head}-{length}.txt", "r") as f:
                lines=f.readlines()
            rule_head=self.rel2idx[head]
            for line in tqdm(lines,desc=f"For length {length} metapaths under relation {head} ",leave=False):
                #对每一条规则
                metapath=line.strip().split("\t")[:-1]
                rule_body=self.Metapath2id(metapath)
                pca_conf, conf, coverage = self.RuleScore(rule_head,rule_body)
                self.metapaths[str(metapath)]=(pca_conf, conf, coverage)
            self.metapaths = dict(sorted(self.metapaths.items(), key=lambda x: x[1], reverse=True))
            utils.write_json(self.metapaths,self.path_rank + f"{head}-{length}.json")

