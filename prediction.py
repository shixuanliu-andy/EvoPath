import os
from tqdm import tqdm
import random
from itertools import chain
from collections import defaultdict
import numpy as np
from scipy import sparse
from utils import load_txt, parse_triple,gen_filtered_query,write_txt,negative_sampling,load_json
from sample import Dataset
from torch.utils.data import DataLoader
from scipy.sparse import csr_matrix, identity
from copy import deepcopy
from sklearn.linear_model import Lasso
from sklearn.metrics import roc_curve, auc, average_precision_score
import ast 



class RuleDataset:
    def __init__(self, heads,fact_rdf, test_rdf, rules, ent2idx, idx2rel, data_dir):
        self.target_heads=heads
        self.rules = rules
        self.ent2idx = ent2idx
        self.e_num = len(ent2idx)
        self.idx2rel = idx2rel
        self.data_dir = data_dir
        self.test_ents = set([i[0] for i in test_rdf])
        self.matrices = {r:sparse.dok_matrix((self.e_num, self.e_num)) for r in self.idx2rel.values()}
        self.start_matrices = {r:sparse.dok_matrix((self.e_num, self.e_num)) for r in self.idx2rel.values()}
        for rdf in fact_rdf:
            h, r, t = parse_triple(rdf)
            self.matrices[r][self.ent2idx[h], self.ent2idx[t]] = 1
            if h in self.test_ents:
                self.start_matrices[r][self.ent2idx[h], self.ent2idx[t]] = 1

    def __len__(self):
        return len(self.rules)
    @staticmethod
    def collate_fn(data):
        return [i[0] for i in data], [i[1] for i in data]

    def get_adj(self):
        tmp=[]
        for rel in self.target_heads:
            path_count = sparse.dok_matrix((self.e_num, self.e_num))
            for body, conf in self.rules[rel].items():
                if body == [] or body == (): continue
                body_adj = sparse.eye(self.e_num) * self.start_matrices[body[1].replace(" ", "")]
                for i in body[3::2]:
                    body_adj = body_adj * self.matrices[i.replace(" ", "")]
                path_count += body_adj * conf
            tmp.append(zip(rel,path_count))
        return dict(chain.from_iterable(tmp))

    def __getitem__(self, idx):
        path_count = sparse.dok_matrix((self.e_num,self.e_num))
        rel = self.idx2rel[idx].replace("inv_","")
        for body, conf in self.rules[rel].items():
            if body == [] or body == (): continue
            body_adj = sparse.eye(self.e_num) * self.start_matrices[body[1]]
            for i in body[3::2]:
                body_adj = body_adj * self.matrices[i.replace(" ","")]
            path_count += body_adj * conf
        return (rel, path_count)

class Prediction:
    def __init__(self, args):
        for key, val in vars(args).items():
            setattr(self, key, val)
        self.data_dir = f'dataset/{args.data_set}/'
        self.dataset = Dataset(args)
        for key in ['fact_rdf', 'train_rdf', 'valid_rdf', 'test_rdf', 'ent2idx']:
            setattr(self, key, vars(self.dataset)[key])
        self.truth = defaultdict(list)
        for rdf in self.fact_rdf+self.train_rdf+self.valid_rdf+self.test_rdf:
            h, r, t = parse_triple(rdf)
            self.truth[(h, r)].append(self.dataset.ent2idx[t])
        self.rdict = self.dataset.rdict
        self.logger = args.logger
        self.entity_vocab=self.dataset.ent2idx
        self.N=len(self.entity_vocab)
        self.metapaths2idx={}
        
        if 'lp' in args.task:
            self.gen_dataset_lp()

    def load_metapath_lp(self):
        self.metapaths = {}
        self.metapaths_conf = {}
        file = f"{self.target_heads}-{self.max_length}.json"
        id=0
        metapaths=load_json(f"{self.path_rank}{file}")
        for key,val in metapaths.items():
            self.metapaths[key]=id;self.metapaths_conf[id]=val[0];id+=1
        self.logger.info(f"Load {len(self.metapaths)} meta-path for rel {self.target_heads}")
   
    def load_metapath_kbc(self):
        self.metapaths = defaultdict(dict)
        self.metapaths_conf = defaultdict(dict)
        if type(self.target_heads) == str:
            files=[f"{self.target_heads}-{self.max_length}.json"]
        elif type(self.target_heads) == list:
            files=[f"{i}-{self.max_length}.json" for i in self.target_heads]
        for file,head in zip(files,self.target_heads):
            metapaths=load_json(f"{self.path_rank}{file}")
            for key,val in metapaths.items():
                key = tuple(ast.literal_eval(key))
                self.metapaths[head][key]=val[0]+val[1]
        # self.logger.info(f"Load {len(self.metapaths)} meta-path for rel {self.target_heads}")

    def KGC(self, pred_num=5000):
        self.logger.info(f"prediction:{self.data_set}  thresold:{self.thresold}" )
        self.load_metapath_kbc()

        rule_dataset = RuleDataset(self.target_heads,self.fact_rdf+self.train_rdf+self.valid_rdf, self.test_rdf,
                                   self.metapaths, self.ent2idx, self.rdict.idx2rel, self.data_dir)
        num_workers = 2 if 'yago' in self.data_dir else len(self.metapaths)//10
        # pred_num = 2000 if 'yagosmall' else pred_num
        rule_loader = DataLoader(rule_dataset, batch_size=2, num_workers=num_workers, collate_fn=RuleDataset.collate_fn)
        print('Getting Scores')
        scores = dict(chain.from_iterable([list(zip(rel, path_count)) for _, (rel, path_count) in tqdm(enumerate(rule_loader))]))
        # scores = rule_dataset.get_adj()

        hit_1, hit_10, mrr = defaultdict(list), defaultdict(list), defaultdict(list)
        hit_1_p, hit_10_p, mrr_p = defaultdict(list), defaultdict(list), defaultdict(list)

        test_rdf = random.sample(self.test_rdf, pred_num) if len(self.test_rdf)>pred_num else self.test_rdf

        for i, rdf in tqdm(enumerate(test_rdf)):
            (q_h, q_r, q_t) = parse_triple(rdf)
            if q_r not in scores:
                continue

            score = np.array(scores[q_r][self.ent2idx[q_h]].todense()).squeeze()
            filter = list(set(self.truth[(q_h, q_r)])-set([self.ent2idx[q_t]]))
            score[filter] = -1
            rank_ = np.sum(score>score[self.ent2idx[q_t]]).item() + 1
            pred_ranks = np.argsort(score)[::-1]
            rank = (pred_ranks == self.ent2idx[q_t]).nonzero()[0].item() + 1

            mrr[q_r].append(1.0/rank)
            hit_1[q_r].append(1 if rank<=1 else 0)
            hit_10[q_r].append(1 if rank<=10 else 0)
            mrr_p[q_r].append(1.0/rank_)
            hit_1_p[q_r].append(1 if rank_<=1 else 0)
            hit_10_p[q_r].append(1 if rank_<=10 else 0)

        mrr_ = np.mean(list(chain.from_iterable(mrr.values()))); mrrs = mrr_
        hit_1_ = np.mean(list(chain.from_iterable(hit_1.values()))); hit_1s = hit_1_
        hit_10_ = np.mean(list(chain.from_iterable(hit_10.values()))); hit_10s = hit_10_
        self.logger.info(f"Results - MRR {mrr_}\t Hits@1 {hit_1_}\t Hits@10 {hit_10_}")

        mrr_p_ = np.mean(list(chain.from_iterable(mrr_p.values()))); mrrs_p= mrr_p_
        hit_1_p_ = np.mean(list(chain.from_iterable(hit_1_p.values()))); hit_1s_p = hit_1_p_
        hit_10_p_ = np.mean(list(chain.from_iterable(hit_10_p.values()))); hit_10s_p = hit_10_p_
        self.logger.info(f"Results Plus- MRR {mrr_p_}\t Hits@1 {hit_1_p_}\t Hits@10 {hit_10_p_}")


    def Link_Prediction(self,regress_split=0.6):
        aps, AUCs = [], []
        
        self.load_metapath_lp() 
        self.metapaths_connects = {v: self.cal_metapath_connect(u) for u, v in tqdm(self.metapaths.items())}
        metapaths_connects = {i: {(j, k): l for [j, k, l] in t} for i, t in self.metapaths_connects.items()}
        
        for i in range(5): 
            features = []  
            for e1, e2, y in self.gen_dataset: 
                feature_vec = [0] * len(self.metapaths) 
                for mp, pair_info in metapaths_connects.items(): 
                    feature_vec[mp] = self.metapaths_conf[mp] if (self.entity_vocab[e1], self.entity_vocab[e2]) in pair_info else 0
                features.append(feature_vec + [y]) 
            random.shuffle(features)
            features = np.array(features)
            features, y_vec = features[:, :-1], features[:, -1]

            # Prune Feature
            indice = np.sum(features, axis=0) != 0
            features = features[:, indice]  
            train_count = int(len(self.gen_dataset) * regress_split)
            features = np.expand_dims(np.sum(features, axis=-1), axis=-1)

            # if self.lp_pool == 'max':
            #     features = np.expand_dims(np.max(features, axis=-1), axis=-1)
            # elif self.lp_pool == 'sum':
            #     features = np.expand_dims(np.sum(features, axis=-1), axis=-1)
            train, test = features[:train_count, :], features[train_count:, :]
            train_y, test_y = y_vec[:train_count], y_vec[train_count:]
            model = Lasso(alpha=1e-5, max_iter=100000);model.fit(train, train_y)
            pre_y = model.predict(test)
            fpr, tpr, thresholds = roc_curve(test_y, pre_y)
            ap, auc_ = average_precision_score(test_y, pre_y), auc(fpr, tpr)
            self.logger.info(f'Round {i} - AP: {ap}, AUC: {auc_}')
            aps.append(ap);AUCs.append(auc_)
        return np.mean(aps), np.var(aps),np.mean(AUCs),np.var(AUCs)

    def gen_dataset_lp(self,test_split=0.2):
        self.graph, self.triples, self.query = defaultdict(set), [], []
        self.dataset.all_rdf = self.dataset.all_rdf+self.dataset.test_rdf
        for rdf in self.dataset.all_rdf:
            e1,rel,e2=tuple(rdf)
            self.triples.append((e1,rel,e2))
            if rel == self.target_heads:
                self.query.append((e1,e2))
            else:
                self.graph[e1].add((rel,e2))
        self.query=list(set(self.query))
        filter_query_dir=self.path_root+f"{self.target_heads}-filter_query"
        if not os.path.exists(filter_query_dir):
            self.filter_query = gen_filtered_query(self.graph, self.query, self.max_length + 1)
            filter_query_str=[];rel_id=self.rel2idx[self.target_heads]
            for [e1,e2] in self.filter_query:
                e1_id=self.dataset.ent2idx[e1];e2_id=self.dataset.ent2idx[e2]
                filter_query_str.append([e1_id,rel_id,e2_id])
            # filter_query_str = [[self.dataset.ent2idx[e1], self.dataset.type2idx(self.target_heads), self.dataset.ent2idx[e2]]
            #                     for [e1, e2] in self.filter_query] #写入ent
            write_txt(filter_query_str, filter_query_dir)
        else:
            self.filter_query = [(e1,e2) for (e1,rel,e2) in load_txt(filter_query_dir)]
        testset_dir = self.path_root + f"{self.target_heads}-testset"
        if not os.path.exists(testset_dir):
            random.shuffle(self.filter_query)
            self.testset = self.filter_query[:int(len(self.filter_query) * test_split)]
            testset_str = [[e1, self.target_heads, e2]for [e1, e2] in self.testset]
            write_txt(testset_str, testset_dir)
        else:
            self.testset = [(e1, e2) for (e1, rel, e2) in load_txt(testset_dir)]
            
        self.gen_negset_lp()

    def gen_negset_lp(self):
        type_e2s=set()
        for _,e2 in self.query:
            for i in self.dataset.ent2type[e2]:
                type_e2s.add(i) 
        type_e2s=list(type_e2s)
        pairs, neg_pairs, existing_query = [tuple(l) for l in self.testset], set(), set(self.query)
        triples = self.triples.copy(); random.shuffle(triples)
        for e1, _, e2 in triples:
            neg_pair_sampled = set(negative_sampling(self.graph, e1, self.dataset.ent2type, type_e2s))
            neg_pairs = neg_pairs | (neg_pair_sampled - existing_query)
            if len(neg_pairs) >= int(0.5*len(pairs)): break
        self.gen_dataset = [[l[0], l[1], 1] for l in pairs] + [[l[0], l[1], 0] for l in neg_pairs] 

    def cal_metapath_connect(self, meta_path):
        graph_dict = defaultdict(csr_matrix)
        meta_path=meta_path.replace("'","").replace(" ","").replace("[","").replace("]","").split(",")
        idx_list=[]
        try:
            for i in range(len(meta_path)):
                if i % 2 == 0:
                    idx_list.append(self.type2idx[meta_path[i]])
                else:
                    idx_list.append(self.rel2idx[meta_path[i]])

            for i in range(len(meta_path)):
                row, col, data = [], [], []
                if i % 2 == 1:  
                    for [src, tgt] in self.dataset.graph_r2ht_idx[idx_list[i]]: 
                        if src in self.type_dict[idx_list[i-1]] and tgt in self.type_dict[idx_list[i+1]]:
                            row.append(src);col.append(tgt);data.append(1.0)
                    graph_dict[i] = csr_matrix((deepcopy(data), (deepcopy(row), deepcopy(col))),
                                                    shape=(self.N, self.N))
            result = identity(self.N)
            for val in graph_dict.values():
                result = result * val
            result = np.concatenate([result.nonzero(), result.data.reshape(1, -1)]).astype(int).T.tolist()
        except:
            zero_matrix = csr_matrix((self.N, self.N))
            result = np.concatenate([zero_matrix.nonzero(), zero_matrix.data.reshape(1, -1)]).astype(int).T.tolist()
            return tuple(result)
        return tuple(result)

