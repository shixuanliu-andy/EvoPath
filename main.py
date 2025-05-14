import logging
import os.path
from sample import Dataset
import prediction
from utils import get_type_dict
from validate import RuleValidator
import argparse


def set_log(dataset):
    # Initialize logger with dual handlers (console + file)
    logger = logging.getLogger(f'EvoPath')
    logger = logging.getLogger(f'{dataset}')
    logger.setLevel(logging.INFO)
    
    # Console handler configuration
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)

    return logger


# New command line argument parsing function
def parse_args():
    parser = argparse.ArgumentParser(description='EvoPath Configuration')
    # Add argument definitions with help messages
    parser.add_argument('--data_set', type=str, default='nell', help='Dataset name') 
    # parser.add_argument('--data_set', type=str, default='yago', help='Dataset name') 
    # parser.add_argument('--data_set', type=str, default='dbpedia', help='Dataset name') 
    parser.add_argument('--threshold', type=float, default=0.0, help='Threshold value')
    # parser.add_argument('--target_heads', nargs='+', default="isCitizenOf", #["worksfor","teamplaysagainstteam","competesWith"]
    #                   help='Target relation heads')
    parser.add_argument('--target_heads', nargs='+', default="worksfor", #["worksfor","teamplaysagainstteam","competesWith"]
                      help='Target relation heads')
    # parser.add_argument('--target_heads', nargs='+', default=None, 
    #                   help='Target relation heads')
    parser.add_argument('--generation', action='store_true', default=True, 
                      help='Enable path generation')
    parser.add_argument('--device', type=str, default='3', help='CUDA device number')
    parser.add_argument('--task', type=str, default='lp', choices=['lp', 'kbc'], 
                      help='Task type: lp=Link Prediction, kbc=Knowledge Base Completion')
    return parser.parse_args()

class Args():
    def __init__(self, data_set=None, thresold=None, target_heads=None):

        # Hybrid parameter initialization (CLI优先)
        cli_args = parse_args()
        
        # Core experiment settings
        self.data_set = cli_args.data_set or data_set  # Dataset identifier
        self.thresold = cli_args.threshold if thresold is None else thresold  # Path filtering threshold
        self.target_heads = cli_args.target_heads if target_heads is None else target_heads  # Target relations
        
        # File system paths configuration
        self.path_metapath = f"results/{self.data_set}/metapath/"  # Raw meta-paths storage
        self.path_metapath_gen = f"results/{self.data_set}/generate/"  # Generated paths
        self.path_rank = f"results/{self.data_set}/Rank/"  # Ranking results
        self.path_root = f"results/{self.data_set}/"
        self.path_data = f"dataset/{self.data_set}/"
        for path in [self.path_root, self.path_metapath, self.path_metapath_gen, 
                    self.path_rank, self.path_data]:
            os.makedirs(path, exist_ok=True)  

        # Logger initialization with file handler
        self.logger = set_log(self.data_set)  # Central logger instance
        file_handler = logging.FileHandler(os.path.join(self.path_root,'EvoPath.log'))
        self.logger.addHandler(file_handler)  # Add persistent log storage
        
        # Runtime parameters (consider adding command line options for these)
        self.num = 10        
        self.max_length = 3    
        self.generation_rounds = 10  
        self.device = cli_args.device
        self.task = cli_args.task


def main(args):
    sample = Dataset(args)
    if "kbc" in args.task: args.target_heads = sample.heads
    sample.bfs_sample(args.max_length, args.num,args.path_data,target_head=args.target_heads)
    # sample.generate_with_llama(args.target_heads)

    fact_dict, rel_dict = sample.graph_r2ht_idx_fact, sample.graph_r2ht_idx
    N,rel2idx,ent2idx = len(sample.ent2idx), sample.head_rdict.rel2idx,sample.ent2idx
    type2idx,ent2type=sample.type2idx,sample.ent2type
    args.type_dict = get_type_dict(ent2type,type2idx,ent2idx)
    args.type2idx = type2idx;args.rel2idx=rel2idx
    Validator = RuleValidator(fact_dict,rel_dict,N,args.target_heads,
                                   rel2idx, type2idx, ent2idx, 
                                   ent2type,args.path_rank, args.path_metapath)
    Validator.RankMetapath(args.max_length)

    predictor = prediction.Prediction(args)
    if 'lp' in args.task:  # Link Prediction task
        mean_ap,var_ap,mean_auc,var_auc = predictor.Link_Prediction()
        return mean_ap,var_ap,mean_auc,var_auc  # Evaluation metrics
    if 'kbc' in args.task:  # Knowledge Base Completion
        predictor.KGC()  # Knowledge graph completion

if __name__ == "__main__":
    main(Args())  # Execute pipeline   
    

