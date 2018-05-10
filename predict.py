'''
 @Date  : 2017/12/28
 @Author: Shuming Ma
 @mail  : shumingma@pku.edu.cn 
 @homepage: shumingma.com
'''

import torch
import torch.utils.data
import argparse
import time
import pickle

import opts
import utils
import models

parser = argparse.ArgumentParser(description='train.py')
parser.add_argument('-src_file', required=True, help="input file for the data")
parser.add_argument('-tgt_file', required=True, help="output file for the data")

opts.model_opts(parser)

opt = parser.parse_args()
config = utils.read_config(opt.config)
torch.manual_seed(opt.seed)
opts.convert_to_config(opt, config)

# cuda
use_cuda = torch.cuda.is_available() and len(opt.gpus) > 0
config.use_cuda = use_cuda
if use_cuda:
    torch.cuda.set_device(opt.gpus[0])
    torch.cuda.manual_seed(opt.seed)


def load_data():
    print('loading data...\n')
    datas = pickle.load(open(config.data+'data.pkl', 'rb'))

    src_vocab = datas['dict']['src']
    tgt_vocab = datas['dict']['tgt']
    config.src_vocab_size = src_vocab.size()
    config.tgt_vocab_size = tgt_vocab.size()

    infos = {}
    f_src = open(opt.src_file, 'r', encoding='utf8').read().strip().lower().split('\n')
    f_tgt = open(opt.tgt_file, 'r', encoding='utf8').read().strip().lower().split('\n')

    srcIds = [src_vocab.convertToIdx(src_line.split(), utils.UNK_WORD) for src_line in f_src]
    tgtIds = [tgt_vocab.convertToIdx(tgt_line.split(), utils.UNK_WORD, utils.BOS_WORD, utils.EOS_WORD) for tgt_line in f_tgt]

    with open(opt.src_file+'.id', 'w') as src_id, open(opt.tgt_file+'.id', 'w') as tgt_id:
        for ids in srcIds:
            src_id.write(" ".join(list(map(str, ids)))+'\n')
        for ids in tgtIds:
            tgt_id.write(" ".join(list(map(str, ids)))+'\n')



def build_model(checkpoints):
    # model
    print('building model...\n')
    model = getattr(models, opt.model)(config)
    if checkpoints is not None:
        model.load_state_dict(checkpoints['model'])
    if use_cuda:
        model.cuda()

    return model


def main():
    # checkpoint
    if opt.restore:
        print('loading checkpoint...\n')
        checkpoints = torch.load(opt.restore)
    else:
        checkpoints = None

    datas = load_data()
    print_log, log_path = build_log()
    model, optim, print_log = build_model(checkpoints, print_log)

    params = {'updates': 0, 'report_loss': 0, 'report_total': 0,
              'report_correct': 0, 'report_time': time.time(),
              'log': print_log, 'log_path': log_path}
    for metric in config.metrics:
        params[metric] = []
    if opt.restore:
        params['updates'] = checkpoints['updates']

    for i in range(1, config.epoch + 1):
        train_model(model, datas, optim, i, params)

    for metric in config.metrics:
        print_log("Best %s score: %.2f\n" % (metric, max(params[metric])))


if __name__ == '__main__':
    main()
