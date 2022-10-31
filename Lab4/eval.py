from dataset import bair_robot_pushing_dataset
from utils import finn_eval_seq, pred, plot_pred

import numpy as np
import torch
from torch.utils.data import DataLoader

def eval(data, loader, iterator, modules, args, device):
    psnr_list = []
    for _ in range(len(data) // args.batch_size):
        try:
            seq, cond = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            seq, cond = next(iterator)
        with torch.no_grad():
            pred_seq = pred(seq.to(device), cond.to(device), modules, args, device)            
        seq = seq.transpose_(0, 1)
        _, _, psnr = finn_eval_seq(seq[args.n_past:args.n_past+args.n_future], pred_seq[args.n_past:args.n_past+args.n_future])
        psnr_list.append(psnr)
        
    ave_psnr = np.mean(np.concatenate(psnr_list))

    return ave_psnr

def main():
    model = torch.load('./logs/fp/rnn_size=256-predictor-posterior-rnn_layers=2-1-n_past=2-n_future=10-lr=0.0020-g_dim=128-z_dim=64-last_frame_skip=False-beta=0.0001000/model.pth')
    encoder = model['encoder']
    decoder = model['decoder']
    frame_predictor = model['frame_predictor']
    posterior = model['posterior']
    args = model['args']

    if args.cuda:
        assert torch.cuda.is_available(), 'CUDA is not available.'
        device = 'cuda'
    else:
        device = 'cpu'

    # --------- transfer to device ------------------------------------
    frame_predictor.to(device)
    posterior.to(device)
    encoder.to(device)
    decoder.to(device)

    # --------- load a dataset ------------------------------------
    test_data = bair_robot_pushing_dataset(args, 'test')
    test_loader = DataLoader(test_data,
                            num_workers=args.num_workers,
                            batch_size=args.batch_size,
                            shuffle=True,
                            drop_last=True,
                            pin_memory=True)
    test_iterator = iter(test_loader)

    modules = {
        'frame_predictor': frame_predictor,
        'posterior': posterior,
        'encoder': encoder,
        'decoder': decoder,
    }

    # --------- evaluate ------------------------------------
    ave_psnr = eval(test_data, test_loader, test_iterator, modules, args, device)

    with open('./test_logs/test_record.txt', 'w') as test_record:
        test_record.write(('====================== test psnr = {:.5f} ========================\n'.format(ave_psnr)))

    try:
        test_seq, test_cond = next(test_iterator)
    except StopIteration:
        test_iterator = iter(test_loader)
        test_seq, test_cond = next(test_iterator)

    with torch.no_grad():
        plot_pred(test_seq.to(device), test_cond.to(device), modules, 0, args, 'test')

if __name__ == '__main__':
    main()