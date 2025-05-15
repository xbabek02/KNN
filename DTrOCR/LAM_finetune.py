
import torch
import torch.utils.data
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import os
import json
import valid
from utils import utils
from utils import sam
from utils import option
from model import DTrOCRLMHeadModel
from torch.utils.data import DataLoader

from config import DTrOCRConfig
from typing import Tuple
import tqdm
import random
from lam_dataset import LAM

import kagglehub


def compute_loss(args, model, inputs, batch_size, criterion, text, length, pred_device):
    def send_inputs_to_device(dictionary, device):
        return {key: value.to(device=device) if isinstance(value, torch.Tensor) else value for key, value in dictionary.items()}

    inputs = send_inputs_to_device(inputs, pred_device)
    preds = model(**inputs)
    preds = preds.float()
    preds_size = torch.IntTensor([preds.size(1)] * batch_size).to(pred_device)
    preds = preds.permute(1, 0, 2).log_softmax(2)

    loss = criterion(preds.to(pred_device), text.to(pred_device), preds_size, length.to(pred_device)).mean()

    return loss


def load_lam_dataloaders(config:DTrOCRConfig,charset=None, batch_size=100):
    dataset_name = "vpippi/lam-dataset"
    path = kagglehub.dataset_download(dataset_name)

    print("Path to dataset files:", path)

    path_to_train_json = f"{path}/LAM/lines/split/basic/train.json"
    path_to_test_json = f"{path}/LAM/lines/split/basic/test.json"
    path_to_valid_json = f"{path}/LAM/lines/split/basic/val.json"
    path_to_images = f"{path}/LAM/lines/img/"

    os.path.exists(path_to_train_json) and os.path.exists(path_to_images) and os.path.exists(path_to_test_json) and os.path.exists(path_to_valid_json)

    train_dataset = LAM(f"{path}/LAM", 'basic', charset=charset,  nameset='train', config=config)
    val_dataset = LAM(f"{path}/LAM", 'basic', charset=charset,  nameset='val', config=config)
    test_dataset  = LAM(f"{path}/LAM", 'basic', charset=charset,  nameset='test', config=config)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    return train_loader, val_loader, test_loader

def main():

    alphabet = "!\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~ ÈàèéìòùÀÈÉÌÒÙ’%"

    config = DTrOCRConfig(
        # attn_implementation='flash_attention_2'
        alphabet_size=len(alphabet)
    )

    args = option.get_args_parser()
    torch.manual_seed(args.seed)

    args.save_dir = os.path.join(args.out_dir, args.exp_name)
    os.makedirs(args.save_dir, exist_ok=True)

    logger = utils.get_logger(args.save_dir)
    logger.info(json.dumps(vars(args), indent=4, sort_keys=True))
    writer = SummaryWriter(args.save_dir)

    dev_str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(dev_str)

    model = DTrOCRLMHeadModel(config)

    total_param = sum(p.numel() for p in model.parameters())
    logger.info('total_param is {}'.format(total_param))

    model.train()
    model = model.to(device)
    model_ema = utils.ModelEma(model, args.ema_decay, device=device)
    model.zero_grad()

    optimizer = sam.SAM(model.parameters(), torch.optim.AdamW, lr=1e-7, betas=(0.9, 0.99), weight_decay=args.weight_decay)
    criterion = torch.nn.CTCLoss(reduction='none', zero_infinity=True)
    converter = utils.CTCLabelConverter(alphabet)

    train_dataloader, validation_dataloader, test_dataloader = load_lam_dataloaders(config, alphabet)

    best_cer, best_wer = 1e+6, 1e+6
    train_loss = 0.0
    step = 0
    EPOCHS = args.epochs

    checkpoint_path = os.path.join(args.save_dir, 'best_CER.pth')
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 3. Load states
    model.load_state_dict(checkpoint['model'])
    model_ema.ema.load_state_dict(checkpoint['state_dict_ema'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    nb_iter = 0

    for epoch in range(EPOCHS):
        logger.info(f'\nEpoch {epoch + 1}/{EPOCHS}')
        
        for step, inputs in enumerate(train_dataloader):
            nb_iter+=1
            label = inputs['label']

            del(inputs['label'])
            optimizer, current_lr = utils.update_lr_cos(nb_iter, args.warm_up_iter, args.total_iter, args.max_lr, optimizer)

            optimizer.zero_grad()
            text, length = converter.encode(label)
            batch_size = length.size(0)
            
            loss = compute_loss(args, model, inputs, batch_size, criterion, text, length, device)
            loss.backward()
            optimizer.first_step(zero_grad=True)
            compute_loss(args, model, inputs, batch_size, criterion, text, length, device).backward()
            optimizer.second_step(zero_grad=True)
            model.zero_grad()
            model_ema.update(model, num_updates=nb_iter / 2)
            train_loss += loss.item()

            if nb_iter % args.print_iter == 0:
                train_loss_avg = train_loss / args.print_iter

                logger.info(f'Iter : {nb_iter} \t LR : {current_lr:0.5f} \t training loss : {train_loss_avg:0.5f} \t ' )

                writer.add_scalar('./Train/lr', current_lr, nb_iter)
                writer.add_scalar('./Train/train_loss', train_loss_avg, nb_iter)
                train_loss = 0.0

            if nb_iter % args.eval_iter == 0:
                model.eval()
                with torch.no_grad():
                    val_loss, val_cer, val_wer, *_ = valid.validation(model_ema.ema,
                                                                                criterion,
                                                                                validation_dataloader,
                                                                                converter, device)

                    if val_cer < best_cer:
                        logger.info(f'CER improved from {best_cer:.4f} to {val_cer:.4f}!!!')
                        best_cer = val_cer
                        checkpoint = {
                            'model': model.state_dict(),
                            'state_dict_ema': model_ema.ema.state_dict(),
                            'optimizer': optimizer.state_dict(),
                        }
                        torch.save(checkpoint, os.path.join(args.save_dir, 'best_CER.pth'))

                    if val_wer < best_wer:
                        logger.info(f'WER improved from {best_wer:.4f} to {val_wer:.4f}!!!')
                        best_wer = val_wer
                        checkpoint = {
                            'model': model.state_dict(),
                            'state_dict_ema': model_ema.ema.state_dict(),
                            'optimizer': optimizer.state_dict(),
                        }
                        torch.save(checkpoint, os.path.join(args.save_dir, 'best_WER.pth'))

                    logger.info(
                        f'Val. loss : {val_loss:0.3f} \t CER : {val_cer:0.4f} \t WER : {val_wer:0.4f} \t ')

                    writer.add_scalar('./VAL/CER', val_cer, nb_iter)
                    writer.add_scalar('./VAL/WER', val_wer, nb_iter)
                    writer.add_scalar('./VAL/bestCER', best_cer, nb_iter)
                    writer.add_scalar('./VAL/bestWER', best_wer, nb_iter)
                    writer.add_scalar('./VAL/val_loss', val_loss, nb_iter)
                    model.train()

if __name__ == '__main__':
    main()
