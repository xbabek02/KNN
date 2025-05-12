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
from dataset import SynthtigerDataset
from torch.utils.data import random_split
from torch.utils.data import Subset

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

    datasets_locations = ["../datasets_f/dataset1", "../datasets_f/dataset2"]

    def load_samples_from_directory(root_dir, gt_file):
        samples = []

        gt_path = os.path.join(root_dir, gt_file)
        if not os.path.exists(gt_path):
            print(f"Warning: {gt_path} not found. Skipping.")
            return
        
        with open(gt_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    img_path, text = parts
                    full_img_path = os.path.join(root_dir, img_path)
                    samples.append((full_img_path, text))

        return samples

    gt_file = "gt_basic.txt"
    datasets_locations = ["../datasets_f/dataset1", "../datasets_f/dataset2"]

    # read samples
    samples = []
    for root_dir in datasets_locations:
        samples += load_samples_from_directory(root_dir, gt_file)

    random.seed(42)          # for reproducible shuffles
    random.shuffle(samples)  # shuffles the list in place

    full_data = SynthtigerDataset(config=config, samples=samples)

    n = len(samples)
    print(f"Number of samples = {n}")

    n_train = int(0.9 * n)
    n_valid = int(0.02 * n)
    n_test  = n - n_train - n_valid

    train_dataset, validation_dataset, test_dataset = random_split(
        full_data,
        [n_train, n_valid, n_test],
        generator=torch.Generator().manual_seed(42)
    )

    optimizer = sam.SAM(model.parameters(), torch.optim.AdamW, lr=1e-7, betas=(0.9, 0.99), weight_decay=args.weight_decay)
    criterion = torch.nn.CTCLoss(reduction='none', zero_infinity=True)
    converter = utils.CTCLabelConverter(alphabet)

    train_dataloader = DataLoader(train_dataset, batch_size=100, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=100, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=100, shuffle=False)

    best_cer, best_wer = 1e+6, 1e+6
    train_loss = 0.0
    nb_iter = 0
    EPOCHS = 1

    checkpoint_path = os.path.join(args.save_dir, 'best_CER.pth')
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 3. Load states
    model.load_state_dict(checkpoint['model'])
    model_ema.ema.load_state_dict(checkpoint['state_dict_ema'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    for epoch in range(EPOCHS):
        logger.info(f'\nEpoch {epoch + 1}/{EPOCHS}')
        
        for step, inputs in enumerate(train_dataloader):
            nb_iter += 1
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
                    val_loss, val_cer, val_wer, preds, labels = valid.validation(model_ema.ema,
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
