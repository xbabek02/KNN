import os 
import torch
import torch.utils.data
import torch.backends.cudnn as cudnn

from utils import utils
import editdistance

def send_inputs_to_device(dictionary, device):
    return {key: value.to(device=device) if isinstance(value, torch.Tensor) else value for key, value in dictionary.items()}


def validation(model, criterion, evaluation_loader, converter, device, save=False, epoch=None, step=None):
    """ validation or evaluation """

    norm_ED = 0
    norm_ED_wer = 0

    tot_ED = 0
    tot_ED_wer = 0

    valid_loss = 0.0
    length_of_gt = 0
    length_of_gt_wer = 0
    count = 0
    all_preds_str = []
    all_labels = []
    
    all_cers = []
    all_wers = []

    for i, inputs in enumerate(evaluation_loader):
        if i % 1_000 == 0:
            print(f"validating: {i}/{len(evaluation_loader)}")
        
        labels = inputs['label']
        del(inputs['label'])

        text_for_loss, length_for_loss = converter.encode(labels)
        batch_size = length_for_loss.size(0)

        inputs = send_inputs_to_device(inputs, device)

        preds = model(**inputs)
        preds = preds.float()
        preds_size = torch.IntTensor([preds.size(1)] * batch_size)
        preds = preds.permute(1, 0, 2).log_softmax(2).cpu()

        cost = criterion(preds, text_for_loss, preds_size, length_for_loss).mean()

        _, preds_index = preds.max(2)
        preds_index = preds_index.transpose(1, 0).contiguous().view(-1)
        preds_str = converter.decode(preds_index.data, preds_size.data)
        valid_loss += cost.item()
        count += 1

        all_preds_str.extend(preds_str)
        all_labels.extend(labels)

        for pred_cer, gt_cer in zip(preds_str, labels):
            tmp_ED = editdistance.eval(pred_cer, gt_cer)
            if len(gt_cer) == 0:
                norm_ED += 1
            else:
                norm_ED += tmp_ED / float(len(gt_cer))
            tot_ED += tmp_ED
            length_of_gt += len(gt_cer)
            all_cers.append(tmp_ED / float(len(gt_cer)))

        for pred_wer, gt_wer in zip(preds_str, labels):
            pred_wer = utils.format_string_for_wer(pred_wer)
            gt_wer = utils.format_string_for_wer(gt_wer)
            pred_wer = pred_wer.split()
            gt_wer = gt_wer.split()
            tmp_ED_wer = editdistance.eval(pred_wer, gt_wer)

            if len(gt_wer) == 0:
                norm_ED_wer += 1
            else:
                norm_ED_wer += tmp_ED_wer / float(len(gt_wer))

            tot_ED_wer += tmp_ED_wer
            length_of_gt_wer += len(gt_wer)
            all_wers.append(tmp_ED_wer / float(len(gt_wer)))

    val_loss = valid_loss / count
    CER = tot_ED / float(length_of_gt)
    WER = tot_ED_wer / float(length_of_gt_wer)
    
    if save:
        epoch = f"_{epoch}" if epoch else ""
        step = f"_{step}" if step else ""
        
        utils.store_predictions(all_preds_str, all_labels, os.path.join("output", f"predictions{epoch}{step}.pkl"))
        utils.store_predictions(all_cers, all_wers, os.path.join("output", f"cers_wers{epoch}{step}.pkl"))
    
    return val_loss, CER, WER, all_cers, all_wers, all_preds_str, all_labels