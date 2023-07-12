import torch
import numpy as np
from networks.resnet import resnet50
from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score
from options.test_options import TestOptions
from data import create_dataloader
from tqdm import tqdm
from networks.vit import ViTModel

def validate(model, opt):
    data_loader = create_dataloader(opt)
    best_acc, best_ap, best_r_acc, best_f_acc = 0, 0, 0, 0
    best_y_true, best_y_pred = [], []
    best_threshold = 0.5
    
    thresholds = [0.5]
    if opt.with_threshold:
        thresholds = np.arange(0,1,0.1)
    
        with torch.no_grad():
            y_true, y_pred = [], []
            for img, label in tqdm(data_loader, "validation: "):
                in_tens = img.cuda()
                y_pred.extend(model(in_tens).sigmoid().flatten().tolist())
                y_true.extend(label.flatten().tolist())

        for threshold in thresholds:
            y_true, y_pred = np.array(y_true), np.array(y_pred)
            r_acc = accuracy_score(y_true[y_true==0], y_pred[y_true==0] > threshold)
            f_acc = accuracy_score(y_true[y_true==1], y_pred[y_true==1] > threshold)
            acc = accuracy_score(y_true, y_pred > threshold)
            ap = average_precision_score(y_true, y_pred)
            
            if acc > best_acc:
                best_acc = acc
                best_ap = ap
                best_r_acc = r_acc
                best_f_acc = f_acc
                best_y_true = y_true
                best_y_pred = y_pred
                best_threshold = threshold
            
    return best_acc, best_ap, best_r_acc, best_f_acc, best_y_true, best_y_pred, best_threshold


if __name__ == '__main__':
    opt = TestOptions().parse(print_options=False)

    # model = resnet50(num_classes=1)
    model = ViTModel()

    state_dict = torch.load(opt.model_path, map_location='cpu')
    model.load_state_dict(state_dict['model'])
    model.cuda()
    model.eval()

    acc, avg_precision, r_acc, f_acc, y_true, y_pred = validate(model, opt)

    print("accuracy:", acc)
    print("average precision:", avg_precision)

    print("accuracy of real images:", r_acc)
    print("accuracy of fake images:", f_acc)
