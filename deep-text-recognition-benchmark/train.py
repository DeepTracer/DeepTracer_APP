import os
import sys
import time
import random
import string
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.optim as optim
import torch.utils.data
import numpy as np

# 추가: OpenCV 및 멀티프로세싱 충돌 방지 설정
import cv2
cv2.setNumThreads(0) 

from utils import CTCLabelConverter, CTCLabelConverterForBaiduWarpctc, AttnLabelConverter, Averager
from dataset import hierarchical_dataset, valid_hierarchical_dataset, AlignCollate, Batch_Balanced_Dataset
from model import Model
from test import validation

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(opt):
    # --- [중요] GPU 연산 안정화 설정 ---
    # TPS 역전파 시 멈춤 현상을 방지하기 위해 cuDNN을 비활성화합니다.
    torch.backends.cudnn.enabled = False 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # --------------------------------

    """ dataset preparation """
    if not opt.data_filtering_off:
        print('Filtering the images containing characters which are not in opt.character')
        print('Filtering the images whose label is longer than opt.batch_max_length')

    opt.select_data = opt.select_data.split('-')
    opt.batch_ratio = opt.batch_ratio.split('-')
    
    print("DEBUG: Loading Train Dataset...")
    train_dataset = Batch_Balanced_Dataset(opt)

    log = open(f'./saved_models/{opt.exp_name}/log_dataset.txt', 'a')
    AlignCollate_valid = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
    
    print("DEBUG: Loading Validation Dataset...")
    valid_dataset, valid_dataset_log = valid_hierarchical_dataset(root=opt.valid_data, opt=opt)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=opt.batch_size,
        shuffle=True,
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_valid, pin_memory=True)
    log.write(valid_dataset_log)
    print('-' * 80)
    log.write('-' * 80 + '\n')
    log.close()
    
    """ model configuration """
    if 'CTC' in opt.Prediction:
        if opt.baiduCTC:
            converter = CTCLabelConverterForBaiduWarpctc(opt.character)
        else:
            converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    if opt.rgb:
        opt.input_channel = 3
    
    print("DEBUG: Initializing Model...")
    model = Model(opt)

    # weight initialization
    for name, param in model.named_parameters():
        if 'localization_fc2' in name:
            continue
        try:
            if 'bias' in name:
                init.constant_(param, 0.0)
            elif 'weight' in name:
                init.kaiming_normal_(param)
        except Exception as e:
            if 'weight' in name:
                param.data.fill_(1)
            continue

    # DataParallel 제거 (단일 GPU 안정성을 위해)
    model = model.to(device)
    model.train()
    
    if opt.saved_model != '':
        print(f'loading pretrained model from {opt.saved_model}')
        if opt.FT:
            model.load_state_dict(torch.load(opt.saved_model), strict=False)
        else:
            model.load_state_dict(torch.load(opt.saved_model))

    """ setup loss """
    if 'CTC' in opt.Prediction:
        if opt.baiduCTC:
            from warpctc_pytorch import CTCLoss 
            criterion = CTCLoss()
        else:
            criterion = torch.nn.CTCLoss(zero_infinity=True).to(device)
    else:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)
    
    loss_avg = Averager()

    filtered_parameters = []
    for p in filter(lambda p: p.requires_grad, model.parameters()):
        filtered_parameters.append(p)

    if opt.adam:
        optimizer = optim.Adam(filtered_parameters, lr=opt.lr, betas=(opt.beta1, 0.999))
    else:
        optimizer = optim.Adadelta(filtered_parameters, lr=opt.lr, rho=opt.rho, eps=opt.eps)

    """ final options """
    with open(f'./saved_models/{opt.exp_name}/opt.txt', 'a') as opt_file:
        opt_log = '------------ Options -------------\n'
        args = vars(opt)
        for k, v in args.items():
            opt_log += f'{k}: {v}\n'
        opt_log += '---------------------------------------\n'
        print(opt_log)
        opt_file.write(opt_log)

    start_iter = 0
    if opt.saved_model != '':
        try:
            start_iter = int(opt.saved_model.split('_')[-1].split('.')[0])
        except:
            pass

    start_time = time.time()
    best_accuracy = -1
    best_norm_ED = -1
    iteration = start_iter

    print("DEBUG: Starting Training Loop...")
    while(True):
        # 1. 데이터 로딩 단계
        image_tensors, labels = train_dataset.get_batch()
        
        # 2. GPU 데이터 전송 단계
        image = image_tensors.to(device)
        text, length = converter.encode(labels, batch_max_length=opt.batch_max_length)
        batch_size = image.size(0)

        # 3. 모델 연산(Forward) 단계
        if 'CTC' in opt.Prediction:
            preds = model(image, text)
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)
            if opt.baiduCTC:
                preds = preds.permute(1, 0, 2)
                cost = criterion(preds, text, preds_size, length) / batch_size
            else:
                preds = preds.log_softmax(2).permute(1, 0, 2)
                cost = criterion(preds, text, preds_size, length)
        else:
            preds = model(image, text[:, :-1])
            target = text[:, 1:]
            cost = criterion(preds.view(-1, preds.shape[-1]), target.contiguous().view(-1))

        # 4. 역전파 및 최적화 단계
        model.zero_grad()
        
        # GPU와 CPU 동기화를 강제하여 정확한 멈춤 지점을 파악하고 락을 방지합니다.
        torch.cuda.synchronize() 
        cost.backward()
        torch.cuda.synchronize()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
        optimizer.step()

        loss_avg.add(cost)

        # 로그 출력 (진행 상황 확인)
        if (iteration + 1) % 10 == 0 or iteration == 0:
            print(f"DEBUG: Iteration {iteration + 1} 완료! Loss: {cost.item():.5f} | Time: {time.time()-start_time:.1f}s")

        # validation part
        if (iteration + 1) % opt.valInterval == 0:
            print(f"DEBUG: Starting Validation at iteration {iteration+1}...")
            elapsed_time = time.time() - start_time
            with open(f'./saved_models/{opt.exp_name}/log_train.txt', 'a') as log:
                model.eval()
                with torch.no_grad():
                    valid_loss, current_accuracy, current_norm_ED, preds, confidence_score, labels, infer_time, length_of_data = validation(
                        model, criterion, valid_loader, converter, opt)
                model.train()

                loss_log = f'[{iteration+1}/{opt.num_iter}] Train loss: {loss_avg.val():0.5f}, Valid loss: {valid_loss:0.5f}, Elapsed_time: {elapsed_time:0.5f}'
                loss_avg.reset()
                current_model_log = f'{"Current_accuracy":17s}: {current_accuracy:0.3f}, {"Current_norm_ED":17s}: {current_norm_ED:0.2f}'

                if current_accuracy > best_accuracy:
                    best_accuracy = current_accuracy
                    torch.save(model.state_dict(), f'./saved_models/{opt.exp_name}/best_accuracy.pth')
                if current_norm_ED > best_norm_ED:
                    best_norm_ED = current_norm_ED
                    torch.save(model.state_dict(), f'./saved_models/{opt.exp_name}/best_norm_ED.pth')
                
                best_model_log = f'{"Best_accuracy":17s}: {best_accuracy:0.3f}, {"Best_norm_ED":17s}: {best_norm_ED:0.2f}'
                loss_model_log = f'{loss_log}\n{current_model_log}\n{best_model_log}'
                print(loss_model_log)
                log.write(loss_model_log + '\n')

        if (iteration + 1) == opt.num_iter:
            print('end the training')
            sys.exit()
        iteration += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # ... (기존 parser 설정 동일)
    parser.add_argument('--exp_name', help='Where to store logs and models')
    parser.add_argument('--train_data', required=True, help='path to training dataset')
    parser.add_argument('--valid_data', required=True, help='path to validation dataset')
    parser.add_argument('--manualSeed', type=int, default=1111, help='for random seed setting')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batch_size', type=int, default=192, help='input batch size')
    parser.add_argument('--num_iter', type=int, default=10000, help='number of iterations to train for')
    parser.add_argument('--valInterval', type=int, default=100, help='Interval between each validation')
    parser.add_argument('--saved_model', default='', help="path to model to continue training")
    parser.add_argument('--FT', action='store_true', help='whether to do fine-tuning')
    parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is Adadelta)')
    parser.add_argument('--lr', type=float, default=1, help='learning rate, default=1.0 for Adadelta')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.9')
    parser.add_argument('--rho', type=float, default=0.95, help='decay rate rho for Adadelta. default=0.95')
    parser.add_argument('--eps', type=float, default=1e-8, help='eps for Adadelta. default=1e-8')
    parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping value. default=5')
    parser.add_argument('--baiduCTC', action='store_true', help='for data_filtering_off mode')
    parser.add_argument('--select_data', type=str, default='/')
    parser.add_argument('--batch_ratio', type=str, default='1')
    parser.add_argument('--total_data_usage_ratio', type=str, default='1.0')
    parser.add_argument('--batch_max_length', type=int, default=25)
    parser.add_argument('--imgH', type=int, default=32)
    parser.add_argument('--imgW', type=int, default=100)
    parser.add_argument('--rgb', action='store_true')
    parser.add_argument('--character', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz')
    parser.add_argument('--sensitive', action='store_true')
    parser.add_argument('--PAD', action='store_true')
    parser.add_argument('--data_filtering_off', action='store_true')
    parser.add_argument('--Transformation', type=str, required=True)
    parser.add_argument('--FeatureExtraction', type=str, required=True)
    parser.add_argument('--SequenceModeling', type=str, required=True)
    parser.add_argument('--Prediction', type=str, required=True)
    parser.add_argument('--num_fiducial', type=int, default=20)
    parser.add_argument('--input_channel', type=int, default=1)
    parser.add_argument('--output_channel', type=int, default=512)
    parser.add_argument('--hidden_size', type=int, default=256)

    opt = parser.parse_args()
    os.makedirs(f'./saved_models/{opt.exp_name}', exist_ok=True)
    opt.character = "0123456789가강거경계고관광구금기김나남너노누다대더도동두등라러로루리마머명모무문미바배뱌버보부북사산서소수아악안양어연영오용우울원육이인자작저전조주중지차천초추충카타파평포하허호홀히"
    
    random.seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    torch.cuda.manual_seed(opt.manualSeed)

    train(opt)