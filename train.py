import os,logging
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.utils as utils
from torch.autograd import Variable, Function
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from models import *
from dataset import prepare_data, Dataset
from utils import *
from skimage.feature import local_binary_pattern
from torch.nn import init

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description="DnCNN")
parser.add_argument("--preprocess", type=bool, default=True, help='run prepare_data or not')
parser.add_argument("--batchSize", type=int, default=16, help="Training batch size")
parser.add_argument("--num_of_layers", type=int, default=17, help="Number of total layers")
parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
parser.add_argument("--train_dir", type=str, default="/home/hgq/LDCT/RED-LDCT-KERAS/data/quarter_abdomen.npy",  help="path of train data")
parser.add_argument("--label_data", type=str, default='/home/hgq/LDCT/RED-LDCT-KERAS/data/full_abdomen.npy', help='path of train data')
parser.add_argument("--milestone", type=int, default=30, help="When to decay learning rate; should be less than epochs")
parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate")
parser.add_argument("--outf", type=str, default="logs/abdomen04", help='path of log files')
parser.add_argument("--mode", type=str, default="B", help='with known noise level (S) or blind training (B)')
parser.add_argument("--noiseL", type=float, default=25, help='noise level; ignored when mode=B')
parser.add_argument("--val_noiseL", type=float, default=25, help='noise level used on validation set')
opt = parser.parse_args()

Tensor = torch.cuda.FloatTensor
#mod = torch.load('./logs/DnCNN-S-25/net16.pth')
def toZeroThreshold(x, t=0.1):
	  zeros = Tensor(x.shape).fill_(0.0)
	  return torch.where(x > t, x, zeros)

gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def sigmoid(x, a):
    y = 1.0 / (1.0 + torch.pow(255, (a - x) * 255))
    return y
    
def weights_init(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv3d') != -1:
        init.xavier_normal_(m.weight.data)
        init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        init.xavier_normal_(m.weight.data)
        init.constant_(m.bias.data, 0.0)

def load_train_data():
    logging.info('loading train data...')
    # file_list = glob.glob('{}/*.npy'.format(args.trdata_dir))
    # for file in file_list:
    data = np.load(opt.train_dir)
    # return data
    logging.info('Size of train data: ({}, {}, {})'.format(data.shape[0], data.shape[1], data.shape[2]))
    return data


def load_label_data():
    logging.info('loading label data...')
    # file_list = glob.glob('{}/*.npy'.format(args.tedata_dir))
    # for file in file_list:
    data = np.load(opt.label_data)
    # return data
    logging.info('Size of label data: ({}, {}, {})'.format(data.shape[0], data.shape[1], data.shape[2]))
    return data

def main():
    # Load dataset
    print('Loading dataset ...\n')
    train_data = load_train_data()
    train_data = train_data.reshape((train_data.shape[0], 1, train_data.shape[1], train_data.shape[2]))
    #print(train_data.shape)
    train_data = train_data.astype('float32') / 255.0
    # train_data = train_datagen(train_data, batch_size=args.batch_size)
    label_data = load_label_data()
    label_data = label_data.reshape((label_data.shape[0], 1, label_data.shape[1], label_data.shape[2]))
    label_data = label_data.astype('float32') / 255.0
    # Build model
    n_feats = 64
    n_colors = 1
    n_resgroups = 17
    n_resblocks = 17
    reduction = 16
    res_scale = 1
    net = DnCNN(n_resgroups, n_resblocks, n_feats, reduction, n_colors , res_scale, conv=default_conv, device=gpu)  
    #net.load_state_dict(torch.load('./logs/abdomen0121/net4.pth'),strict = False)
    net.apply(weights_init)
    #net.apply(weight)
    criterion = nn.MSELoss(size_average=False)
    #criter = lbploss()
    # Move to GPU
    device_ids = [0]
    model = nn.DataParallel(net, device_ids=device_ids).cuda()
    criterion.cuda()
    #criter.cuda()
    # Optimizer
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr)
    # training
    writer = SummaryWriter(opt.outf)
    step = 0
    noiseL_B = [0, 55]  # ingnored when opt.mode=='S'
    
    total_loss2 = 0
    for epoch in range(opt.epochs):
        if epoch < opt.milestone:
            current_lr = opt.lr
        else:
            current_lr = opt.lr / 10.
        # set learning rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
        print('learning rate %f' % current_lr)
        # train
        batch_size = opt.batchSize
        permutated_indexes= np.random.permutation(train_data.shape[0])
        total_loss1 = 0
        for index in range(int(train_data.shape[0]/ batch_size)):
            batch_indexes= permutated_indexes[index*batch_size:(index+1)*batch_size]
            train_batch= train_data[batch_indexes]
            # training step
            model.train()
            model.zero_grad()
            optimizer.zero_grad()
            imgn_train = train_batch
            imgn_train = torch.from_numpy(imgn_train)
            
            img_train = label_data[batch_indexes]
            img_train = torch.from_numpy(img_train)
            img_train, imgn_train = Variable(img_train.cuda()), Variable(imgn_train.cuda())
            #print(imgn_train.shape)
            out_train = model(imgn_train)
            out_train = imgn_train - out_train          
            loss1 = (criterion(out_train, img_train)) / (imgn_train.size()[0] * 2)
            loss1.backward()
           
            #loss2 = 1000*(criter(out_train, img_train)) / (imgn_train.size()[0] * 2)
            #loss = loss1 + loss2
            #loss2.backward
            
            optimizer.step()
            # results
            model.eval()
            out_train = torch.clamp(imgn_train - model(imgn_train), 0., 1.)
            psnr_train = batch_PSNR(out_train, img_train)  # , 1.)
            print("[epoch %d][%d/%d] loss: %.4f PSNR_train: %.4f" %
                  (epoch + 1, index + 1, int(train_data.shape[0]/ batch_size), loss1.item(), psnr_train))
            # if you are using older version of PyTorch, you may need to change loss.item() to loss.data[0]
            if step % 10 == 0:
                # Log the scalar values
                writer.add_scalar('loss', loss1.item(), step)
                writer.add_scalar('PSNR on training data', psnr_train, step)
            step += 1
            del loss1
            del out_train
            del psnr_train
            torch.cuda.empty_cache()
        ## the end of each epoch
        model.eval()
        # validate
        """
        psnr_val = 0
        for k in range(len(dataset_val)):
            img_val = torch.unsqueeze(dataset_val[k], 0)
            noise = torch.FloatTensor(img_val.size()).normal_(mean=0, std=opt.val_noiseL / 255.)
            imgn_val = img_val + noise
            img_val, imgn_val = Variable(img_val.cuda()), Variable(imgn_val.cuda())
            imgn_val = imgn_val - model(imgn_val)
            out_val = torch.clamp(imgn_val, 0., 1.)
            psnr_val += batch_PSNR(out_val, img_val)  # , 1.)
            #del out_vall
            del out_val
        psnr_val /= len(dataset_val)

        print("\n[epoch %d] PSNR_val: %.4f" % (epoch + 1, psnr_val))
        """
        
        #writer.add_scalar('PSNR on validation data', psnr_val, epoch)
        # log the images
        out_train = torch.clamp(imgn_train - model(imgn_train), 0., 1.)
        Img = utils.make_grid(img_train.data, nrow=8, normalize=True, scale_each=True)
        Imgn = utils.make_grid(imgn_train.data, nrow=8, normalize=True, scale_each=True)
        Irecon = utils.make_grid(out_train.data, nrow=8, normalize=True, scale_each=True)
        writer.add_image('clean image', Img, epoch)
        writer.add_image('noisy image', Imgn, epoch)
        writer.add_image('reconstructed image', Irecon, epoch)
        # save model
        torch.save(model.state_dict(), os.path.join(opt.outf, 'net%d.pth'%epoch))
        torch.cuda.empty_cache()
    del criterion,img_train,imgn_train
        



if __name__ == "__main__":
    if opt.preprocess:
        if opt.mode == 'S':
            prepare_data(data_path='data', patch_size=40, stride=10, aug_times=1)
        if opt.mode == 'B':
            prepare_data(data_path='data', patch_size=50, stride=10, aug_times=1)
    main()
