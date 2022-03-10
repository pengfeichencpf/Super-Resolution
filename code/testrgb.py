# -*- coding:utf-8 -*-
import os,codecs,pyssim,math,torch
import numpy as np
from skimage import transform, io, color

def savelog(path,psnr,ssim,log_path='./log/'):
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    log=codecs.open(log_path+'test_log'+'.txt','a+','utf-8')
    log.writelines("=======================================\n")
    log.writelines(path+'\n'+'PSNR==>%f  \n'%psnr+'SSIM==>%f  \n'%ssim)
    log.close()

def data_trans(im,num):
    org_image = im
    if num ==0:    
        ud_image = np.flipud(org_image)
        tranform = ud_image
    elif num ==1:      
        lr_image = np.fliplr(org_image)
        tranform = lr_image
    elif num ==2:
        lr_image = np.fliplr(org_image)
        lrud_image = np.flipud(lr_image)        
        tranform = lrud_image
    elif num ==3:
        rotated_image1 = np.rot90(org_image)        
        tranform = rotated_image1
    elif num ==4: 
        rotated_image2 = np.rot90(org_image, -1)
        tranform = rotated_image2
    elif num ==5: 
        rotated_image1 = np.rot90(org_image) 
        ud_image1 = np.flipud(rotated_image1)
        tranform = ud_image1
    elif num ==6:        
        rotated_image2 = np.rot90(org_image, -1)
        ud_image2 = np.flipud(rotated_image2)
        tranform = ud_image2
    else:
        tranform = org_image
    return tranform

def data_trans_inv(im,num):
    org_image = im
    if num ==0:    
        ud_image = np.flipud(org_image)
        tranform = ud_image
    elif num ==1:      
        lr_image = np.fliplr(org_image)
        tranform = lr_image
    elif num ==2:
        lr_image = np.fliplr(org_image)
        lrud_image = np.flipud(lr_image)        
        tranform = lrud_image
    elif num ==3:
        rotated_image1 = np.rot90(org_image,-1)        
        tranform = rotated_image1
    elif num ==4: 
        rotated_image2 = np.rot90(org_image)
        tranform = rotated_image2
    elif num ==5: 
        rotated_image1 = np.rot90(org_image) 
        ud_image1 = np.flipud(rotated_image1)
        tranform = ud_image1
    elif num ==6:        
        rotated_image2 = np.rot90(org_image, -1)
        ud_image2 = np.flipud(rotated_image2)
        tranform = ud_image2
    else:
        tranform = org_image
    return tranform

def evalrgb(testdir='all',model=None,model_path=None,scale=4):
    if testdir == 'all':
        testdirs=["./data/sr/Set5","./data/sr/Set14","./data/sr/bsd100","./data/sr/Urban100"]
        for t in testdirs:
            avg_psnr,avg_ssim = evaluate_by_path(os.path.join('./data',t),model,model_path,scale)
            savelog(t,avg_psnr,avg_ssim)
    else:
        avg_psnr,avg_ssim = evaluate_by_path(os.path.join('./data',testdir),model,model_path,scale)
        return avg_psnr,avg_ssim

def evaluate_by_path(path,model,model_path,scale):
    pimages=[os.listdir(path)[i] for i in range(len(os.listdir(path))) if os.listdir(path)[i][-4:]=='.bmp']
    s_psnr=0
    s_ssim=0
    for pimg in pimages:
        img = io.imread(path+'/'+pimg)
        im_list = []
        # for i in range(8):
        #     tmp = data_trans(img,i)
        #     seim1=predict(tmp,model,model_path,scale)[0,:,:]
        #     seim2=data_trans_inv(seim1,i)
        #     im_list.append(seim2)
        # # print(np.array(im_list).shape)
        # avg = np.mean(np.array(im_list),axis=0)
        avg = predict(img,model,model_path,scale)
        psnr,ssim = eva_ensemble(avg,img,pimg,scale)
        s_psnr+=psnr
        s_ssim+=ssim
    avg_psnr=s_psnr/len(pimages)
    avg_ssim=s_ssim/len(pimages)
    print("Scale=%d || PSNR=%.3f || SSIM=%.4f"%(scale,avg_psnr,avg_ssim))
    return avg_psnr,avg_ssim

def predict(img_read,model,model_path,scale):
    im_gt_y = img_read
    img_y = resize_image(im_gt_y, 1.0 / scale)
    h,w,c = img_y.shape
    img_y= img_y.transpose(2,0,1)
    im_input = torch.from_numpy( img_y/ 255).view(1, c, h,w).float()
    if model is None:
        if model_path is not None:
            model = torch.load(model_path)
        else:
            raise ValueError("model is None")
    model = model.cuda()
    im_input = im_input.cuda()
    with torch.no_grad():
        GHR = model(im_input)
    im_h_y = GHR.detach().cpu().numpy().astype(np.float32)
    im_h_y = im_h_y * 255.
    im_h_y[im_h_y < 0] = 0
    im_h_y[im_h_y > 255.] = 255.
    im_h_y = im_h_y[0, :, :]
    return im_h_y.transpose(1, 2, 0)

def eva_ensemble(im_h,img_read,name,scale):
    im_gt_y = convert_rgb_to_y(img_read)
    im_h_y = convert_rgb_to_y(im_h)
    save_figure(im_h, name)
    psnr_predicted = PSNR(im_gt_y, im_h_y, shave_border=scale)
    ssim_predicted = pyssim.compute_ssim(im_gt_y, im_h_y)
    return psnr_predicted, ssim_predicted

def save_figure(img,name,out_path='./save_img/'):
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    img = img*255.
    io.imsave(out_path+name[:-4]+'.png',img.astype(np.uint8))

def PSNR(pred, gt, shave_border=0):
    height, width = pred.shape[:2]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)

def convert_rgb_to_y(image):
    return color.rgb2ycbcr(image)[:, :, 0]

def resize_image(image, scale):
    width, height = image.shape[1], image.shape[0]
    new_width = int(width * scale)
    new_height = int(height * scale)
    return transform.resize(image, (new_height, new_width), order=3)     

if __name__ == '__main__':
    eval()