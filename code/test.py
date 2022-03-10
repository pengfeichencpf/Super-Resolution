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

def eval(testdir='all',model=None,model_path=None,scale=4):
    if testdir == 'all':
        testdirs=["sr/Set5","sr/Set14","sr/BSD100","sr/Urban100"]
        # testdirs=["sr/Urban100"]
        for t in testdirs:
            avg_psnr,avg_ssim = evaluate_by_path(os.path.join('./data',t),model,model_path,scale)
            savelog(t,avg_psnr,avg_ssim)
    else:
        avg_psnr,avg_ssim = evaluate_by_path(os.path.join('./data',testdir),model,model_path,scale)
        return avg_psnr,avg_ssim

def split_image(im):
    h,w = im.shape
    im1 = im[0:h//2,:]
    im2 = im[h//2:h,:]
    return [im1,im2]

def merge_image(im_raw,im1,im2):
    h,w = im_raw.shape[0],im_raw.shape[1]
    im = np.zeros((h,w),dtype=np.uint8)
    im[0:h//2,:] = im1
    im[h//2:h,:] = im2
    return im

def check_size(im,scale):
    h,w,c = im.shape
    if h%scale !=0 or w%scale !=0:
        im = im[0:h-h%scale,0:w-w%scale,:]
    return im

def evaluate_by_path(path,model,model_path,scale):
    pimages=[os.listdir(path)[i] for i in range(len(os.listdir(path))) if os.listdir(path)[i][-4:]=='.bmp']
    s_psnr=0
    s_ssim=0
    for pimg in pimages:
        img = io.imread(path+'/'+pimg)
        if path == './data/sr/Urban100':
            img = check_size(img,scale=8)
            split_images = split_image(convert_rgb_to_y(img))
            avg_list = []
            for sim in split_images:
                im_list = []
                for i in range(8):
                    tmp = data_trans(sim,i)
                    seim1=predict(tmp,model,model_path,scale,convert_gt=False)[0,:,:]
                    seim2=data_trans_inv(seim1,i)
                    im_list.append(seim2)
                # print(np.array(im_list).shape)
                avg_split = np.mean(np.array(im_list),axis=0)
                avg_list.append(avg_split)
            avg = merge_image(img,avg_list[0],avg_list[1])
        else:
            im_list = []
            for i in range(8):
                tmp = data_trans(img,i)
                seim1=predict(tmp,model,model_path,scale)[0,:,:]
                seim2=data_trans_inv(seim1,i)
                im_list.append(seim2)
            # print(np.array(im_list).shape)
            avg = np.mean(np.array(im_list),axis=0)
        # avg = predict(img,model,model_path,scale)[0,:,:]
        psnr,ssim = eva_ensemble(avg,img,pimg,scale)
        s_psnr+=psnr
        s_ssim+=ssim
    avg_psnr=s_psnr/len(pimages)
    avg_ssim=s_ssim/len(pimages)
    print("Scale=%d || PSNR=%.3f || SSIM=%.4f"%(scale,avg_psnr,avg_ssim))
    return avg_psnr,avg_ssim

def predict(img_read,model,model_path,scale,convert_gt = True):
    if convert_gt:
        im_gt_y = convert_rgb_to_y(img_read).astype("float32")
    else:
        im_gt_y = img_read.astype("float32")
    img_y = resize_image(im_gt_y, 1.0 / scale)
    im_input = torch.from_numpy(img_y / 255).view(1, -1, img_y.shape[0], img_y.shape[1])
    if model is None:
        if model_path is not None:
            model = torch.load(model_path)['model']
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
    return im_h_y

def eva_ensemble(im_h_y,img_read,name,scale):
    im_gt_y = convert_rgb_to_y(img_read).astype("float32")
    gt_yuv = convert_rgb_to_ycbcr(img_read)
    recon = convert_y_and_cbcr_to_rgb(im_h_y, gt_yuv[:, :, 1:3])
    save_figure(recon, name)
    psnr_predicted = PSNR(im_gt_y, im_h_y, shave_border=scale)
    ssim_predicted = pyssim.compute_ssim(im_gt_y, im_h_y)
    return psnr_predicted, ssim_predicted

def save_figure(img,name,out_path='./save_img/'):
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    io.imsave(out_path+name[:-4]+'.png',img)

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

def convert_rgb_to_ycbcr(image):
    return color.rgb2ycbcr(image)

def convert_y_and_cbcr_to_rgb(y_image, cbcr_image):
    ycbcr_image = np.zeros([y_image.shape[0], y_image.shape[1], 3])
    ycbcr_image[:, :, 0] = y_image
    ycbcr_image[:, :, 1:3] = cbcr_image[:, :, 0:2]
    return convert_ycbcr_to_rgb(ycbcr_image)

def convert_ycbcr_to_rgb(ycbcr_image):
    return color.ycbcr2rgb(ycbcr_image)

def resize_image(image, scale):
    width, height = image.shape[1], image.shape[0]
    new_width = int(width * scale)
    new_height = int(height * scale)
    return transform.resize(image, (new_height, new_width), order=3)     

if __name__ == '__main__':
    eval(testdir='all',model=None,model_path='./checkpoints/model_best_du.pth',scale=4)