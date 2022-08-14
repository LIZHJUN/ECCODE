import argparse
import  time
import cv2,os
import numpy as np
import tensorflow as tf
import neuralgym as ng
from glob import glob
from inpaint_model_gc import InpaintGCModel
from inpaint_ops import random_bbox, bbox2mask, free_form_mask_tf

parser = argparse.ArgumentParser()


parser.add_argument('--dataset_dir', default=r'H:\quantitive\MC\comp\place2-free-CA+MC+ours2-2030', type=str,                                                                    #待修复的图像文件名
                    help='The filename of image to be completed.')
parser.add_argument('--maskset_dir', default=r'H:\quantitive\MC\ori\place2-free-CA+MC+ours2-2030', type=str,                                                                    #待修复的图像文件名
                    help='The filename of image to be completed.')



if __name__ == "__main__":
    # ng.get_gpus(-1)
    args = parser.parse_args()
    file_list_image = glob('{}/*.*'.format(args.dataset_dir))
    file_list_image2 = glob('{}/*.*'.format(args.maskset_dir))  # 掩码列表
    zipped = zip(file_list_image, file_list_image2)

    total_loss = [ ]
    for zip in zipped:
        file_name_im = os.path.basename(zip[0]).split('.')[0]
        image = cv2.imread(zip[0])
        image = image/127.5 - 1.
        print('name1:',file_name_im)
        print('Shape of image: {}'.format(image.shape))  # 打印图像尺寸


        image2 = cv2.imread(zip[1])
        image2 = image2/127.5 - 1.
        file_name_im2 = os.path.basename(zip[1]).split('.')[0]
        print('name2:', file_name_im2)
        print('Shape of image2: {}'.format(image2.shape))  # 打印图像尺寸
        loss = np.mean(np.abs(image-image2))
        print('loss:',loss)


        total_loss.append(loss)
        print('total_loss',total_loss)

    loss_result = np.mean(total_loss)
    print('loss result:',loss_result)




