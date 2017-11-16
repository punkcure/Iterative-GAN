"""
Some codes from https://github.com/Newmu/dcgan_code
"""
from __future__ import division
import math
import json
import random
import pprint
import scipy.misc
import numpy as np
import os
import time
from time import gmtime, strftime
from six.moves import xrange
from functools import reduce

import tensorflow as tf
import tensorflow.contrib.slim as slim

def sigmoid_cross_entropy_with_logits(x, y):
  try:
    return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
  except:
    return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, targets=y)

pp = pprint.PrettyPrinter()

get_stddev = lambda x, k_h, k_w: 1/math.sqrt(k_w*k_h*x.get_shape()[-1])

def conv_to_list(str_types,is_int=True):
  y_type_list_str = str_types.split(',')
  y_type_list = list()
  for i in range(len(y_type_list_str)):
    if is_int:
      y_type_list.append(int(y_type_list_str[i]))
    else:
      y_type_list.append(y_type_list_str[i])
   
  return y_type_list

def conv_to_list_2(str_type):
  y_type_list_str = str_type.split(';')
  y_type_list = list()
  for i in range(len(y_type_list_str)):
    y_label_list=list()
    y_label_str = y_type_list_str[i].split(',')
    for j in range(len(y_label_str)):
      y_label_list.append(int(y_label_str[j]))
    y_type_list.append(y_label_list)

  return  y_type_list

def concat(z,y):
  return tf.concat([z,y],1)

def conv_concat(x,y,y_dim):
  xshape = tf.shape(x)
  batch_num = tf.shape(x)[0]
  height = tf.shape(x)[1]
  width = tf.shape(x)[2]
  y = tf.reshape(y,[batch_num,1,1,y_dim])
  y = y*tf.ones([batch_num,height,width,y_dim])
  return tf.concat([x,y],3)

def sample_y_by_y_dim(sample_size,y_dim,type_list):
  y = np.zeros([sample_size,y_dim])
  index = 0
  if y_dim != 0:
    for i in range(sample_size):
      #index = index % len(type_list)
      #y[i,type_list[index]] = 1
      j = index % y_dim
      y[i][j]=1
      index += 1

  return y

def get_y_by_attribute(file_names,dic_attributes,y_type_list,y_dim):
  y_out = np.zeros([len(file_names),y_dim])
  if y_dim != 0:
    i = 0
    for file in file_names:
      file_string_array = file.split('\\')
      file_name = file_string_array[-1]
      if file_name in dic_attributes:
        for j in range(y_dim):
          y_out[i][j] = dic_attributes[file_name][y_type_list[j]]
        i = i+1
      else:
        print("**************************************Error:file attribute not found!!!!!!!!!!!!!!!!!!!!!!")

  return y_out

def show_all_variables():
  model_vars = tf.trainable_variables()
  slim.model_analyzer.analyze_vars(model_vars, print_info=True)

def getAttributeOfImageFromTxt(txtPath):
  dictionary = {}
  f = open(txtPath,"r")  
  #lines = f.readlines()
  i = 0 
  for line in f :
    i = i + 1
    listValue = []

    if(i >= 3):
      linelist =line.split()
      for attribute in linelist:
        if attribute == "-1" :
          listValue.append(float(0.0))
        elif attribute == "1":
          listValue.append(float(1.0))
      dictionary[linelist[0]] = listValue
      
  f.close()
    
  return dictionary


def FilterbyAttribute(sample_files,ImageFilter,ImageAttributeDic):
    delList = []
    for sample_file in sample_files:
      splitstr = sample_file.split('\\')
      if(splitstr[-1] in ImageAttributeDic):
        i = 0 
        for attributeRequired in ImageFilter:
          attributeOfImage = ImageAttributeDic[splitstr[-1]][i]
          i = i + 1
          if(attributeRequired == '1' and attributeOfImage == "0"):
            delList.append(sample_file)
            break

    for delItem in delList:
      sample_files.remove(delItem)

    return sample_files
    
def get_image(image_path, input_height, input_width,
              resize_height=64, resize_width=64,
              crop=True, grayscale=False,recordName = False):
  image = imread(image_path, grayscale)
  #if recordName == True:
   # output = open("./logs/data.txt",'w+')
    #output.write(image_path)
    #output.close()
  
  return transform(image, input_height, input_width,
                   resize_height, resize_width, crop)

def save_images(images, size, image_path):
  return imsave(inverse_transform(images), size, image_path)

def imread(path, grayscale = False):
  if (grayscale):
    return scipy.misc.imread(path, flatten = True).astype(np.float)
  else:
    return scipy.misc.imread(path).astype(np.float)

def merge_images(images, size):
  return inverse_transform(images)

def merge(images, size):
  h, w = images.shape[1], images.shape[2]
  if (images.shape[3] in (3,4)):
    c = images.shape[3]
    img = np.zeros((h * size[0], w * size[1], c))
    for idx, image in enumerate(images):
      i = idx % size[1]
      j = idx // size[1]
      img[j * h:j * h + h, i * w:i * w + w, :] = image
    return img
  elif images.shape[3]==1:
    img = np.zeros((h * size[0], w * size[1]))
    for idx, image in enumerate(images):
      i = idx % size[1]
      j = idx // size[1]
      img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
    return img
  else:
    raise ValueError('in merge(images,size) images parameter '
                     'must have dimensions: HxW or HxWx3 or HxWx4')

def imsave(images, size, path):
  image = np.squeeze(merge(images, size))
  return scipy.misc.imsave(path, image)

def center_crop(x, crop_h, crop_w,
                resize_h=64, resize_w=64):
  if crop_w is None:
    crop_w = crop_h
  h, w = x.shape[:2]
  j = int(round((h - crop_h)/2.))
  i = int(round((w - crop_w)/2.))
  return scipy.misc.imresize(
      x[j:j+crop_h, i:i+crop_w], [resize_h, resize_w])

def transform(image, input_height, input_width, 
              resize_height=64, resize_width=64, crop=True):
  if crop:
    cropped_image = center_crop(
      image, input_height, input_width, 
      resize_height, resize_width)
  else:
    cropped_image = scipy.misc.imresize(image, [resize_height, resize_width])
  return np.array(cropped_image)/127.5 - 1.

def inverse_transform(images):
  return (images+1.)/2.

def to_json(output_path, *layers):
  with open(output_path, "w") as layer_f:
    lines = ""
    for w, b, bn in layers:
      layer_idx = w.name.split('/')[0].split('h')[1]

      B = b.eval()

      if "lin/" in w.name:
        W = w.eval()
        depth = W.shape[1]
      else:
        W = np.rollaxis(w.eval(), 2, 0)
        depth = W.shape[0]

      biases = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(B)]}
      if bn != None:
        gamma = bn.gamma.eval()
        beta = bn.beta.eval()

        gamma = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(gamma)]}
        beta = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(beta)]}
      else:
        gamma = {"sy": 1, "sx": 1, "depth": 0, "w": []}
        beta = {"sy": 1, "sx": 1, "depth": 0, "w": []}

      if "lin/" in w.name:
        fs = []
        for w in W.T:
          fs.append({"sy": 1, "sx": 1, "depth": W.shape[0], "w": ['%.2f' % elem for elem in list(w)]})

        lines += """
          var layer_%s = {
            "layer_type": "fc", 
            "sy": 1, "sx": 1, 
            "out_sx": 1, "out_sy": 1,
            "stride": 1, "pad": 0,
            "out_depth": %s, "in_depth": %s,
            "biases": %s,
            "gamma": %s,
            "beta": %s,
            "filters": %s
          };""" % (layer_idx.split('_')[0], W.shape[1], W.shape[0], biases, gamma, beta, fs)
      else:
        fs = []
        for w_ in W:
          fs.append({"sy": 5, "sx": 5, "depth": W.shape[3], "w": ['%.2f' % elem for elem in list(w_.flatten())]})

        lines += """
          var layer_%s = {
            "layer_type": "deconv", 
            "sy": 5, "sx": 5,
            "out_sx": %s, "out_sy": %s,
            "stride": 2, "pad": 1,
            "out_depth": %s, "in_depth": %s,
            "biases": %s,
            "gamma": %s,
            "beta": %s,
            "filters": %s
          };""" % (layer_idx, 2**(int(layer_idx)+2), 2**(int(layer_idx)+2),
               W.shape[0], W.shape[3], biases, gamma, beta, fs)
    layer_f.write(" ".join(lines.replace("'","").split()))

def make_gif(images, fname, duration=2, true_image=False):
  import moviepy.editor as mpy

  def make_frame(t):
    try:
      x = images[int(len(images)/duration*t)]
    except:
      x = images[-1]

    if true_image:
      return x.astype(np.uint8)
    else:
      return ((x+1)/2*255).astype(np.uint8)

  clip = mpy.VideoClip(make_frame, duration=duration)
  clip.write_gif(fname, fps = len(images) / duration)

def mul(a, b):
    "Same as a * b."
    return a * b

def _tensor_size(tensor):
    return reduce(mul, (d.value for d in tensor.get_shape()), 1)

#important 计算list_feature_map1和list_feature_map2所记录的特征图的差异
def feature_loss(list_feature_map1, list_feature_map2, name='feature_loss'):
  
  #记录总的loss
  loss = 0
  for i in range(len(list_feature_map1)):
    #dis = 2*tf.square(tf.subtract(x[i],mu[i]))
    #计算每一层的特征图之l1距离。l2距离被注释
    dis = abs(list_feature_map1[i]-list_feature_map2[i])
    #按论文要求把距离除以特征图的大小
    dis_over_size = tf.div(dis,_tensor_size(list_feature_map1[i]))
    #逐层求和得总的loss
    loss += tf.reduce_mean(dis_over_size) 

  return loss

def visualize(sess, dcgan, config, option):
  
  image_frame_dim = int(math.ceil(config.batch_size**.5))
  current_time = time.strftime('%Y-%m-%d_%H_%M_%S',time.localtime(time.time()))
  if not os.path.exists(config.sample_dir+"/"+current_time):
    os.makedirs(config.sample_dir+"/"+current_time)
  if option == 0:
    #从随机z中生成图片
    z_sample = np.random.uniform(-0.5, 0.5, size=(config.batch_size, dcgan.z_dim))
    samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
    save_images(samples, [image_frame_dim, image_frame_dim], './samples/test_%s.png' % strftime("%Y%m%d%H%M%S", gmtime()))
  elif option == 1:
    #随机抽取一批真实图片，重构，并保存文件名
    sample_files = dcgan.data[0:int(config.sample_num)]
    length = len(dcgan.data)
    for i in range(10):  
      print("[*]%d"%i)
      record_file_name = './samples/'+current_time+'/sample_file_name_'+str(i)+'.txt'
      output = open(record_file_name,'w+')
      for j in range(config.sample_num):
        sample_files[j] = dcgan.data[np.random.randint(0,length)-1]
        output.write(str(j) +'  :  ' +sample_files[j] + "\r\n")
      output.close()

      sample_images = [get_image(sample_file,
                      input_height=config.input_height,
                      input_width=config.input_width,
                      resize_height=config.output_height,
                      resize_width=config.output_width,
                      crop=config.crop,
                      grayscale=dcgan.grayscale,recordName = True) for sample_file in sample_files]
      sample_images = np.array(sample_images).astype(np.float32)
      save_images(sample_images, image_manifold_size(config.sample_num),'./samples/%s/sample_images_%d.png'%(current_time,i))

      c_Copy,z_Copy = sess.run([dcgan.cat_Sample,dcgan.z_Sample],feed_dict = {dcgan.inputs:sample_images})
      sample_real_copy_image = sess.run([dcgan.sampler_data],feed_dict={dcgan.z:z_Copy,dcgan.y:c_Copy})
      if isinstance(sample_real_copy_image,list):
        sample_real_copy_image = np.array(sample_real_copy_image)
        sample_real_copy_image = np.reshape(sample_real_copy_image,[config.sample_num,config.output_height,config.output_width,dcgan.c_dim])
      save_images(sample_real_copy_image,image_manifold_size(config.sample_num),'./samples/%s/sample_images_copy_%d.png'%(current_time,i))

  elif option == 2:
    #输入n张指定图片 重构指定图片
    test_files = conv_to_list(config.Test_file_names,is_int=False) 
    sample_files = dcgan.data[0:int(config.sample_num)]
    for j in range(config.sample_num):
      if j < len(test_files):
        sample_files[j] =  config.DatasetDir + '/' + test_files[j]
      else:
        sample_files[j] = dcgan.data[np.random.randint(0,len(dcgan.data))-1] # 
    sample_images = [get_image(sample_file,
                    input_height=config.input_height,
                    input_width=config.input_width,
                    resize_height=config.output_height,
                    resize_width=config.output_width,
                    crop=config.crop,
                    grayscale=dcgan.grayscale,recordName = True) for sample_file in sample_files]
    sample_images = np.array(sample_images).astype(np.float32)
    save_images(sample_images,image_manifold_size(config.sample_num),'./samples/%s/sample_images.png'%(current_time))
      
    c_Copy,z_Copy = sess.run([dcgan.cat_Sample,dcgan.z_Sample],feed_dict = {dcgan.inputs:sample_images})
    sample_real_copy_image = sess.run([dcgan.sampler_data],feed_dict={dcgan.z:z_Copy,dcgan.y:c_Copy})
    if isinstance(sample_real_copy_image,list):
      sample_real_copy_image = np.array(sample_real_copy_image)
      sample_real_copy_image = np.reshape(sample_real_copy_image,[config.sample_num,config.output_height,config.output_width,dcgan.c_dim])
    save_images(sample_real_copy_image,image_manifold_size(config.sample_num),'./samples/%s/sample_images_copy.png'%(current_time))
    
  elif option == 3:
    #输入n张指定图片 先按指定的属性label来重构图片 再重构指定图片 
    test_files = conv_to_list(config.Test_file_names,is_int=False) 
    label_list = conv_to_list_2(config.Test_file_label)
    label_len = len(label_list)
    for i in range(len(test_files)):
      print(" [*] %d" % i)
      sample_files = dcgan.data[0:int(config.sample_num)]
      for j in range(config.sample_num):
        if j < label_len+1:
          sample_files[j] = config.DatasetDir + '/' + test_files[i]
        else:
          sample_files[j] = dcgan.data[np.random.randint(0,len(dcgan.data))-1]
      sample_images = [get_image(sample_file,
                      input_height=config.input_height,
                      input_width=config.input_width,
                      resize_height=config.output_height,
                      resize_width=config.output_width,
                      crop=config.crop,
                      grayscale=dcgan.grayscale,recordName = True) for sample_file in sample_files]
      sample_images = np.array(sample_images).astype(np.float32)
      c_Copy,z_Copy = sess.run([dcgan.cat_Sample,dcgan.z_Sample],feed_dict = {dcgan.inputs:sample_images})
      for k in xrange(1,len(label_list)):
        for l in range(len(dcgan.y_type_list)):
          c_Copy[k][l] = label_list[k][l]
      sample_real_copy_image = sess.run([dcgan.sampler_data],feed_dict={dcgan.z:z_Copy,dcgan.y:c_Copy})
      if isinstance(sample_real_copy_image,list):
        sample_real_copy_image = np.array(sample_real_copy_image)
        sample_real_copy_image = np.reshape(sample_real_copy_image,[config.sample_num,config.output_height,config.output_width,dcgan.c_dim])
      save_images(sample_real_copy_image,image_manifold_size(config.sample_num),'./samples/%s/sample_images_copy_%d.png'%(current_time,i))
  elif option == 4:
    #从随机z中生成图片，并替换属性
    label_list = conv_to_list_2(config.Test_file_label)
    label_len = len(label_list)
    for i in range(200):
      print("[*]%d"% i)
      num_per_page = int(config.sample_num/(label_len))
      z_sample = np.zeros([config.sample_num,dcgan.z_dim])
      y = np.zeros([config.sample_num,dcgan.y_dim])
      z_index = y_index = 0
      for j in range(num_per_page):
        sample_temp = np.random.uniform(-1,1,size=(1,dcgan.z_dim))       
        for k in range(label_len):
          z_sample[z_index] = sample_temp[0:]
          y[y_index] = label_list[k]
          z_index += 1
          y_index += 1
      sample_image = sess.run([dcgan.sampler_data],feed_dict={dcgan.z:z_sample,dcgan.y:y})
      if isinstance(sample_image,list):
        sample_image = np.array(sample_image)
        sample_image = np.reshape(sample_image,[config.sample_num,config.output_height,config.output_width,dcgan.c_dim])
      save_images(sample_image,image_manifold_size(config.sample_num),'./samples/%s/sample_images_%d.png'%(current_time,i))

  elif option == 5:
    #随机选取真实的图片，重构并替换属性
    label_list = conv_to_list_2(config.Test_file_label)
    label_len = len(label_list)


    for i in range(10):
      #获取真实图片   
      real_image_batch_files = [dcgan.data[np.random.randint(0,len(dcgan.data))-1] for sample_index in range(config.sample_num)]
      sample_images = [get_image(sample_file,
                      input_height=config.input_height,
                      input_width=config.input_width,
                      resize_height=config.output_height,
                      resize_width=config.output_width,
                      crop=config.crop,
                      grayscale=dcgan.grayscale,recordName = True) for sample_file in real_image_batch_files]
      sample_images = np.array(sample_images).astype(np.float32)
      #保存
      save_images(sample_images,image_manifold_size(config.sample_num),'./samples/%s/real_image_%d.png'%(current_time,i))

      #逆向映射
      c_Copy,z_Copy = sess.run([dcgan.cat_Sample,dcgan.z_Sample],feed_dict = {dcgan.inputs:sample_images})

      z_Copy_total = np.zeros([config.sample_num*(1+label_len),dcgan.z_dim])
      c_Copy_total = np.zeros([config.sample_num*(1+label_len),dcgan.y_dim])
      processed_index = 0

      for j in range(len(z_Copy)):
        for k in range(label_len+1):
          z_Copy_total[processed_index] = z_Copy[j]
          if k == 0:
            c_Copy_total[processed_index] = c_Copy[j]
          else:
            for l in range(len(c_Copy[j])):
              #根据标志位判定是否需要反向
              if label_list[k-1][l] == 0:
                c_Copy_total[processed_index][l] = c_Copy[j][l]
              else:
                if c_Copy[j][l] < 0.5:
                  c_Copy_total[processed_index][l] = 1.0
                else:
                  c_Copy_total[processed_index][l] = 0.0
          processed_index += 1
      
      #分隔
      z_Copy_page = np.zeros([config.sample_num,dcgan.z_dim])
      c_Copy_page = np.zeros([config.sample_num,dcgan.y_dim])
      for k in range(label_len+1):
        z_Copy_page = z_Copy_total[k*config.sample_num : (k+1)*config.sample_num]
        c_Copy_page = c_Copy_total[k*config.sample_num : (k+1)*config.sample_num]
        sample_image = sess.run([dcgan.sampler_data],feed_dict={dcgan.z:z_Copy_page,dcgan.y:c_Copy_page})
        if isinstance(sample_image,list):
          sample_image = np.array(sample_image)
          sample_image = np.reshape(sample_image,[config.sample_num,config.output_height,config.output_width,dcgan.c_dim])
        save_images(sample_image,image_manifold_size(config.sample_num),'./samples/%s/sample_images_%d_%d.png'%(current_time,i,k))

     

  print("test complete!")

def image_manifold_size(num_images):
  manifold_h = int(np.floor(np.sqrt(num_images)))
  manifold_w = int(np.ceil(np.sqrt(num_images)))
  assert manifold_h * manifold_w == num_images
  return manifold_h, manifold_w
