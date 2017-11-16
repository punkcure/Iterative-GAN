from __future__ import division
import os
import time
import math
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange

from ops import *
from utils import *

def conv_out_size_same(size, stride):
  return int(math.ceil(float(size) / float(stride)))

def conv_to_type_list(str_types):
  y_type_list_str = str_types.split(',')
  y_type_list_int = list()
  for i in range(len(y_type_list_str)):
    y_type_list_int.append(int(y_type_list_str[i]))
   
  return y_type_list_int

class DCGAN(object):
  def __init__(self, sess,config):

    self.sess = sess

    self.gf_dim = 64
    self.df_dim = 64

    self.gfc_dim = 1024
    self.dfc_dim = 1024
    
    self.y_type_list = conv_to_type_list(config.y_types_string)
    
    self.z_dim = 100
    self.y_dim = len(self.y_type_list)

    # batch normalization : deals with poor initialization helps gradient flow
    self.d_bn1 = batch_norm(name='d_bn1')
    self.d_bn2 = batch_norm(name='d_bn2')
    self.d_bn3 = batch_norm(name='d_bn3')
    self.d_bn_share = batch_norm(name='d_bn_share')

    self.g_bn0 = batch_norm(name='g_bn0')
    self.g_bn1 = batch_norm(name='g_bn1')
    self.g_bn2 = batch_norm(name='g_bn2')
    self.g_bn3 = batch_norm(name='g_bn3')

    self.c_bn0 = batch_norm(name="c_bn0")

    self.ImageAttributeDic = getAttributeOfImageFromTxt(config.AttibutetxtPath)  
    self.data = glob(os.path.join(config.DatasetDir, config.input_fname_pattern))
    imreadImg = imread(self.data[0])
    if len(imreadImg.shape) >= 3: #check if image is a non-grayscale image by checking channel number
      self.c_dim = imread(self.data[0]).shape[-1]
    else:
      self.c_dim = 1

    self.grayscale = (self.c_dim == 1)

    self.build_model(config)
    
  def build_model(self,config):

    self.config = config
  
    image_dims = [config.output_height, config.output_width, self.c_dim]

    self.inputs = tf.placeholder(tf.float32, [config.batch_size] + image_dims, name='real_images')
    #self.y = tf.placeholder(tf.float32, [len(self.y_type_list),config.batch_size, config.y_dim],name='y')
    self.y = tf.placeholder(tf.float32,[config.batch_size,self.y_dim],name='y')
    self.z = tf.placeholder(tf.float32, [config.batch_size, self.z_dim], name='z')
    
    #important 
    #生成器
    self.G = self.generator(self.z, self.y)
    #判别器 输入：真实图片 输出：真假概率，真假概率（用于交叉熵函数），共享层（用于传递给classifier），各层的featureMap队列
    self.D_Real, self.D_Real_logits,self.D_Share_real,self.real_image_features = self.discriminator(self.inputs, reuse=False)
    #判别器 输入：伪造图片 输出：真假概率，真假概率（用于交叉熵函数），共享层（用于传递给classifier），各层的featureMap队列
    self.D_Fake, self.D_Fake_logits_, self.D_Share_fake,self.rebuild_image_features = self.discriminator(self.G,  reuse=True)
    #判别器 用于test sample 不参与训练
    self.D_Sample,self.D_Sample_logits,self.D_Share_sample,_ = self.discriminator(self.inputs,reuse=True)
    #和生成器架构几乎一致的测试器，不参与训练，用于测试自己采样的z和标签生成的图片
    self.sampler_data = self.generator(self.z, self.y,reuse=True)
    
    #分类器 输入：判别器提供的真实图片的共享层 输出：one-hot标签类型 one-hot标签类型（用于交叉熵函数） Z噪声数据
    self.cat_real,self.cat_real_logit,self.z_real = self.classifier(self.D_Share_real,reuse=False)
    #分类器 输入：判别器提供的伪造图片的共享层 输出：one-hot标签类型 one-hot标签类型（用于交叉熵函数） Z噪声数据
    self.cat_fake,self.cat_fake_logit,self.z_fake = self.classifier(self.D_Share_fake,reuse=True)
    #测试用的判别器 不参与训练
    self.cat_Sample,self.cat_Sample_logit,self.z_Sample = self.classifier(self.D_Share_sample,reuse = True)
    
    #和生成器架构几乎一致的测试器，不参与训练，基于classifier输出的z和标签，重构图片
    self.sampler_z_real = self.generator(self.z_real,self.cat_real,reuse=True)

    #标准的GAN loss之判别器loss
    self.d_loss_real = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_Real_logits, tf.ones_like(self.D_Real)))
    self.d_loss_fake = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_Fake_logits_,tf.zeros_like(self.D_Fake)))
    self.d_loss = self.d_loss_real + self.d_loss_fake
    
    #标准的GAN loss之生成器loss
    self.g_loss = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_Fake_logits_,tf.ones_like(self.D_Fake)))
    
    #生成噪音Z的loss：MSE Loss + 像素loss + feature loss 其中的featureloss函数用于计算特征图的差异，详细实现见utils.py
    self.z_loss = tf.reduce_mean(tf.square(self.z_fake-self.z))
    self.z_loss_pixel =  config.pixel_loss_lamda*tf.reduce_mean(tf.square(self.sampler_z_real - self.inputs)) 
    self.z_loss_feature = config.feature_loss_lamda*feature_loss(self.real_image_features,self.rebuild_image_features)
    
    #标签y的loss 把真实的和伪造的图片、标签对都丢进loss计算
    self.y_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.cat_real_logit,labels=self.y))
    self.y_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.cat_fake_logit,labels=self.y))

    self.z_loss_total = self.z_loss + self.z_loss_feature + self.z_loss_pixel

    #c的双loss 
    self.y_loss = config.fake_images_y_loss_lamda*self.y_loss_fake + config.true_images_y_loss_lamda*self.y_loss_real 
    
    self.q_loss = self.z_loss + self.y_loss

    t_vars = tf.trainable_variables()
    self.d_vars = [var for var in t_vars if 'd_' in var.name]
    self.g_vars = [var for var in t_vars if 'g_' in var.name]
    #self.c_vars = [var for var in t_vars if ('c_' in var.name) or ('d_' in var.name) or ('g_' in var.name)]
    self.c_vars = [var for var in t_vars if ('c_' in var.name)]
    
    self.saver = tf.train.Saver()

  def train(self, config):

    #important 最主要的三个loss， gan的和c+z的
    d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
              .minimize(self.d_loss, var_list=self.d_vars)
    g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
              .minimize(self.g_loss, var_list=self.g_vars)
    q_optim = tf.train.AdamOptimizer(config.learning_rate,beta1=config.beta1) \
              .minimize(self.q_loss,var_list=self.c_vars)

    try:
      tf.global_variables_initializer().run()
    except:
      tf.initialize_all_variables().run()
    
    #初始化sample_z
    sample_z = np.zeros([config.sample_num,self.z_dim])

    interval = len(self.y_type_list)
    num = int(config.sample_num/interval)
    for i in range(num):
      sample_temp = np.random.uniform(-1,1,size=(1,self.z_dim))
      for j in range(interval):
        sample_z[i*interval + j] = sample_temp[0:]
    
    sample_temp = np.random.uniform(-1,1,size=(1,self.z_dim))
    for i in range(config.sample_num%interval):
      sample_z[num*interval+i] = sample_temp[0:] 
    
    #sample_z = np.random.uniform(-1,1,size=(config.sample_num,self.z_dim))

    #初始化sample_y 
    sample_y = sample_y_by_y_dim(config.sample_num,self.y_dim,self.y_type_list)
    
    sample_files = self.data[0:int(config.sample_num)]
    length = len(self.data)
    for i in range(config.sample_num):
      sample_files[i] = self.data[np.random.randint(0,length)-1]

    sample_images = [get_image(sample_file,
                      input_height=config.input_height,
                      input_width=config.input_width,
                      resize_height=config.output_height,
                      resize_width=config.output_width,
                      crop=config.crop,
                      grayscale=self.grayscale,recordName = True) for sample_file in sample_files]
    sample_images = np.array(sample_images).astype(np.float32)
    save_images(sample_images, image_manifold_size(config.sample_num),'./{}/sample_images.png'.format(config.OutputDirName))
    
    counter = 1
    start_time = time.time()
    could_load, checkpoint_counter = self.load(config.checkpoint_dir)

    for epoch in xrange(config.epoch):
      data_lenth = len(self.data)
      train_size = config.train_size
      batch_idxs = min(data_lenth,train_size) // config.batch_size

      for idx in xrange(0, batch_idxs):
        batch_files = self.data[idx*config.batch_size:(idx+1)*config.batch_size]
        batch = [
            get_image(batch_file,
                      input_height=config.input_height,
                      input_width=config.input_width,
                      resize_height=config.output_height,
                      resize_width=config.output_width,
                      crop=config.crop,
                      grayscale=self.grayscale,recordName = True) for batch_file in batch_files]
        if self.grayscale:
          batch_images = np.array(batch).astype(np.float32)[:, :, :, None]
        else:
          batch_images = np.array(batch).astype(np.float32)

        batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]) \
              .astype(np.float32)

        batch_y = get_y_by_attribute(batch_files,self.ImageAttributeDic,self.y_type_list,self.y_dim)

        #important
        # Update D network
        _ = self.sess.run(d_optim,feed_dict={ self.inputs: batch_images, self.z: batch_z,self.y:batch_y})
        
        #Updata G network
        k = 2
        for _ in range(k):
          _ = self.sess.run([g_optim],feed_dict={ self.z: batch_z ,self.y : batch_y})
        
        #update z network
        #_ = self.sess.run([z_optim],feed_dict={self.z:batch_z,self.y:batch_y,self.inputs:batch_images})
        
        #Update q network
        _ = self.sess.run([q_optim],feed_dict={self.z:batch_z,self.y:batch_y,self.inputs:batch_images})
            
        errD_fake = self.d_loss_fake.eval({ self.z: batch_z,self.y:batch_y })
        errD_real = self.d_loss_real.eval({ self.inputs: batch_images,self.y:batch_y })
        errG = self.g_loss.eval({self.z:batch_z,self.y:batch_y})
        errQ = self.q_loss.eval({self.z:batch_z,self.y:batch_y,self.inputs:batch_images})
        #errYFake = self.y_loss_fake.eval({self.z:batch_z,self.y:batch_y,self.inputs:batch_images})
        #errYReal = self.y_loss_real.eval({self.z:batch_z,self.y:batch_y,self.inputs:batch_images})
        #errY = self.y_loss.eval({self.z:batch_z,self.y:batch_y,self.inputs:batch_images})


        counter += 1
        print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f,q_loss:%.8f" \
          % (epoch, idx, batch_idxs,
            time.time() - start_time, errD_fake+errD_real, errG,errQ))

        '''print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f, z_loss:%.8f y_fakeloss:%.8f,y_realloss:%.8f" \
          % (epoch, idx, batch_idxs,
            time.time() - start_time, errD_fake+errD_real, errG, errZ,errYFake,errYReal))'''

        if np.mod(counter, 1000) == 1:
          print("*****************************[Sample]**********************************") 

          #训练sample_data数据生成图片
          samples = self.sess.run(
            [self.sampler_data],
            feed_dict={
            self.z: sample_z,
            self.y: sample_y
            },
            )
          if isinstance(samples,list):
            samples = np.array(samples)
            samples = np.reshape(samples,[config.sample_num,config.output_height,config.output_width,self.c_dim])
            
          save_images(samples, image_manifold_size(config.sample_num),'./{}/train_{:02d}_{:04d}.png'.format(config.OutputDirName, epoch, idx))

          #重构sample_data产生的图片并保存
          c_Copy,z_Copy = self.sess.run([self.cat_Sample,self.z_Sample],feed_dict = {self.inputs:samples})
          sample_data_copy_image = self.sess.run([self.sampler_data],feed_dict={self.z:z_Copy,self.y:c_Copy})
          if isinstance(sample_data_copy_image,list):
            sample_data_copy_image = np.array(sample_data_copy_image)
            sample_data_copy_image = np.reshape(sample_data_copy_image,[config.sample_num,config.output_height,config.output_width,self.c_dim])
          
          save_images(sample_data_copy_image,image_manifold_size(config.sample_num),'./{}/CopySampleData_{:02d}_{:04d}.png'.format(config.OutputDirName, epoch, idx))

          #替换标签后重构sample_data图片并保存
          for i in range(config.sample_num):
            for j in range(self.y_dim):
              if(c_Copy[i][j]>0.5):
                c_Copy[i][j]=0
              else:
               c_Copy[i][j]=1

          sample_copy_image = self.sess.run([self.sampler_data],feed_dict={self.z:z_Copy,self.y:c_Copy})
          if isinstance(sample_copy_image,list):
            sample_copy_image = np.array(sample_copy_image)
            sample_copy_image = np.reshape(sample_copy_image,[config.sample_num,config.output_height,config.output_width,self.c_dim])

          save_images(sample_copy_image, image_manifold_size(config.sample_num),'./{}/CopySampleDataReplace_{:02d}_{:04d}.png'.format(config.OutputDirName, epoch, idx))

          #重构sample_texture并保存
          c_Copy,z_Copy = self.sess.run([self.cat_Sample,self.z_Sample],feed_dict={self.inputs:sample_images})
          sample_copy_image = self.sess.run([self.sampler_data],feed_dict={self.z:z_Copy,self.y:c_Copy})
          if isinstance(sample_copy_image,list):
            sample_copy_image = np.array(sample_copy_image)
            sample_copy_image = np.reshape(sample_copy_image,[config.sample_num,config.output_height,config.output_width,self.c_dim])

          save_images(sample_copy_image, image_manifold_size(config.sample_num),'./{}/CopySampleReal_{:02d}_{:04d}.png'.format(config.OutputDirName, epoch, idx))

          #替换标签后重构sample_texture并保存
          for i in range(config.sample_num):
            for j in range(self.y_dim):
              if(c_Copy[i][j]>0.5):
                c_Copy[i][j]=0
              else:
               c_Copy[i][j]=1

          sample_copy_image = self.sess.run([self.sampler_data],feed_dict={self.z:z_Copy,self.y:c_Copy})
          if isinstance(sample_copy_image,list):
            sample_copy_image = np.array(sample_copy_image)
            sample_copy_image = np.reshape(sample_copy_image,[config.sample_num,config.output_height,config.output_width,self.c_dim])

          save_images(sample_copy_image, image_manifold_size(config.sample_num),'./{}/CopySampleRealReplace_{:02d}_{:04d}.png'.format(config.OutputDirName, epoch, idx))
        

        if np.mod(counter, 1000) == 2:
          self.save(config.checkpoint_dir, counter)

#important 重要网络模型
#分类器 输入：判别器提供的共享特征层 输出：类别和Z
  def classifier(self,x,reuse):
    with tf.variable_scope("classifier") as scope:
      if reuse:
        scope.reuse_variables()
      

      out_z_logit = linear(tf.reshape(x,[self.config.batch_size,-1]),self.z_dim,'c_z_lin')
      out_z = tf.nn.sigmoid(out_z_logit)

      h0 = lrelu(self.c_bn0(linear(x,128,'c_h0_lin')))
      out_c_logit = linear(tf.reshape(x,[self.config.batch_size,-1]),self.y_dim,'c_y_lin')     
      out_c = tf.nn.sigmoid(out_c_logit)
      #out_z = linear(tf.reshape(x,[self.config.batch_size,-1]),self.z_dim,'c_z_lin')
      
      #h0 = lrelu(self.c_bn0(linear(x,128,'c_h0_lin')))

      #out_c_logit_list = [linear(h0,self.config.y_dim,'c_out_logit_lin_%d'%(i)) for i in range(len(self.y_type_list))]
      #out_c_list = [tf.nn.softmax(out_c_logit_single) for  out_c_logit_single in out_c_logit_list]

      #for i in range(len(out_c_logit_list)):
      #  out_c_logit_list[i] = tf.reshape(out_c_logit_list[i],[1,self.config.batch_size,self.config.y_dim])
      #for i in range(len(out_c_list)):
      #  out_c_list[i] = tf.reshape(out_c_list[i],[1,self.config.batch_size,self.config.y_dim])

      #list->tensor
      #out_c = out_c_list[0]
      #for i in range(len(out_c_list)-1):
      #  out_c = tf.concat([out_c,out_c_list[i+1]],0)
      
      #out_c_logit = out_c_logit_list[0]
      #for i in range(len(out_c_logit_list)-1):
      #  out_c_logit = tf.concat([out_c_logit,out_c_logit_list[i]],0)

      #out_c_logit = [linear(h0,self.config.y_dim,'c_out_logit_lin')]
      #out_c = tf.nn.softmax(out_c_logit)
      
      #out_c_logit = linear(h0,self.config.y_dim,'c_out_logit')
      #out_c = tf.nn.tanh(out_c_logit)

      return out_c,out_c_logit,out_z

#GAN中的判别器 改动：1.比传统的判别器多输出一个共享层 2.创建一个list来记录每一层的特征图，用以做 feature_loss
  def discriminator(self, image, reuse=False):
    with tf.variable_scope("discriminator") as scope:
      if reuse:
        scope.reuse_variables()
      
      #特征图list
      features = list() 

      h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
      #记录特征图 以下同
      features.append(h0)
      
      h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
      features.append(h1)
      
      h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
      features.append(h2)
      
      h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, name='d_h3_conv')))
      features.append(h3)
      
      #输出共享层
      share = lrelu(self.d_bn_share(linear(tf.reshape(h3,[self.config.batch_size,-1]),1024,'d_share')))

      #全连接后 输出真假判定概率
      h4 = linear(tf.reshape(h3, [self.config.batch_size, -1]), 1, 'd_h4_lin')
      

      return tf.nn.sigmoid(h4), h4,share,features

#GAN中的生成器 改进：z和标签y连接起来生成
  def generator(self, z, y=None,reuse = False):
    with tf.variable_scope("generator") as scope:
      if reuse:
        scope.reuse_variables()
      #连接Z和标签Y
      if  self.y_dim and self.y_dim != 0:
          '''batch_num = int(z.get_shape()[0])
          addition = self.config.y_dim*len(self.y_type_list)
          y_reshape = tf.reshape(y,[batch_num,addition])
          z = concat(z,y_reshape)'''
          z = concat(z,y)

      s_h, s_w = self.config.output_height, self.config.output_width
      s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
      s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
      s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
      s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

      # project `z` and reshape 线性乘法
      self.z_, self.h0_w, self.h0_b = linear(
          z, self.gf_dim*8*s_h16*s_w16, 'g_h0_lin', with_w=True) #[bz,100]

      self.h0 = tf.reshape(
          self.z_, [-1, s_h16, s_w16, self.gf_dim * 8])#[bz,4,4,512]
      h0 = tf.nn.relu(self.g_bn0(self.h0))

      #一路反卷积
      self.h1, self.h1_w, self.h1_b = deconv2d(
          h0, [self.config.batch_size, s_h8, s_w8, self.gf_dim*4], name='g_h1', with_w=True)#[bz,8,8,256]
      h1 = tf.nn.relu(self.g_bn1(self.h1))

      h2, self.h2_w, self.h2_b = deconv2d(
           h1, [self.config.batch_size, s_h4, s_w4, self.gf_dim*2], name='g_h2', with_w=True)#[bz,16,16,128]
      h2 = tf.nn.relu(self.g_bn2(h2))

      h3, self.h3_w, self.h3_b = deconv2d(
          h2, [self.config.batch_size, s_h2, s_w2, self.gf_dim*1], name='g_h3', with_w=True)#[bz,32,32,64]
      h3 = tf.nn.relu(self.g_bn3(h3))

      h4, self.h4_w, self.h4_b = deconv2d(
          h3, [self.config.batch_size, s_h, s_w, self.c_dim], name='g_h4', with_w=True)#[bz,64,64,3]

      return tf.nn.tanh(h4)

#和生成器G几乎完全一样的生成器 应该省略掉
  '''def sampler(self, z, y=None):
    with tf.variable_scope("generator") as scope:
      scope.reuse_variables()

      if self.config.y_dim:
       for i in range(len(self.y_type_list)):
          z = concat(z,y[i])

      s_h, s_w = self.config.output_height, self.config.output_width
      s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
      s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
      s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
      s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

      # project `z` and reshape
      h0 = tf.reshape(
          linear(z, self.gf_dim*8*s_h16*s_w16, 'g_h0_lin'),
          [-1, s_h16, s_w16, self.gf_dim * 8])
      h0 = tf.nn.relu(self.g_bn0(h0, train=False))

      h1 = deconv2d(h0, [self.config.batch_size, s_h8, s_w8, self.gf_dim*4], name='g_h1')
      h1 = tf.nn.relu(self.g_bn1(h1, train=False))

      h2 = deconv2d(h1, [self.config.batch_size, s_h4, s_w4, self.gf_dim*2], name='g_h2')
      h2 = tf.nn.relu(self.g_bn2(h2, train=False))

      h3 = deconv2d(h2, [self.config.batch_size, s_h2, s_w2, self.gf_dim*1], name='g_h3')
      h3 = tf.nn.relu(self.g_bn3(h3, train=False))

      h4 = deconv2d(h3, [self.config.batch_size, s_h, s_w, self.c_dim], name='g_h4')

      return tf.nn.tanh(h4)'''

  @property
  def model_dir(self):
    return "{}_{}_{}".format(
        self.config.batch_size,
        self.config.output_height, self.config.output_width)
      
  def save(self, checkpoint_dir, step):
    model_name = "DCGAN.model"
    checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)

    self.saver.save(self.sess,
            os.path.join(checkpoint_dir, model_name),
            global_step=step)

  def load(self, checkpoint_dir):
    import re
    print(" [*] Reading checkpoints...")
    checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
      self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
      counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
      print(" [*] Success to read {}".format(ckpt_name))

      uninit_vars_after = []
      for var in tf.all_variables():
        try:
          self.sess.run(var)
          print(var.name)
        except tf.errors.FailedPreconditionError:
          uninit_vars_after.append(var)

      for var in uninit_vars_after:
        print("uninitial----",var.name)

      return True, counter
    else:
      print(" [*] Failed to find a checkpoint")
      return False, 0
