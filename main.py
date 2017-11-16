import os
import scipy.misc
import numpy as np
import time

from model import DCGAN
from utils import pp, visualize, to_json, show_all_variables,getAttributeOfImageFromTxt

import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_integer("epoch", 300, "Epoch to train [300]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("sample_num",64,"The size of sample")
flags.DEFINE_integer("input_height", 218, "The size of image to use (will be center cropped). [96]")
flags.DEFINE_integer("input_width", 178, "The size of image to use (will be center cropped). If None, same value as input_height [96]")
flags.DEFINE_integer("output_height", 80, "The size of the output images to produce [48]")
flags.DEFINE_integer("output_width", 80, "The size of the output images to produce. If None, same value as output_height [48]")
flags.DEFINE_string("input_fname_pattern", "*.png", "Glob pattern of filename of input images [*]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("train", True, "True for training, False for testing [False]")
flags.DEFINE_boolean("crop", True, "True for training, False for testing [False]")
flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")
flags.DEFINE_float("pixel_loss_lamda",0.5,"pixel_loss coefficient")
flags.DEFINE_float("feature_loss_lamda",1.0,"feature_loss coefficient")
flags.DEFINE_float("true_images_y_loss_lamda",1.0,"true images of y loss coefficient") 
flags.DEFINE_float("fake_images_y_loss_lamda",1.0,"false images of y loss coefficient")

#弯眉 秃头 刘海 浓眉 eyeglass 高颧骨 male smile wave_hair young
#flags.DEFINE_string("y_types_string",'1,4,5,12,15,19,20,31,33,39',"the index of attribute list is set to 1")
#刘海 眼镜 男女 微笑 年轻  attri1
#flags.DEFINE_string("y_types_string",'5,15,20,31,39',"the index of attribute list is set to 1")
#刘海  浓眉  眼镜 微笑 卷发   attri2
flags.DEFINE_string("y_types_string",'5,12,15,31,33',"the index of attribute list is set to 1")

flags.DEFINE_string("OutputDirName", "./Result","Output dir of the results")
flags.DEFINE_string("DatasetDir","F:/TrainningResource/CelebA/Img/img_align_celeba_png.7z/img_align_celeba_png.7z/img_align_celeba_png","FullPath of dataset")
flags.DEFINE_string("AttibutetxtPath","F:/TrainningResource/CelebA/Anno/list_attr_celeba.txt","Full Path of Image attribute txt file")
#flags.DEFINE_string("DatasetDir","E:/ML Code/TrainningSource/celeA/img_align_celeba_png","FullPath of dataset")
#flags.DEFINE_string("AttibutetxtPath","E:/ML Code/TrainningSource/celeA/list_attr_celeba.txt","Full Path of Image attribute txt file")
flags.DEFINE_string("Test_file_names","013585.png,023822.png,033551.png,075862.png,096946.png,098041.png,136564.png,173911.png,189432.png,192206.png","The name of file to be tested")
#attribute1
#flags.DEFINE_string("Test_file_label","1,0,0,0,0;0,1,0,0,0;0,0,1,0,0;0,0,0,1,0;0,0,0,0,1;1,1,0,0,0;1,1,0,1,0;1,1,0,1,1","label of image to be rebuild")
#attribute2
flags.DEFINE_string("Test_file_label","0,0,0,0,0;1,0,0,0,0;0,1,0,0,0;1,1,0,0,0;0,0,1,0,0;1,1,1,0,0;1,1,1,1,0;1,1,1,1,1","label of image to be rebuild")
FLAGS = flags.FLAGS

def main(_):
  pp.pprint(flags.FLAGS.__flags)

  if FLAGS.input_width is None:
    FLAGS.input_width = FLAGS.input_height
  if FLAGS.output_width is None:
    FLAGS.output_width = FLAGS.output_height

  if not os.path.exists(FLAGS.checkpoint_dir):
    os.makedirs(FLAGS.checkpoint_dir)
  if not os.path.exists(FLAGS.sample_dir):
    os.makedirs(FLAGS.sample_dir)
  
  current_time = time.strftime('%Y-%m-%d_%H_%M_%S',time.localtime(time.time()))
  FLAGS.OutputDirName += "/"+current_time
  if not os.path.exists(FLAGS.OutputDirName):
    os.makedirs(FLAGS.OutputDirName)

  #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
  run_config = tf.ConfigProto()
  run_config.gpu_options.allow_growth=True

  with tf.Session(config=run_config) as sess:
      dcgan = DCGAN(
          sess,
          FLAGS)

      show_all_variables()

      if FLAGS.train:
        dcgan.train(FLAGS)
      else:
        if not dcgan.load(FLAGS.checkpoint_dir)[0]:
          raise Exception("[!] Train a model first, then run test mode")
      

    # to_json("./web/js/layers.js", [dcgan.h0_w, dcgan.h0_b, dcgan.g_bn0],
    #                 [dcgan.h1_w, dcgan.h1_b, dcgan.g_bn1],
    #                 [dcgan.h2_w, dcgan.h2_b, dcgan.g_bn2],
    #                 [dcgan.h3_w, dcgan.h3_b, dcgan.g_bn3],
    #                 [dcgan.h4_w, dcgan.h4_b, None])

    # Below is codes for visualization
      OPTION = 4
      visualize(sess, dcgan, FLAGS, OPTION)

if __name__ == '__main__':
  tf.app.run()
