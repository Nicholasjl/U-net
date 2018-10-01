# -*- coding:utf-8 -*-
"""
#====#====#====#====
# Project Name:     U-net 
# File Name:        data_TF 
# Date:             2/10/18 8:38 PM 
# Using IDE:        PyCharm Community Edition  
# From HomePage:    https://github.com/DuFanXin/U-net
# Author:           DuFanXin 
# BlogPage:         http://blog.csdn.net/qq_30239975  
# E-mail:           18672969179@163.com
# Copyright (c) 2018, All Rights Reserved.
#====#====#====#==== 
"""
import Augmentor
import os
import glob
import cv2
import tensorflow as tf
import numpy as np
TRAIN_SET_NAME = 'train_set.tfrecords'
VALIDATION_SET_NAME = 'validation_set.tfrecords'
TEST_SET_NAME = 'test_set.tfrecords'
PREDICT_SET_NAME = 'predict_set.tfrecords'
VAL_IMAGE_DIRECTORY='../data_set/val_images'
VAL_LABEL_DIRECTORY='../data_set/val_labels'
ORIGIN_MERGED_SOURCE_DIRECTORY = '../data_set/merged_origin_data_set'
ORIGIN_IMAGE_DIRECTORY = '../data_set/images'
ORIGIN_LABEL_DIRECTORY = '../data_set/labels'
ORIGIN_PREDICT_DIRECTORY = '../data_set/test'
# Augmentor.Pipeline的参数'output_directory'有毒，非要绝对路径，只能这样咯
AUGMENT_OUTPUT_DIRECTORY = \
	os.getcwd()[:os.getcwd().rindex('/')] + '/data_set/my_set/merged_augment_data_set'
AUGMENT_IMAGE_PATH = '../data_set/val_images'
AUGMENT_LABEL_PATH = '../data_set/val_labels'

INPUT_IMG_WIDE, INPUT_IMG_HEIGHT, INPUT_IMG_CHANNEL = 512, 512, 1
OUTPUT_IMG_WIDE, OUTPUT_IMG_HEIGHT, OUTPUT_IMG_CHANNEL = 512, 512, 1
TRAIN_SET_SIZE = 2100
VALIDATION_SET_SIZE = 27
TEST_SET_SIZE = 30
PREDICT_SET_SIZE = 30


def augment():
	p = Augmentor.Pipeline(
		source_directory=ORIGIN_MERGED_SOURCE_DIRECTORY,
		output_directory=AUGMENT_OUTPUT_DIRECTORY
	)
	p.rotate(probability=0.2, max_left_rotation=2, max_right_rotation=2)
	p.zoom(probability=0.2, min_factor=1.1, max_factor=1.2)
	p.skew(probability=0.2)
	p.random_distortion(probability=0.2, grid_width=100, grid_height=100, magnitude=1)
	p.shear(probability=0.2, max_shear_left=2, max_shear_right=2)
	p.crop_random(probability=0.2, percentage_area=0.8)
	p.flip_random(probability=0.2)
	p.sample(n=TRAIN_SET_SIZE + VALIDATION_SET_SIZE + TEST_SET_SIZE)


def split_merged_augment_data_set():
	merged_data_set_paths = glob.glob(os.path.join(AUGMENT_OUTPUT_DIRECTORY, '*.JPEG'))
	augment_image_path = AUGMENT_IMAGE_PATH
	if not os.path.lexists(augment_image_path):
		os.mkdir(augment_image_path)
	augment_label_path = AUGMENT_LABEL_PATH
	if not os.path.lexists(augment_label_path):
		os.mkdir(augment_label_path)
	for index, merged_data_set_path in enumerate(merged_data_set_paths):
		merged_image = cv2.imread(merged_data_set_path)
		print(merged_data_set_path)
		image = merged_image[:, :, 0]
		label = merged_image[:, :, 2]
		cv2.imwrite(filename=os.path.join(augment_image_path, '%d.jpg' % index), img=image)
		cv2.imwrite(filename=os.path.join(augment_label_path, '%d.jpg' % index), img=label)
	print('Done split merged augment data_set. \nImages at %s\nLabels at %s' % (augment_image_path, augment_label_path))


def write_img_to_tfrecords():
	augment_image_path = AUGMENT_IMAGE_PATH
	augment_label_path = AUGMENT_LABEL_PATH
	train_set_writer = tf.python_io.TFRecordWriter(os.path.join('../data_set', TRAIN_SET_NAME))  # 要生成的文件
	validation_set_writer = tf.python_io.TFRecordWriter(os.path.join('../data_set', VALIDATION_SET_NAME))
	test_set_writer = tf.python_io.TFRecordWriter(os.path.join('../data_set', TEST_SET_NAME))  # 要生成的文件
	predict_set_writer = tf.python_io.TFRecordWriter(os.path.join('../data_set', PREDICT_SET_NAME))  # 要生成的文件

	
	# train_set
	origin_image_path = ORIGIN_IMAGE_DIRECTORY
	origin_label_path = ORIGIN_LABEL_DIRECTORY
	for _dir in os.listdir(origin_image_path):
		image_file_dir = os.path.join(origin_image_path,_dir)
		for dirr in os.listdir(image_file_dir):		
			train_image = cv2.imread(os.path.join(image_file_dir, dirr), flags=0)
			train_label = cv2.imread(os.path.join(origin_label_path,_dir[:-4]+'_labelMark', dirr), flags=0)

			# train_image = cv2.resize(src=train_image, dsize=(INPUT_IMG_WIDE, INPUT_IMG_HEIGHT))
			# train_image = np.asarray(a=train_image, dtype=np.uint8)
			train_image = cv2.resize(src=train_image, dsize=(INPUT_IMG_WIDE, INPUT_IMG_HEIGHT))
			
			train_label = cv2.resize(src=train_label, dsize=(OUTPUT_IMG_WIDE, OUTPUT_IMG_HEIGHT))
			train_label = np.asarray(a=train_label, dtype=np.uint8)
			train_label[train_label == 128] = 1
			train_label[train_label == 191] = 2
			train_label[train_label == 255] = 3
			# train_image = io.imread(file_path)
			# train_image = transform.resize(train_image, (INPUT_IMG_WIDE, INPUT_IMG_HEIGHT, INPUT_IMG_CHANNEL))
			# sample_image = train_image[:, :, 0]
			# label_image = train_image[:, :, 2]
			# label_image[label_image < 100] = 0
			# label_image[label_image > 100] = 10
			example = tf.train.Example(features=tf.train.Features(feature={
				'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[train_label.tobytes()])),
				'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[train_image.tobytes()]))
			}))  # example对象对label和image数据进行封装l
			train_set_writer.write(example.SerializeToString())  # 序列化为字符串
		
	train_set_writer.close()
	print("Done train_set writing")
	
	# validation_set

	origin_image_path = VAL_IMAGE_DIRECTORY
	origin_label_path = VAL_LABEL_DIRECTORY
	for _dir in os.listdir(origin_image_path):
		image_file_dir = os.path.join(origin_image_path,_dir)
		for dirr in os.listdir(image_file_dir):		
			validation_image = cv2.imread(os.path.join(image_file_dir, dirr), flags=0)
			validation_label = cv2.imread(os.path.join(origin_label_path,_dir[:-4]+'_labelMark', dirr), flags=0)
			
		# validation_image = cv2.resize(src=validation_image, dsize=(INPUT_IMG_WIDE, INPUT_IMG_HEIGHT))
		# validation_image = np.asarray(a=validation_image, dtype=np.uint8)
			validation_image = cv2.resize(src=validation_image, dsize=(INPUT_IMG_WIDE, INPUT_IMG_HEIGHT))
		# validation_label = np.asarray(a=validation_label, dtype=np.uint8)
			validation_label = cv2.resize(src=validation_label, dsize=(OUTPUT_IMG_WIDE, OUTPUT_IMG_HEIGHT))
			validation_label = np.asarray(a=validation_label, dtype=np.uint8)
			validation_label[validation_label == 128] = 1
			validation_label[validation_label == 191] = 2
			validation_label[validation_label == 255] = 3
		# validation_image = io.imread(file_path)
		# validation_image = transform.resize(validation_image, (INPUT_IMG_WIDE, INPUT_IMG_HEIGHT, INPUT_IMG_CHANNEL))
		# sample_image = validation_image[:, :, 0]
		# label_image = validation_image[:, :, 2]
		# label_image[label_image < 100] = 0
		# label_image[label_image > 100] = 10
			example = tf.train.Example(features=tf.train.Features(feature={
				'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[validation_label.tobytes()])),
				'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[validation_image.tobytes()]))
			}))  # example对象对label和image数据进行封装
			validation_set_writer.write(example.SerializeToString())  # 序列化为字符串
		
	validation_set_writer.close()
	print("Done validation_set writing")
	'''
	# test_set
	test_image_path=ORIGIN_PREDICT_DIRECTORY
	for dir in os.listdir(test_image_path):
		image_file_dir = os.path.join(origin_image_path,dir)
		for dirr in os.listdir(image_file_dir):	
		#test_image = cv2.imread(os.path.join(augment_image_path, '%d.jpg' % index), flags=0)
		#test_label = cv2.imread(os.path.join(augment_label_path, '%d.jpg' % index), flags=0)
		# test_image = cv2.resize(src=test_image, dsize=(INPUT_IMG_WIDE, INPUT_IMG_HEIGHT))
		# test_image = np.asarray(a=test_image, dtype=np.uint8)
		test_image = cv2.resize(src=test_image, dsize=(INPUT_IMG_WIDE, INPUT_IMG_HEIGHT))
		# test_label = np.asarray(a=test_label, dtype=np.uint8)
		test_label = cv2.resize(src=test_label, dsize=(OUTPUT_IMG_WIDE, OUTPUT_IMG_HEIGHT))
		#test_label[test_label <= 100] = 0
		#test_label[test_label > 100] = 1
		# test_image = io.imread(file_path)
		# test_image = transform.resize(test_image, (INPUT_IMG_WIDE, INPUT_IMG_HEIGHT, INPUT_IMG_CHANNEL))
		# sample_image = test_image[:, :, 0]
		# label_image = test_image[:, :, 2]
		# label_image[label_image < 100] = 0
		# label_image[label_image > 100] = 10
		example = tf.train.Example(features=tf.train.Features(feature={
			'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[test_label.tobytes()])),
			'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[test_image.tobytes()]))
		}))  # example对象对label和image数据进行封装
		test_set_writer.write(example.SerializeToString())  # 序列化为字符串
		if index % 10 == 0:
			print('Done test_set writing %.2f%%' % ((index - TRAIN_SET_SIZE - VALIDATION_SET_SIZE) / TEST_SET_SIZE * 100))
	test_set_writer.close()
	print("Done test_set writing")
	'''
	# predict_set
	test_image_path=ORIGIN_PREDICT_DIRECTORY
	for dir in os.listdir(test_image_path):
		image_file_dir = os.path.join(test_image_path,dir)
		
		for dirr in os.listdir(image_file_dir):		
			predict_image = cv2.imread(os.path.join(image_file_dir, dirr), flags=0)
			predict_label = cv2.imread(os.path.join(image_file_dir, dirr), flags=0)
		
		
		# predict_image = cv2.resize(src=predict_image, dsize=(INPUT_IMG_WIDE, INPUT_IMG_HEIGHT))
		# predict_image = np.asarray(a=predict_image, dtype=np.uint8)
			predict_image = cv2.resize(src=predict_image, dsize=(INPUT_IMG_WIDE, INPUT_IMG_HEIGHT))
		# predict_label = np.asarray(a=predict_label, dtype=np.uint8)
			predict_label = cv2.resize(src=predict_label, dsize=(OUTPUT_IMG_WIDE, OUTPUT_IMG_HEIGHT))
			predict_label = np.asarray(a=train_label, dtype=np.uint8)
			predict_label[predict_label == 128] = 1
			predict_label[predict_label == 191] = 2
			predict_label[predict_label == 255] = 3
		# predict_image = io.imread(file_path)
		# predict_image = transform.resize(predict_image, (INPUT_IMG_WIDE, INPUT_IMG_HEIGHT, INPUT_IMG_CHANNEL))
		# sample_image = predict_image[:, :, 0]
		# label_image = predict_image[:, :, 2]
		# label_image[label_image < 100] = 0
		# label_image[label_image > 100] = 10
			example = tf.train.Example(features=tf.train.Features(feature={
				'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[predict_label.tobytes()])),
				'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[predict_image.tobytes()]))
			}))  # example对象对label和image数据进行封装
			predict_set_writer.write(example.SerializeToString())  # 序列化为字符串
	predict_set_writer.close()
	print("Done predict_set writing")


def write_img_to_tfrecords2():
	aug_merge_path = '../data_set/aug_merge'
	aug_train_path = "../data_set/aug_train"
	aug_label_path = "../data_set/aug_label"
	images = []
	for indir in os.listdir(aug_merge_path):
		trainpath = os.path.join(aug_train_path, indir)
		labelpath = os.path.join(aug_label_path, indir)
		print(trainpath, labelpath)
		imgs = glob.glob(trainpath + '/*' + '.tif')
		images.extend(imgs)
	print(len(images))
	# for imgname in images:
	# 	trainmidname = imgname[imgname.rindex('/') + 1:]
	# 	labelimgname = imgname[imgname.rindex('/') + 1:imgname.rindex('_')] + '_label.tif'
	# 	print(trainmidname, labelimgname)
	# img = load_img(trainPath + '/' + trainmidname, grayscale=True)
	# label = load_img(labelPath + '/' + labelimgname, grayscale=True)

if __name__ == '__main__':
	# augment()
	# split_merged_augment_data_set()
	# write_img_to_tfrecords()
	write_img_to_tfrecords()
	# ORIGIN_SOURCE_DIRECTORY = os.getcwd()
	# trainmidname = AUGMENT_OUTPUT_DIRECTORY[:AUGMENT_OUTPUT_DIRECTORY.rindex('/')]
	AUGMENT_OUTPUT_DIRECTORY = \
		os.getcwd()[:os.getcwd().rindex('/')] + '/data_set/my_set/merged_augment_data_set'
	# print(AUGMENT_OUTPUT_DIRECTORY)
