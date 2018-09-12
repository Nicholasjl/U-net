import  sys
import  os
sys.path.append('../')
from  Unet.data_Keras  import Augmentation, DataProcess

mydata = DataProcess(512, 1024)
mydata.write_img_to_tfrecords()
aug = Augmentation()
aug.augmentation()
aug.split_merge()
os.system('python3 unet-TF-withBatchNormal.py')
