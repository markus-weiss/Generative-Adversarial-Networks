import tensorflow as tf
import numpy as np
import os
from SLFiles import SignLanguageFiles as SLF
from cnn_img_features import *
from config import *

h = 144#72#np.int(H/down_sampling_factor)
w = 193#97
def build_model(input_image, input_flo, labels, no_classes, trainable):
    mean, var = tf.nn.moments(input_image,[0, 1, 2])
    input_image = tf.nn.batch_normalization(input_image, mean, var, offset = None, scale = None, variance_epsilon = 1e-3)
    pool_5_img = build_img_features_stream('img_stream', input_image, trainable = trainable)
    pool_5_flo = build_img_features_stream('flow_stream', input_flo, trainable = trainable)
    fc8 = average_stream(pool_5_img, pool_5_flo, no_classes, evaluation= not trainable, trainable=trainable)
    shape = fc8.get_shape().as_list()
    batch_size_training = [shape[0]]
    print('shape inputs ', shape)
    #reshape to (batch_size_training = 1, encoder_inputs)
    
    if trainable:
       with tf.name_scope('loss'):           
           cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=fc8)

       with tf.name_scope('adam_optimizer'):
           global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
           train_step = tf.train.AdamOptimizer(1e-5).minimize(cross_entropy, global_step = global_step)

        
    else:
        cross_entropy = tf.nn.softmax(fc8)
        train_step = None
        global_step = None
    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.cast(tf.argmax(fc8, 1),tf.int32), labels)
        correct_prediction = tf.cast(correct_prediction, tf.float32)
        accuracy = tf.reduce_mean(correct_prediction)
    return cross_entropy, train_step, global_step, accuracy

# Reads an image from a file, decodes it into a dense tensor, and resizes it
# to a fixed shape.
def _parse_function(filename, label):
  image_string = tf.read_file(filename)
  image_decoded = tf.image.decode_image(image_string,channels=3)
  image_decoded.set_shape([None, None, 3])
  image_resized = tf.image.resize_images(image_decoded, [h,w])
  #return tf.cast(image_decoded, tf.float32), label
  return image_resized, label
  

    
if __name__ == '__main__':
    print(tf.VERSION)
    iso_no = 450
    con_no = 780
    gs = 0

    W = 776
    H = 578
    down_sampling_factor = 4
    #144 = np.int(H/down..)
    #194 #np.int(W/down_sampling_factor)
    start_iter = 0
    no_epochs = 1
    tf.reset_default_graph()
    train_graph = tf.Graph()
    infer_graph = tf.Graph()
    
    train_sess = tf.Session(graph=train_graph)
    infer_sess = tf.Session(graph=infer_graph)

    with train_graph.as_default():
        signer_no = 0
        start_iter = 0
        iterator = SLF(database_path, "iso", signer_no, start_iter)
        no_classes = 450
        j = 0
        all_files = []
        all_labels  = []
        all_files_flo = []
        batch_size = 60
        #input_shape = [batch_size, h, w, 3]
        #print('tensor input shape: ', input_shape)    
        for val in iterator:            
            iso_files, iso_files_flo, translation_file, label = val
            #print('transl file, sign_no: ', translation_file, i)
            all_files.append(iso_files)
            all_files_flo.append(iso_files_flo)
            clabels = np.zeros((batch_size), dtype = np.int32)
            clabels[:] = label - 1
            all_labels.append(clabels)
            j += 1
        all_files = np.asarray(all_files)
        all_files = all_files.flatten()
        all_files_flo = np.asarray(all_files_flo)
        all_files_flo = all_files_flo.flatten()
        all_labels = np.asarray(all_labels)
        #print('files shape: ', all_files.shape)
        
        all_labels = all_labels.flatten()
        print('labels: ', all_labels.shape)    
        #all_files = tf.constant(["D:/SIGNUM/s03-p01\iso0001\s03-p01-i0001-f0011.jpg", "D:/SIGNUM/s03-p01\iso0001\s03-p01-i0001-f0012.jpg","D:/SIGNUM/s03-p01\iso0001\s03-p01-i0001-f0013.jpg"])
        #all_labels = tf.constant([0,0,0],dtype = tf.int32)    
#==============================================================================
        dataset = tf.data.Dataset.from_tensor_slices((all_files, all_labels))
        dataset = dataset.map(_parse_function)
        dataset = dataset.batch(batch_size)         
        iterator_ds = dataset.make_initializable_iterator()
        img_input, labels = iterator_ds.get_next()
        
        
        all_files_flo = tf.constant(["D:/SIGNUM/s03-p01\iso0001\s03-p01-i0001-f0011.jpg", "D:/SIGNUM/s03-p01\iso0001\s03-p01-i0001-f0012.jpg","D:/SIGNUM/s03-p01\iso0001\s03-p01-i0001-f0013.jpg"])
        dataset_flo = tf.data.Dataset.from_tensor_slices((all_files_flo, all_labels))
        dataset_flo = dataset_flo.map(_parse_function)
        dataset_flo = dataset_flo.batch(batch_size)
         
        iterator_ds_flo = dataset_flo.make_initializable_iterator()
        flo_input, _ = iterator_ds_flo.get_next()
         #img_input = tf.placeholder(tf.float32, shape=[60, h, w, 3], name='imgs_as_rgb')
         #labels = tf.placeholder(tf.int32, shape = [60,450])
        cross_entropy,train_step, global_step, accuracy = build_model(img_input, flo_input, labels, no_classes, True)
        initializer = tf.global_variables_initializer()         
        train_saver = tf.train.Saver()
        #print(tf.global_variables())
        var_img_stream = [v for v in tf.trainable_variables() if v.name.startswith('img_stream')]
        print('vars to be transferred: ', var_img_stream)
        train_saver_img_stream = tf.train.Saver(var_img_stream)
    
    
    
    train_sess.run(initializer)
    
    checkpoint_dir = checkpoints_path+'/classification/'
    path = tf.train.latest_checkpoint(checkpoint_dir)
    if path:
        print('restore path: ', path)
        train_saver.restore(train_sess,path)
    
    f = open('log.txt','w')
    sum_ce = tf.summary.scalar("cost", cross_entropy)
    sum_acc = tf.summary.scalar("accuracy", accuracy)
    #sum_acc_t = tf.summary.scalar("accuracy_test", accuracy_infer)
    train_writer = tf.summary.FileWriter(graph_path+'/train',train_graph )
    #test_writer = tf.summary.FileWriter(graph_path+'/test', infer_graph)
    for epoch in range(no_epochs):
        train_sess.run(iterator_ds.initializer)
        train_sess.run(iterator_ds_flo.initializer)
        acc = 0.0
        i = 0
        try:
            while True:
                ce, _, glob_step, acc_, sum_ce_, sum_acc_ = train_sess.run([cross_entropy,  train_step, global_step, accuracy, sum_ce, sum_acc]) 
                train_writer.add_summary(sum_ce_, i*epoch + i)
                train_writer.add_summary(sum_acc_, i*epoch + i)
                if (i%300 == 0):
                    print('accuracy, per batch: ',acc_, ce, i, glob_step)
                    f.write('train accuracy, per batch: '+ str(acc_) +' '+ str(i)+ '\n')
                i += 1
                acc += acc_
        except tf.errors.OutOfRangeError:
            print('endof epoch: ', epoch)
             #save after one epoch
        print('accuracy, per epoch: ',acc/i)   
        f.write('train accuracy, per epoch: '+ str(acc/i) + '\n')
        train_saver.save(train_sess, checkpoint_dir, global_step = glob_step)
        train_saver_img_stream.save(train_sess,checkpoint_dir+'tf/tlchkpt')
    print('filenames: ', all_files[0:4])
    print('filenames_flow: ', all_files_flo[0:4])
    f.close()
#==============================================================================
    train_writer.close()
    train_sess.close()    
#==============================================================================
#     with infer_graph.as_default():                
#         iterator_infer = SLF(database_path_test, "iso", signer_no, start_iter)
#         no_classes = 450
#         j = 0
#         all_files = []
#         all_files_flo = []
#         all_labels  = []
#         batch_size = 60
#         #input_shape = [batch_size, h, w, 3]
#         #print('tensor input shape: ', input_shape)    
#         for val in iterator_infer:            
#             iso_files, iso_files_flo, translation_file, label = val
#             print('transl file, label: ', translation_file, label)
#             all_files.append(iso_files)
#             all_files_flo.append(iso_files_flo)
#             clabels = np.zeros((batch_size), dtype = np.int32)
#             clabels[:] = label - 1
#             all_labels.append(clabels)
#             j += 1
#         all_files = np.asarray(all_files)
#         all_files = all_files.flatten()
#         all_labels = np.asarray(all_labels)
#         all_labels = all_labels.flatten() 
#         all_files_flo = np.asarray(all_files_flo)
#         all_files_flo = all_files_flo.flatten()
#         #all_files = tf.constant(["E:/SIGNUM/s03-p01\iso0001\s03-p01-i0001-f0011.jpg", "E:/SIGNUM/s03-p01\iso0001\s03-p01-i0001-f0012.jpg","E:/SIGNUM/s03-p01\iso0001\s03-p01-i0001-f0013.jpg"])
#         #all_labels = tf.constant([0,0,0],dtype = tf.int32) 
#         dataset = tf.data.Dataset.from_tensor_slices((all_files, all_labels))
#         dataset = dataset.map(_parse_function)
#         dataset = dataset.batch(batch_size)
#         iterator_infer_ds = dataset.make_one_shot_iterator()
#         img_input, labels = iterator_infer_ds.get_next()
#         
#         
#         dataset_test_flo = tf.data.Dataset.from_tensor_slices((all_files_flo, all_labels))
#         dataset_test_flo = dataset_test_flo.map(_parse_function)
#         dataset_test_flo = dataset_test_flo.batch(batch_size)
#          
#         iterator_ds_test_flo = dataset_test_flo.make_one_shot_iterator()
#         flo_input, _ = iterator_ds_test_flo.get_next()
# 
#         cross_entropy_infer, _, _, accuracy_infer = build_model(img_input, flo_input, labels, no_classes, False)
#         infer_saver = tf.train.Saver()
#         
#     checkpoint_dir = checkpoints_path+'/classification/'
#     path = tf.train.latest_checkpoint(checkpoint_dir)
#     if path:
#         print('restore path: ', path)        
#         infer_saver.restore(infer_sess,path)
#         acc = 0.0
#         i = 0
#         while True:
#             try:
#                 ce, acc_ = infer_sess.run([cross_entropy_infer, accuracy_infer]) 
#                 #test_writer.add_summary(acc_, i)
#                 if (i%300 == 0):
#                     print('accuracy, per batch: ',acc_, i)
#                     #f.write('test accuracy, per batch: '+ str(acc_) +' '+ str(i)+ '\n')
#                 i += 1
#                 acc += acc_
#             except tf.errors.OutOfRangeError:
#                 break
#              #save after one epoch
#         print('accuracy, per epoch: ',acc/i)   
#         #f.write('test accuracy, per epoch: '+ str(acc/i) + '\n')
#     
#     
#     #test_writer.close()
#     infer_sess.close()
#==============================================================================
