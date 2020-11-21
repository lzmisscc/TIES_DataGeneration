import argparse
import random
from threading import Thread
from PIL import Image
from table import MyTable as Table
import pickle
import string
import os
import cv2
import traceback
import numpy as np
import tensorflow as tf
import warnings
import jsonlines
import time

warnings.filterwarnings("ignore")


Reader = jsonlines.open("pubtabnet/PubTabNet_2.0.0.jsonl", "r").iter()



class Logger:
    def __init__(self):
        pass
        # self.file=open('logtxt.txt','a+')

    def write(self, txt):
        file = open('logfile.txt', 'a+')
        file.write(txt)
        file.close()


class GenerateTFRecord:
    def __init__(self, outpath, filesize, visualizebboxes):
        self.outtfpath = outpath  # directory to store tfrecords
        self.filesize = filesize  # number of images in each tfrecord
        self.logger = Logger()  # if we want to use logger and store output to file
        self.num_of_max_vertices = 900
        self.max_length_of_word = 1000  # max possible length of each word
        # minimum number of rows in a table (includes headers)
        self.row_min = 3
        self.row_max = 15  # maximum number of rows in a table
        self.col_min = 3  # minimum number of columns in a table
        self.col_max = 9  # maximum number of columns in a table
        self.minshearval = -0.1  # minimum value of shear to apply to images
        self.maxshearval = 0.1  # maxmimum value of shear to apply to images
        self.minrotval = -0.01  # minimum rotation applied to images
        self.maxrotval = 0.01  # maximum rotation applied to images
        self.num_data_dims = 5  # data dimensions to store in tfrecord
        self.max_height = 768  # max image height
        self.max_width = 1366  # max image width
        self.visualizebboxes = visualizebboxes


    def create_dir(self, fpath):  # creates directory fpath if it does not exist
        if(not os.path.exists(fpath)):
            os.mkdir(fpath)

    def str_to_int(self, str):  # converts each character in a word to equivalent int
        intsarr = np.array([ord(chr) for chr in str])
        padded_arr = np.zeros(shape=(self.max_length_of_word), dtype=np.int64)
        padded_arr[:len(intsarr)] = intsarr
        return padded_arr

    def convert_to_int(self, arr):  # simply converts array to a string
        return [int(val) for val in arr]

    # will pad the input array with zeros to make it equal to 'shape'
    def pad_with_zeros(self, arr, shape):
        dummy = np.zeros(shape, dtype=np.int64)
        dummy[:arr.shape[0], :arr.shape[1]] = arr
        return dummy

    def generate_tf_record(self, im, cellmatrix, rowmatrix, colmatrix, arr, tablecategory, imgindex, output_file_name):
        '''This function generates tfrecord files using given information'''
        cellmatrix = self.pad_with_zeros(
            cellmatrix, (self.num_of_max_vertices, self.num_of_max_vertices))
        colmatrix = self.pad_with_zeros(
            colmatrix, (self.num_of_max_vertices, self.num_of_max_vertices))
        rowmatrix = self.pad_with_zeros(
            rowmatrix, (self.num_of_max_vertices, self.num_of_max_vertices))

        im = im.astype(np.int64)
        img_height, img_width = im.shape

        words_arr = arr[:, 1].tolist()
        no_of_words = len(words_arr)

        lengths_arr = self.convert_to_int(arr[:, 0])
        vertex_features = np.zeros(
            shape=(self.num_of_max_vertices, self.num_data_dims), dtype=np.int64)
        lengths_arr = np.array(lengths_arr).reshape(len(lengths_arr), -1)
        sample_out = np.array(np.concatenate(
            (arr[:, 2:], lengths_arr), axis=1))
        vertex_features[:no_of_words, :] = sample_out

        if(self.visualizebboxes):
            self.draw_matrices(
                im, arr, [rowmatrix, colmatrix, cellmatrix], imgindex, output_file_name)

        vertex_text = np.zeros(
            (self.num_of_max_vertices, self.max_length_of_word), dtype=np.int64)
        vertex_text[:no_of_words] = np.array(
            list(map(self.str_to_int, words_arr)))

        feature = dict()
        feature['image'] = tf.train.Feature(
            float_list=tf.train.FloatList(value=im.astype(np.float32).flatten()))
        feature['global_features'] = tf.train.Feature(float_list=tf.train.FloatList(value=np.array(
            [img_height, img_width, no_of_words, tablecategory]).astype(np.float32).flatten()))
        feature['vertex_features'] = tf.train.Feature(float_list=tf.train.FloatList(
            value=vertex_features.astype(np.float32).flatten()))
        feature['adjacency_matrix_cells'] = tf.train.Feature(
            int64_list=tf.train.Int64List(value=cellmatrix.astype(np.int64).flatten()))
        feature['adjacency_matrix_cols'] = tf.train.Feature(
            int64_list=tf.train.Int64List(value=colmatrix.astype(np.int64).flatten()))
        feature['adjacency_matrix_rows'] = tf.train.Feature(
            int64_list=tf.train.Int64List(value=rowmatrix.astype(np.int64).flatten()))
        feature['vertex_text'] = tf.train.Feature(
            int64_list=tf.train.Int64List(value=vertex_text.astype(np.int64).flatten()))

        all_features = tf.train.Features(feature=feature)

        seq_ex = tf.train.Example(features=all_features)
        return seq_ex

    def generate_tables(self, *args, N_imgs):

        all_table_categories = [0, 0, 0, 0]
        data_arr = []
        while len(data_arr)!=N_imgs:
            gt = Reader.__next__()
            try:
                print(gt['filename'])

                table = Table(gt=gt)
                draw = Image.new(size=(self.max_width, self.max_height), mode='RGB', color=(255,255,255))
                same_cell_matrix, same_col_matrix, same_row_matrix, id_count, html_content, tablecategory, im, bboxes = table.create()
                W, H = im.size
                if W > self.max_width or H > self.max_height:
                    continue
            except Exception as E:
                print("ERRO:", gt['filename'])
                continue
            draw.paste(im, [0,0]+list(im.size))
            data_arr.append([[same_row_matrix, same_col_matrix,
                            same_cell_matrix, bboxes, [tablecategory]], [draw]])

        return data_arr, all_table_categories

    def draw_matrices(self, img, arr, matrices, imgindex, output_file_name):
        '''Call this fucntion to draw visualizations of a matrix on image'''
        no_of_words = len(arr)
        colors = np.random.randint(0, 255, (no_of_words, 3))
        arr = arr[:, 2:]

        img = img.astype(np.uint8)
        img = np.dstack((img, img, img))

        mat_names = ['row', 'col', 'cell']
        output_file_name = output_file_name.replace('.tfrecord', '')

        for matname, matrix in zip(mat_names, matrices):
            im = img.copy()
            x = 1
            indices = np.argwhere(matrix[x] == 1)
            for index in indices:
                cv2.rectangle(im, (int(arr[index, 0])-3, int(arr[index, 1])-3),
                              (int(arr[index, 2])+3, int(arr[index, 3])+3),
                              (0, 255, 0), 1)

            x = 4
            indices = np.argwhere(matrix[x] == 1)
            for index in indices:
                cv2.rectangle(im, (int(arr[index, 0])-3, int(arr[index, 1])-3),
                              (int(arr[index, 2])+3, int(arr[index, 3])+3),
                              (0, 0, 255), 1)

            img_name = os.path.join(
                'bboxes/', output_file_name+'_'+str(imgindex)+'_'+matname+'.jpg')
            cv2.imwrite(img_name, im)

    def write_tf(self, filesize, threadnum):
        '''This function writes tfrecords. Input parameters are: filesize (number of images in one tfrecord), threadnum(thread id)'''
        options = tf.compat.v1.io.TFRecordOptions(
            tf.compat.v1.io.TFRecordCompressionType.GZIP)

        if True:

            # randomly select a name of length=20 for tfrecords file.
            output_file_name = ''.join(random.choices(
                string.ascii_uppercase + string.digits, k=20)) + '.tfrecord'
            print('\nThread: ', threadnum, ' Started:', output_file_name)

            data_arr, all_table_categories = self.generate_tables(
                N_imgs=filesize,)

            if(data_arr is not None):
                if(len(data_arr) == filesize):
                    with tf.io.TFRecordWriter(os.path.join(self.outtfpath, output_file_name), options=options) as writer:
                        for imgindex, subarr in enumerate(data_arr):
                            arr = subarr[0]

                            img = np.asarray(subarr[1][0], np.int64)[:, :, 0]
                            colmatrix = np.array(arr[1], dtype=np.int64)
                            cellmatrix = np.array(arr[2], dtype=np.int64)
                            rowmatrix = np.array(arr[0], dtype=np.int64)
                            bboxes = np.array(arr[3])
                            tablecategory = arr[4][0]
                            seq_ex = self.generate_tf_record(
                                img, cellmatrix, rowmatrix, colmatrix, bboxes, tablecategory, imgindex, output_file_name)
                            writer.write(seq_ex.SerializeToString())


    def write_to_tf(self, threadnum):
        '''This function starts tfrecords generation with number of threads = max_threads with each thread
        working on a single tfrecord'''

        from threading import Thread

        # create all directories here
        if(self.visualizebboxes):
            self.create_dir('bboxes')

        # create output directory if it does not exist
        self.create_dir(self.outtfpath)

        starttime = time.time()
        self.write_tf(self.filesize, threadnum=threadnum)

        # nums = 5
        # pools = []
        # for i in range(nums):
        #     p = Thread(target=self.write_tf, args=(self.filesize, i,))
        #     pools.append(p)
        #     p.start()
        #     self.write_tf(self.filesize, threadnum=1)
        # for p in pools:
        #     p.join()
        print(time.time()-starttime)


if __name__ == "__main__":
    outpath,filesize,visualizebboxes = 'tfrecords/', 1000, 'bboxes'
    t = GenerateTFRecord(outpath=outpath,filesize=filesize,visualizebboxes=visualizebboxes,)
    for i in range(10):
        t.write_to_tf(i)