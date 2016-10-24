#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import os
import os.path as osp
import shutil
import chainer
import argparse
import numpy as np
import cv2
import cPickle as pickle
from VGGNet import VGGNet
from chainer import cuda
from chainer import serializers
from chainer import Variable
from skimage.transform import rotate

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('-i', '--images', nargs='+', required=True)
    args = parser.parse_args()

    gpu = args.gpu
    images = args.images

    mean = np.array([103.939, 116.779, 123.68])

    # Load model
    print('Loading VGG16 net model...')
    vgg = VGGNet()
    serializers.load_hdf5('VGG.model', vgg)
    if gpu >= 0:
        vgg.to_gpu(gpu)
    print('Done.')

    if not osp.exists('out'):
        os.mkdir('out')

    html = [
        '<link href="http://getbootstrap.com/dist/css/bootstrap.min.css" rel=stylesheet>',
        '<div class="container">',
        '<div class="row">'
    ]
    for img_file in images:
        print(img_file)

        img_orig = cv2.imread(img_file)
        img_resized = cv2.resize(img_orig, (224, 224))
        cv2.imwrite(osp.join('out', osp.basename(img_file)), img_resized)

        results = []
        for angle in [0, -90, -180, -270]:
            img = img_resized.copy()
            img = rotate(img, angle, preserve_range=True)
            img = img.astype(np.float32)
            img -= mean
            img = img.transpose((2, 0, 1))
            img = img[np.newaxis, :, :, :]

            if gpu >= 0:
                img = cuda.to_gpu(img, device=gpu)

            pred = vgg(Variable(img), None)

            if gpu >= 0:
                pred = cuda.to_cpu(pred.data)
            else:
                pred = pred.data

            words = open('data/synset_words.txt').readlines()
            words = [(w[0], ' '.join(w[1:])) for w in [w.split() for w in words]]
            words = np.asarray(words)

            top5 = np.argsort(pred)[0][::-1][:5]
            probs = np.sort(pred)[0][::-1][:5]
            # for w, p in zip(words[top5], probs):
            #     print('{}\tprobability:{}'.format(w, p))

            top_label_name = words[top5][0][1]
            top_prob = probs[0]
            result = '{} {}'.format(top_label_name, top_prob)
            results.append(result)
        html.append('''
<div class="col-md-8">
  <ul class="list-group">
    <li class="list-group-item">000: {res1}</li>
    <li class="list-group-item">090: {res2}</li>
    <li class="list-group-item">180: {res3}</li>
    <li class="list-group-item">270: {res4}</li>
  </ul>
</div>
<div class="col-md-4">
  <img src="{file}" style="width: 224px; height: 224px;" />
</div>'''.format(res1=results[0], res2=results[1], res3=results[2], res4=results[3],
                 file=osp.join('out', osp.basename(img_file))))
    html.append('</div>')
    html.append('</div>')

    with open('out.html', 'w') as f:
        f.write('\n'.join(html))
