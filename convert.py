#!/usr/bin/env python

import os
import sys
import numpy as np
from caffe import CaffeError
from caffe.tensorflow import TensorFlowTransformer

def main():
    args = sys.argv[1:]
    if len(args) not in (3, 4):
        print('usage: %s path.prototxt path.caffemodel data-output-path [code-output-path.py]'%os.path.basename(__file__))
        exit(-1)
    def_path, data_path, data_out_path = args[:3]
    src_out_path = args[3] if len(args)==4 else None
    try:
        transformer = TensorFlowTransformer(def_path, data_path)
        print('Converting data...')
        data = transformer.transform_data()
        print('Saving data...')
        with open(data_out_path, 'wb') as data_out:
            np.save(data_out, data)
        if src_out_path is not None:
            print('Saving source...')
            with open(src_out_path, 'wb') as src_out:
                src_out.write(transformer.transform_source())
        print('Done.')
    except KaffeError as err:
        print('Error encountered: %s'%err)
        exit(-1)

if __name__ == '__main__':
    main()
