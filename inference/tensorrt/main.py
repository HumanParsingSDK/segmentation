import os

import pycuda.autoinit # this need for CUDA context initialisation

import tensorrt as trt
import pycuda.driver as cuda
import numpy as np
import cv2

import time

from train_config.dataset import Dataset

LOGGER = trt.Logger(trt.Logger.WARNING)
MAX_BATCH_SIZE = 1
IMG_SIZE = 300
MAX_WORKSPACE_SIZE = 1 << 20
INPUT_SHAPE = (3, IMG_SIZE, IMG_SIZE)
DTYPE = trt.float32


def build_engine(model_file):
    print('build engine...')

    builder = trt.Builder(LOGGER)
    network = builder.create_network()
    builder.max_workspace_size = MAX_WORKSPACE_SIZE
    builder.max_batch_size = MAX_BATCH_SIZE
    if DTYPE == trt.float16:
        builder.fp16_mode = True
    parser = trt.OnnxParser(network, LOGGER)
    with open(model_file, 'rb') as model:
        parser.parse(model.read())

    return builder.build_cuda_engine(network)


def allocate_buffers(engine):
    print('allocate buffers')

    h_input = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(0)), dtype=trt.nptype(DTYPE))
    h_output = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(1)), dtype=trt.nptype(DTYPE))
    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)

    return h_input, d_input, h_output, d_output


def load_input(img, host_buffer):
    img_array = np.asarray(np.moveaxis(img.astype(np.float32) / 255., -1, 0)).ravel()
    np.copyto(host_buffer, img_array)


def do_inference(context, h_input, d_input, h_output, d_output):
    # Transfer input data to the GPU.
    cuda.memcpy_htod(d_input, h_input)

    # Run inference.
    context.execute(batch_size=1, bindings=[int(d_input), int(d_output)])

    # Transfer predictions back from the GPU.
    cuda.memcpy_dtoh(h_output, d_output)

    return h_output


if __name__ == '__main__':
    # TODO: add command line args
    engine = build_engine('exp1.onnx')
    h_input, d_input, h_output, d_output = allocate_buffers(engine)

    with engine.create_execution_context() as context:
        total_time = 0

        dataset = Dataset().load_indices(r'../../data/indices/test_indices.npy')

        for i, data in enumerate(dataset):
            img = data['data']
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

            st = time.time()
            load_input(img, h_input)
            output = do_inference(context, h_input, d_input, h_output, d_output)
            output = np.reshape(output, (IMG_SIZE, IMG_SIZE))
            total_time += time.time() - st

        print('FPS: {}'.format(len(dataset) / total_time))
