import cv2
import numpy as np
from pietoolbelt.viz import ColormapVisualizer

from train_config.dataset import create_augmented_dataset


if __name__ == '__main__':
    dataset = create_augmented_dataset(is_train=True, to_pytorch=False)
    vis = ColormapVisualizer([0.5, 0.5])

    print(len(dataset))

    for img_idx, d in enumerate(dataset):
        original_img = d['data'].copy()
        print(d['target'].max(), d['target'].shape)
        if d['target'].max() > 0:
            img = vis.process_img(d['data'], (d['target'] * 255).astype(np.uint8))
        else:
            img = d['data']

        # cv2.imwrite('original{}.jpg'.format(img_idx), original_img)
        # cv2.imwrite('img{}.jpg'.format(img_idx), img)
        # cv2.imwrite('mask{}.jpg'.format(img_idx), d['target'].astype(np.float32))

        cv2.imshow('original', original_img)
        cv2.imshow('img', img)
        cv2.imshow('mask', d['target'].astype(np.float32))
        cv2.waitKey()
