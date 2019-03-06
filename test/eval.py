import numpy as np
import torch
import tensorflow as tf
import os
import sys
sys.path.append('.')
from prnet import PRNet


def main(args):
    torch_model = PRNet(3, 3)
    torch_model.load_state_dict(torch.load('from_tf.pth'))
    torch_model.eval()

    sys.path.append(args.prnet_dir)
    from predictor import resfcn256

    tf_network_def = resfcn256(256, 256)

    x = tf.placeholder(
        tf.float32, shape=[None, 256, 256, 3])
    tf_model = tf_network_def(x, is_training=False)

    tf_config = tf.ConfigProto(device_count={'GPU': 0})
    sess = tf.Session(config=tf_config)

    saver = tf.train.Saver(tf_network_def.vars)
    saver.restore(
        sess, os.path.join(args.prnet_dir, 'Data', 'net-data', '256_256_resfcn256_weight'))

    for i in range(args.step):
        # Data
        random_image = np.random.randn(4, 256, 256, 3).astype(np.float32)

        # tf
        feed_dict = {x: random_image}
        tf_out = sess.run(
            tf_model, feed_dict=feed_dict)
        tf_watched_out = tf_out

        # torch
        image_bchw = np.transpose(random_image, (0, 3, 1, 2))

        image_tensor = torch.tensor(image_bchw)
        # torch_watched_out = torch_model.input_conv(image_tensor)
        # torch_watched_out = np.transpose(torch_watched_out.cpu().detach().numpy(), (0, 2, 3, 1))

        torch_out = torch_model(image_tensor)
        torch_out = np.transpose(
            torch_out.cpu().detach().numpy(), (0, 2, 3, 1))
        torch_watched_out = torch_out

        print('step {}| is_close: {}| mse: {:.4f}'.format(
            i, np.allclose(tf_watched_out, torch_watched_out, rtol=1e-4, atol=1e-5),
            np.sum(np.square(tf_watched_out - torch_watched_out))))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--prnet_dir')
    parser.add_argument(
        '--step', help='number of steps to run', type=int, default=50)
    main(parser.parse_args())
