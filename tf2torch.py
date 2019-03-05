import torch
import os
import sys
import tensorflow as tf
import numpy as np
from collections import OrderedDict

from prnet_full import PRNet as PRNetFull
from prnet import PRNet

def main(args):
    sys.path.append(args.prnet_dir) 
    from predictor import resfcn256   # import from prnet_dir, maybe using importlib is a better idea
    tf_network_def = resfcn256(256, 256)
    # tensorflow network forward
    net_input = tf.placeholder(
        tf.float32, shape=[None, 256, 256, 3])
    tf_model = tf_network_def(net_input, is_training=False)
    tf_config = tf.ConfigProto(
        gpu_options=tf.GPUOptions(allow_growth=False))
    tf_config = tf.ConfigProto(
        device_count={"GPU":0}
    )
    sess = tf.Session(config=tf_config)

    saver = tf.train.Saver(tf_network_def.vars)
    saver.restore(
        sess, os.path.join(args.prnet_dir, 'Data', 'net-data', '256_256_resfcn256_weight'))

    graph = sess.graph
    # print([node.name for node in graph.as_graph_def().node])

    torch_model = PRNetFull(3, 3)
    torch_dict = OrderedDict()

    for node in graph.as_graph_def().node:
        if node.name in torch_model.tf_map:
            torch_name = torch_model.tf_map[node.name]
            data = graph.get_operation_by_name(node.name).outputs[0]
            data_np = sess.run(data)
            if len(data_np.shape) > 1:
                # weight layouts  |   tensorflow   |     pytorch     |  transpose   |
                # conv2d_transpose (H, W, out, in) -> (in, out, H, W)  (3, 2, 0, 1)
                # conv2d           (H, W, in, out) -> (out, in, H, W)  (3, 2, 0, 1)
                torch_dict[torch_name] = torch.tensor(np.transpose(data_np, (3, 2, 0, 1)).astype(np.float32))
            else:
                torch_dict[torch_name] = torch.tensor(data_np.astype(np.float32))
        else:
            if node.name.find('save') == -1:
                pass
                # print('not in {}'.format(node.name))
    torch.save(torch_dict, 'from_tf.pth')

    del torch_model
    torch_model = PRNet(3, 3)
    torch_model.load_state_dict(torch_dict)
    torch_model.eval()

    # Test with images
    from skimage.io import imread
    from skimage.transform import resize
    img = imread(os.path.join(args.prnet_dir, 'TestImages','0.jpg')) / 255.
    img_np = resize(img, (256, 256))[np.newaxis, :,:,:]  # simply using resize
    img_bchw = np.transpose(resize(img, (256, 256))[np.newaxis, :,:,:], (0, 3, 1, 2)).astype(np.float32)
    torch_input = torch.from_numpy(img_bchw)
    torch_out = torch_model(torch_input).cpu().detach().numpy()
    torch_out = np.transpose(torch_out, (0, 2, 3, 1)).squeeze()

    net_out = sess.run(tf_model, feed_dict={
        net_input: img_np
    })
    tf_out = net_out.squeeze()
    
    print('shape', tf_out.shape, torch_out.shape)
    print('mse', np.sum(np.square(tf_out - torch_out)))
    print('close',np.allclose(tf_out, torch_out))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--prnet_dir', help='path to prnet repository', required=True)
    main(parser.parse_args())
