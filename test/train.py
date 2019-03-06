import numpy as np
import torch
import tensorflow as tf
import os
import sys
sys.path.append('.')
from prnet import PRNet


def main(args):
    TRAIN_CONFIG = {
        'learning_rate': 1e-3,
    }
    torch_model = PRNet(3, 3)
    torch_model.load_state_dict(torch.load('from_tf.pth'))
    torch_model.train()
    torch_optimizer = torch.optim.SGD(torch_model.parameters(),
                                      lr=TRAIN_CONFIG['learning_rate'],
                                      weight_decay=0.0002  # equalivent to tcl.L2_regularizer)
                                      )
    torch_mse = torch.nn.MSELoss()

    sys.path.append(args.prnet_dir)
    from predictor import resfcn256

    def tf_train_def(loss_val, var_list, train_config):
        lr = train_config['learning_rate']
        global_step = tf.Variable(0, trainable=False)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
        grads = optimizer.compute_gradients(loss_val, var_list=var_list)
        return optimizer.apply_gradients(grads, global_step=global_step), global_step

    tf_network_def = resfcn256(256, 256)

    x = tf.placeholder(
        tf.float32, shape=[None, 256, 256, 3])
    y_ = tf.placeholder(
        tf.float32, shape=[None, 256, 256, 3]
    )
    tf_model = tf_network_def(x, is_training=True)
    tf_loss = tf.square(y_ - tf_model)
    tf_loss = tf.reduce_mean(tf_loss)
    tf_trainable_var = tf.trainable_variables()
    tf_train_op, tf_global_step_op = tf_train_def(
        tf_loss, tf_trainable_var, TRAIN_CONFIG)
    tf_watched_op = tf.get_default_graph().get_operation_by_name(
        'resfcn256/Conv/Relu').outputs[0]

    tf_config = tf.ConfigProto(device_count={'GPU': 0})
    sess = tf.Session(config=tf_config)
    init = tf.global_variables_initializer()
    sess.run(init)

    saver = tf.train.Saver(tf_network_def.vars)
    saver.restore(
        sess, os.path.join(args.prnet_dir, 'Data', 'net-data', '256_256_resfcn256_weight'))

    # Data
    random_image = np.random.randn(4, 256, 256, 3).astype(np.float32)
    random_label = np.random.randn(4, 256, 256, 3).astype(np.float32)

    for i in range(args.step):
        # tf
        feed_dict = {x: random_image,
                     y_: random_label}
        tf_out, _, tf_train_loss, tf_watched_out = sess.run(
            [tf_model, tf_train_op, tf_loss, tf_watched_op], feed_dict=feed_dict)
        tf_watched_out = tf_out

        # torch
        image_bchw = np.transpose(random_image, (0, 3, 1, 2))
        label_bchw = np.transpose(random_label, (0, 3, 1, 2))

        image_tensor = torch.tensor(image_bchw)
        label_tensor = torch.tensor(label_bchw)
        # torch_watched_out = torch_model.input_conv(image_tensor)
        # torch_watched_out = np.transpose(torch_watched_out.cpu().detach().numpy(), (0, 2, 3, 1))

        torch_out = torch_model(image_tensor)
        torch_train_loss = torch_mse(torch_out, label_tensor)
        torch_out = np.transpose(
            torch_out.cpu().detach().numpy(), (0, 2, 3, 1))
        torch_watched_out = torch_out

        torch_optimizer.zero_grad()
        torch_train_loss.backward()
        torch_optimizer.step()
        torch_train_loss = torch_train_loss.item()

        print('step {}| is_close: {}| mse: {:.4f}| loss {:.6f}/{:.6f}'.format(i, np.allclose(tf_watched_out, torch_watched_out,
                                                                                            atol=1e-3, rtol=1e-3), np.sum(np.square(tf_watched_out - torch_watched_out)), tf_train_loss, torch_train_loss))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--prnet_dir')
    parser.add_argument('--step', help='number of steps to run', type=int, default=50)
    main(parser.parse_args())
