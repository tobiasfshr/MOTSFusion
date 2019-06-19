from refinement_net.scripts.rename_weights import rename
import tensorflow as tf
import numpy as np


def switch_initial_conv_order(checkpoint_dir, prefix):
  checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)
  with tf.Session() as sess:
    for var_name, _ in tf.contrib.framework.list_variables(checkpoint_dir):
      # Load the variable
      var = tf.contrib.framework.load_variable(checkpoint_dir, var_name)

      if var_name == prefix + "/conv0/W":
        var = np.flip(var, 2)

      var = tf.Variable(var, name=var_name)

    # Save the variables
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    saver.save(sess, checkpoint_dir)


def prepare_resnet_checkpoint():
  rename("models/resnet_test/resnet_test-1",
         [("bn/mean/EMA", "bn/mean_ema"), ("bn/variance/EMA", "bn/var_ema")], "resnet/",
         ["global_step", "learning_rate"], convert_global_step_to_int32=True, dry_run=False)


def prepare_frcnn_checkpoint():
  rename("models/frcnn_test/frcnn_test-1",
         [("bn/mean/EMA", "bn/mean_ema"), ("bn/variance/EMA", "bn/var_ema")], "frcnn/",
         ["global_step", "learning_rate"], convert_global_step_to_int32=True, dry_run=False)
  switch_initial_conv_order("models/frcnn_test/frcnn_test-1", "frcnn")


if __name__ == '__main__':
  prepare_frcnn_checkpoint()
