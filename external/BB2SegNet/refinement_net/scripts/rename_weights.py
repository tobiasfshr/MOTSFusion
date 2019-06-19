#!/usr/bin/env python
#Adapted from https://gist.github.com/batzner/7c24802dd9c5e15870b4b56e22135c96

import tensorflow as tf


def rename(checkpoint_dir, replace_pairs, add_prefix, exclude, convert_global_step_to_int32=False, dry_run=False):
  checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)
  with tf.Session() as sess:
    for var_name, _ in tf.contrib.framework.list_variables(checkpoint_dir):
      # Load the variable
      var = tf.contrib.framework.load_variable(checkpoint_dir, var_name)

      if var_name == "global_step" and convert_global_step_to_int32:
        var = tf.cast(var, dtype=tf.int32)

      new_name = var_name
      if exclude and var_name not in exclude:
        # Set the new name
        if replace_pairs:
          for replace_pair in replace_pairs:
            new_name = new_name.replace(replace_pair[0], replace_pair[1])
        if add_prefix:
          new_name = add_prefix + new_name

      if dry_run:
        print('%s would be renamed to %s.' % (var_name, new_name))
      else:
        print('Renaming %s to %s.' % (var_name, new_name))
        # Rename the variable
        var = tf.Variable(var, name=new_name)

    if not dry_run:
      # Save the variables
      saver = tf.train.Saver()
      sess.run(tf.global_variables_initializer())
      # saver.save(sess, checkpoint.model_checkpoint_path)
      saver.save(sess, checkpoint_dir)
