# import tensorflow as tf

# class Logger(object):
#     """TensorBoard logger."""

#     def __init__(self, log_dir):
#         """Initialize summary writer with GPU device."""
#         with tf.device('/GPU:0'):  # Explicitly place the writer on the GPU
#             self.writer = tf.summary.create_file_writer(log_dir)

#     def scalar_summary(self, tag, value, step):
#         """Add scalar summary."""
#         with self.writer.as_default():
#             summary = tf.summary.scalar(tag, value, step=step)
#             self.writer.flush()
import tensorflow as tf


class Logger(object):
    """Tensorboard logger."""

    def __init__(self, log_dir):
        """Initialize summary writer."""
        # self.writer = tf.summary.create_file_writer(log_dir)

    def scalar_summary(self, tag, value, step):
        """Add scalar summary."""
        # with self.writer.as_default():
            # tf.summary.scalar(tag, value, step=step)
