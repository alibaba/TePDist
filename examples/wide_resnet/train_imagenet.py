from resnet import *
from tensorflow.core.protobuf import rewriter_config_pb2
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('batch_size', 16, "batch size")
tf.app.flags.DEFINE_integer('input_size', 224, "max steps")
tf.app.flags.DEFINE_integer('max_steps', 500000, "max steps")
tf.app.flags.DEFINE_integer('model_type', 0, "type of wresnet model")

tf.compat.v1.enable_resource_variables()

wresnet_specs = [
  # C   W   blocks
  [160, 2, [3, 4, 6, 3]], #250MB (50 layers)
  [224, 2, [3, 4, 6, 3]], #500MB (50 layers)
  [320, 2, [3, 4, 6, 3]], #1B (50 layers)
  [448, 2, [3, 4, 6, 3]], #2B (50 layers)
  [640, 2, [3, 4, 6, 3]], #4B (50 layers)
  [320, 16, [3, 4, 6, 3]], #6.8B (50 layers)
  [320, 12, [3, 4, 23, 3]], #13B (501 layers)
]

def fake_inputs():
    image = tf.constant(0.5, shape=[FLAGS.input_size, FLAGS.input_size, 3])
    label = tf.constant(1, shape=[], dtype=tf.int64)
    dataset = tf.data.Dataset.from_tensors((image, label)).repeat(FLAGS.batch_size * 100)
    dataset = dataset.batch(FLAGS.batch_size)
    return dataset

def model(features, labels, mode, params):
    logits = inference(features,
                      num_classes=1000,
                      num_filters=wresnet_specs[FLAGS.model_type][0],
                      width_factor=wresnet_specs[FLAGS.model_type][1],
                      is_training=True,
                      bottleneck=True,
                      num_blocks=wresnet_specs[FLAGS.model_type][2])
    loss_batch = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    loss = tf.reduce_mean(loss_batch)

    optimizer = tf.train.AdamOptimizer(0.1)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    train_op = tf.group([train_op])
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

def main(_):
    session_config=tf.ConfigProto(
    log_device_placement=False,
    allow_soft_placement=True,
    intra_op_parallelism_threads=64,
    inter_op_parallelism_threads=64,
    gpu_options=tf.GPUOptions(allow_growth=True,
                              force_gpu_compatible=True,
                              per_process_gpu_memory_fraction=1.0))

    on = rewriter_config_pb2.RewriterConfig.ON
    off = rewriter_config_pb2.RewriterConfig.OFF
    session_config.graph_options.rewrite_options.init_from_remote = on
    session_config.graph_options.rewrite_options.remapping = off
    session_config.graph_options.rewrite_options.constant_folding = off

    run_config = tf.estimator.RunConfig(
        model_dir="./ckpt/",
        session_config=session_config,
        tf_random_seed=123123,
        # Disable checkpoints temporarily.
        save_checkpoints_steps=None,
        save_checkpoints_secs=None,
        log_step_count_steps=1
    )

    network = tf.estimator.Estimator(
        model_fn=model,
        config=run_config)

    network.train(input_fn=fake_inputs)

if __name__ == '__main__':
    tf.app.run()
