from pathlib import Path
import argparse
import fnmatch
from functools import partial
import numpy as np
import os
import shutil
import subprocess
import tensorflow as tf
from tensorflow.python.ops import nn_ops

# This script generates input files and expected result files
# for the conv2d operation. The files are copied to a
# location where they are automatically picked up by XCode and
# deployed to the iPhone. A corresponding unit test in the
# mps-test project verifies correctness of the conv2d MPS
# implementation. This works only on an iPhone (no simulator).

def get_script_dir():
    return os.path.dirname(os.path.realpath(__file__))

def prepare_TestReadNumpyFileImpl():
    data = np.arange(start=0, stop=16, dtype=np.float32)
    data = np.reshape(data, [2,4,2])
    np.save(get_script_dir() + "/test_numpy_input.npy", data)

def remove_temporary_files(tmpdir):
    files = os.listdir(tmpdir)
    for f in files:
        if f.endswith(".npy") or f.endswith('_graph.pb'):
            os.remove(os.path.join(tmpdir, f))

def prepare_TestMPSConv2d():
    # Load module.
    so_path = (get_script_dir() + "/../../../../../" +
        "bazel-bin/tensorflow/core/user_ops/mps.so")
    assert os.path.isfile(so_path), \
            "Please build mps operation with\n" \
            "./configure && " \
            "bazel build --config opt //tensorflow/core/user_ops:mps.so\n" \
            "from tensorflow root"
    mps_module = tf.load_op_library(so_path)

    # Build and save a smiple graph to run on iOS.
    # -----

    session = tf.InteractiveSession()
    image_in_ = tf.placeholder(tf.float32, name='image_in')
    filter_in_ = tf.placeholder(tf.float32, name='filter_in')

    conv = mps_module.conv2dmps(
        image_in_,
        filter_in_,
        bias=[],
        strides=[1,1,1,1],
        padding='VALID',
        data_format='NHWC'
    )
    ident = tf.identity(conv, name="filtered_output")


    init = tf.global_variables_initializer()
    session.run(init)

    # as_text=False is important to allow loading of graph in C++/iOS.
    tf.train.write_graph(session.graph_def, '',
        "{0}/conv2d_graph.pb".format(get_script_dir()), as_text=False)

    # Run the original conv2d operation to generate gold data for iOS test.
    # -----

    class Conv2DTestData:
        def __init__(self,
                input_batch,
                input_depth,
                input_cols,
                input_rows,
                output_depth,
                filters_generator
            ):
            self.input_batch = input_batch
            self.input_depth = input_depth
            self.input_cols = input_cols
            self.input_rows = input_rows
            self.output_depth = output_depth

            self.filters = filters_generator(input_depth, output_depth)

        def get_images(self):
            input_numel = (self.input_batch *
                self.input_depth *
                self.input_cols *
                self.input_rows
            )

            # To allow correctness test keep values integer and small.
            return np.random.randint(
                        0, 5, input_numel
                    ).reshape(
                        self.input_batch,
                        self.input_rows,
                        self.input_cols,
                        self.input_depth
                    ).astype(np.float32)

        def get_filters(self):
            return self.filters

    def generate_identity_filter(filter_rows,
            filter_cols,
            weight,
            input_depth,
            output_depth):
        f = np.ones([
            filter_rows,
            filter_cols,
            input_depth,
            output_depth
        ]).astype(np.float32)

        f[filter_rows//2, filter_cols//2, input_depth//2, :] = weight
        return f

    # Define test vectors.
    # -----
    tests = [
        # Very simple identity.
        Conv2DTestData(
            1, 1, 5, 5, 1,
            partial(generate_identity_filter, 3, 3, 2)
        ),
        # Non-square identity.
        Conv2DTestData(
            1, 1, 5, 20, 1,
            partial(generate_identity_filter, 3, 3, 2)
        ),
        # Non-square identity.
        Conv2DTestData(
            1, 1, 20, 5, 1,
            partial(generate_identity_filter, 3, 3, 2)
        ),
        # Non-square identity, depth = 2.
        Conv2DTestData(
            1, 2, 5, 5, 2,
            partial(generate_identity_filter, 3, 3, 1)
        ),
        # Non-square identity, depth = 3.
        Conv2DTestData(
            1, 3, 5, 5, 3,
            partial(generate_identity_filter, 3, 3, 1)
        ),
        # Non-square identity, depth = 4.
        Conv2DTestData(
            1, 4, 5, 5, 4,
            partial(generate_identity_filter, 3, 3, 1)
        ),
        # Non-square identity, depth = 5.
        Conv2DTestData(
            1, 5, 3, 3, 5,
            partial(generate_identity_filter, 3, 3, 1)
        ),
        # Very simple identity, batched.
        Conv2DTestData(
            2, 1, 5, 5, 1,
            partial(generate_identity_filter, 3, 3, 2)
        ),
        # Very simple identity, batched, depth = 2.
        Conv2DTestData(
            2, 2, 5, 5, 2,
            partial(generate_identity_filter, 3, 3, 2)
        ),
        # Very simple identity, batched, depth = 5.
        Conv2DTestData(
            2, 5, 5, 5, 5,
            partial(generate_identity_filter, 3, 3, 2)
        ),
        # Very simple identity, batched, input depth = 5, output depth = 10.
        Conv2DTestData(
            2, 5, 5, 5, 10,
            partial(generate_identity_filter, 3, 3, 2)
        ),
        # Data for performance measurement.
        Conv2DTestData(
            1, 64, 128, 128, 128,
            partial(generate_identity_filter, 3, 3, 2)
        ),

    ]

    test_id = 0
    for test in tests:
        images = test.get_images()
        filters = test.get_filters()

        with tf.Session():
            gold_output = nn_ops.conv2d(
                images,
                filters,
                strides=[1,1,1,1],
                padding='VALID',
            ).eval()

        np.save("{0}/conv2d_image_input{1}.npy".format(
                get_script_dir(),
                test_id),
            images
        )
        np.save("{0}/conv2d_filter_input{1}.npy".format(
                get_script_dir(),
                test_id),
            filters
        )
        np.save("{0}/conv2d_expected_result{1}.npy".format(
                get_script_dir(),
                test_id),
              gold_output
              #gold_output.transpose(0,3,1,2)
        )
        test_id += 1

def clean_target(full_target_dir):
    """Remove and recreate target dir."""
    if full_target_dir.is_dir():
        shutil.rmtree(str(full_target_dir))
    full_target_dir.mkdir(parents=True)

def get_rsync_base_command(verbose):
    # Return rsync base command.
    cmd = 'rsync'
    args = '-pvtrL'
    if not verbose:
        args = args + 'q'
    return [cmd, args]


def rsync(source, destination, verbose):
    """Run rsync, sync from source to destination."""
    rsync_command = (get_rsync_base_command(verbose) +
                     [str(source), str(destination)])
    subprocess.call(rsync_command)


def copy_files(full_target_dir, verbose=True):
    for file in os.listdir('.'):
        if fnmatch.fnmatch(file, '*.npy') or fnmatch.fnmatch(file, '*.pb'):
            rsync(file, full_target_dir, verbose)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Deploy tensorflow test input files to iOS device.'
    )
    parser.add_argument('--target_dir', help='Target directory.')
    args = parser.parse_args()

    full_target_dir = Path(os.environ.get('BUILT_PRODUCTS_DIR', '')) / \
                       os.environ.get('CONTENTS_FOLDER_PATH', '') / \
                       'Library' / \
                       'Application Support' / \
                       args.target_dir

    remove_temporary_files(get_script_dir())

    prepare_TestReadNumpyFileImpl()
    prepare_TestMPSConv2d()

    clean_target(full_target_dir)
    copy_files(full_target_dir)
