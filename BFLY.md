# TF2 and TFLite
A this fork contains custom TF2/TFLite code needed
by the DL team for iOS model support. See the TF2 online
documentation for bazel related targets and build info. In particular, you'll need
to set up a python environment and run ./configure from the command line to stage your build.
For Android, we use docker tools available from google and just volume in our code (nice!). We run these
on a machine we spin-up for this purpose, `dl-android`

So far, our principal additions have been a port of TF-Addon code to
tflite. Likely we'll have additional customizations as well.

## iOS

The principal iOS target here would be

    //tensorflow/lite/ios:TensorFlowLiteC_framework

This is a c library.  Make sure to select iOS support in your initial run of `./configure`

## Python & OSX 

For OSX and python, we need a wheel when converting saved_models to tflite models.  The build target is

    $ bazel build --config=noaws --config=nogcp --config=nohdfs --config=nonccl  //tensorflow/tools/pip_package:build_pip_package

To build the package afterwards: (nightly flag needed for, ah, nightly builds... not sure what it does except change
package names... Should not be used for master builds.) 

    $ ./bazel-bin/tensorflow/tools/pip_package/build_pip_package --nightly_flag /tmp/tensorflow_pkg

One pickle here - on OSX you need to rename the resulting package 
(e.g. `cp tf_nightly-2.7.0-cp38-cp38-macosx_11_0_x86_64.whl tf_nightly-2.7.0-cp38-cp38-macosx_10_15_x86_64.whl`)
as the version "11_0" is not supported - I just dummy this in with a version that is supported. 

### Wheels
As a convenience, wheels are stored on data2:
```
ben@cortex:/data2/tensorflow-custom$ tree
.
└── nightly+
    ├── linux
    │   ├── 64f9ccdfef4ac5b4044bee0b5ef4c8f3a2168a97.whl
    │   └── tf_nightly-2.7.0-cp38-cp38-linux_x86_64.whl -> 64f9ccdfef4ac5b4044bee0b5ef4c8f3a2168a97.whl
    └── osx
        ├── 64f9ccdfef4ac5b4044bee0b5ef4c8f3a2168a97.whl
        └── tf_nightly-2.7.0-cp38-cp38-macosx_11_0_x86_64.whl -> 64f9ccdfef4ac5b4044bee0b5ef4c8f3a2168a97.whl

```


## Android
We build out of docker by voluming in our code. To launch:
    
    docker run -it -v $PWD:/host_dir tflite-builder bash

If the container is still around, it's easier to just start it and attach:

```
ubuntu@ip-10-0-6-111:~$ docker container ls -a
CONTAINER ID   IMAGE            COMMAND   CREATED       STATUS                      PORTS     NAMES
1305b9415342   tflite-builder   "bash"    6 weeks ago   Exited (0) 17 seconds ago             awesome_robinson
```

    docker start 1305b9415342
    docker attach 1305b9415342

Here we build two targets:

    bazel build --config=android_arm64 --config=noaws --config=nogcp --config=nohdfs --config=nonccl  //tensorflow/lite/c:tensorflowlite_c
    cp bazel-bin/tensorflow/lite/c/libtensorflowlite_c.so libtensorflowlite_c_android_arm_64.so
    bazel build --config=android_x86_64 --config=noaws --config=nogcp --config=nohdfs --config=nonccl  //tensorflow/lite/c:tensorflowlite_c
    cp bazel-bin/tensorflow/lite/c/libtensorflowlite_c.so libtensorflowlite_c_android_x86_64.so


# Branches
We have the following branches
* tf1 - this is the historical 1.13.x branch used in the early days (and still, now) for development and the phone
* nightly+ - this is the upstream nightly build plus custom changes we have made to support things like connected
  components
* master - this is a release build (currently v2.5.0_rc2) plus our changes.

## History
**Note**: branch `master` has moved to a tf2;
 tag v2.5.0_rc2 as of this note, but overtime it will evolve
 as new releases are made.

The old TF1 `master` branch, in production as of early 2021Q2 has
been moved to the branch `tf1`.  This legacy branch, like
`master`, has been
setup as a protected branch in github, and requires PR
review prior to submission.
