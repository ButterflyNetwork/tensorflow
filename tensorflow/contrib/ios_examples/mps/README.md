# MPS Test


## Prerequisite
Note that tensorflow should be installed in your python environment first. You should
also be able to compile tensorflow from source on your machine.


# Explanation
This test is used to run tensorflow operations that utilize metal performance shaders.
First, the iOS specific headers must be generated and iOS libraries must be compiled.

Run `./tensorflow/contrib/makefile/build_64_ios.sh` to do that. This builds an
iOS 64bit only version of tensorflow with the MPS operations active.

Next, to run the tests you need to compile the host side MPS operations. They don't
do anything (MPS is available on iOS only) but we use them to generate a graph on
the host that can be computed by the iPhone.

```
./configure
bazel build --config opt //tensorflow/core/user_ops:mps.so
```

Finally open `tensorflow/contrib/ios_examples/mps/mps-test/mps-test.xcodeproj/`,
connect a phone and compile.

Compiling this project will first attempt to recompile tensorflow (useful if
modifications are made to tf code) and then run `prepare_data.py`. This script
generates unit test data for the iPhone. XCode will copy this data to the phone
for the project to execute the test.
