### Butterfly specific changes to tensorflow build.

#### Prerequisites

You'll need to install the following libraries:

```bash
$ brew install autoconf
$ brew install automake
$ brew install libtool
```

#### Background 

We are currently building our own tensorflow binary since by default the binary for mobiles do not include some of the ops required by our models.
Tensorflow for ios specifically is compiled using the script 

```
./tensorflow/contrib/makefile/build_all_ios.sh
```
That scipts calls `tensorflow/contrib/makefile/Makefile` which sets the macro `__ANDROID_TYPES_SLIM__` which is used extensively when defining ops.

An op can be excluded from the final mobile binary files for several reasons:

- The op was not registered in `tensorflow/contrib/makefile/tf_op_files.txt`. 
  In that case all we do is add it. e.g.
  ```
  tensorflow/core/kernels/cwise_op_atan2.cc
  ```
  
- For mobile the op is usually registered only for the first type in the list available at the time of registration. 
  e.g. In the follwing case:
```
REGISTER7(BinaryOp, CPU, "Pow", functor::pow, float, Eigen::half, double, int32,
          int64, complex64, complex128);
```

the only operation that will be available on mobile will be "Pow" with type `float` because `float` is the first type mentioned out of 7 types (float, Eigen::half, double, int32,
          int64, complex64, complex128). In case our model needs `int32` we will need to modify the code and specifically add a line that will registered that op for the `int32` type e.g.

```
REGISTER(BinaryOp, CPU, "Pow", functor::pow, int32);
```

You can see an example in the original tensorflow codebase here:
```
https://github.com/tensorflow/tensorflow/blob/79e65acb81f750ffa88b366c566646d48d16c574/tensorflow/core/kernels/cwise_op_mul_1.cc#L23
```


- The op is registered but not available in the binary. 
In that case we need to modify the Makefile to compile the source code for the op.
e.g. We could add the following line in the Makefile in the place where sources are defined to include source code for image_ops._
```
$(wildcard tensorflow/contrib/image/kernels/ops/*.cc)
```

#### How to build

```
sh ./tensorflow/contrib/makefile/build_all_ios.sh
```

```
sh pack_for_bni.sh
```

Change the version in `TensorflowPod.podspec` and create a release.

### Android (WIP)

The script for building android is colocated with the ios script:
```
./tensorflow/tensorflow/contrib/makefile/build_all_android.sh*
```
This script defaults to building for an `armeabi-v7a` architecture.
Butterfly needs to build the following targets:

```
    ./tensorflow/contrib/makefile/build_all_android.sh -a arm64-v8a
    ./tensorflow/contrib/makefile/build_all_android.sh -a x86_64
```

#### NDK

For android, you need to install the NDK. This is the toolchain used for
cross compilation. There are a number
of ways to do this.  The easiest is to install Android Studio.
In instances where you don't want to install the Android SDK,
you can install the Android command line tools
directly using  
```
wget https://dl.google.com/android/repository/commandlinetools-linux-6200805_latest.zip
```

Finally, if you need an older version of the ndk, you can get these 
from [here](https://developer.android.com/ndk/downloads/older_releases).
For instance:
```
wget https://dl.google.com/android/repository/android-ndk-r15c-linux-x86_64.zip
```
One issue with Android: linking requires a mapping of std:: to
 std::__ndk1. For tensorflow, this requires linking against the
 `llvm libc++` family of libraries rather than the default `gcc libstdc++`
 libraries. Supporting changes to build file include paths, library paths,
 and library archives have been made on the branch
 [llvm](https://github.com/ButterflyNetwork/tensorflow/compare/master...ButterflyNetwork:llvm):

#### Butterfly/Tensorflow (~1.13)
 
Because our fork is an older version of 
tensorflow, we need to select a compatible $NDK$ version.
For our branch, the candidate versions
are described 
[here](https://github.com/ButterflyNetwork/tensorflow/blob/5f94511e57d55d6fbe840f117b8fec3f77f6aa44/configure.py#L46)
```
_SUPPORTED_ANDROID_NDK_VERSIONS = [10, 11, 12, 13, 14, 15, 16, 17, 18]
```
Any of these should work. But the makefile seems most compatible with
the `android-ndk-r15c` structure.

I had some problems buiding on OSX, so I spun up a new `dl-android` node
on an m5a.4xlarge
instance running ubuntu 16.04 and set up as follows:
```
    1  sudo apt update
    2  sudo apt upgrade
    3  sudo apt-get install build-essential
    5  sudo apt-get install autoconf automake libtool curl make g++ unzip zlib1g-dev git python
    7  wget https://dl.google.com/android/repository/android-ndk-r15c-linux-x86_64.zip
   10  unzip android-ndk-r15c-linux-x86_64.zip 
   12  export NDK_ROOT=/home/ubuntu/android-ndk-r15c
   13  git clone git@github.com:ButterflyNetwork/tensorflow.git
   26  cd tensorflow/
   29  ./tensorflow/contrib/makefile/build_all_android.sh -a arm64-v8a
```

Running against android-ndk-r15c with architecture `arm64-v8a` builds
the 4 libraries of interest. However, the build script,
 ```
./build_all_android.sh -a arm64-v8a
```
fails when building follow-up(?) benchmarks with what looks like some
namespace related errors. This is worrisome and needs some investigation.
```
/home/ubuntu/tensorflow/tensorflow/contrib/makefile/gen/bin/android_arm64-v8a/benchmark
```


Libraries and headers are installed here:
[software/develop](https://github.com/ButterflyNetwork/software/tree/develop/host/3rdParty/tensorflow-1.13.2)
 
In particular, we have the following static libraries
```
libnsync.a
libprotobuf-lite.a
libprotobuf.a
libtensorflow-core.a
```
