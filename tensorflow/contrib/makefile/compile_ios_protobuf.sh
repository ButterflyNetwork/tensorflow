#!/bin/bash -x -e
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# Builds protobuf 3 for iOS.

SCRIPT_DIR=$(dirname $0)
source "${SCRIPT_DIR}/build_helper.subr"

cd tensorflow/contrib/makefile

HOST_GENDIR="$(pwd)/gen/protobuf-host"
mkdir -p "${HOST_GENDIR}"
if [[ ! -f "./downloads/protobuf/autogen.sh" ]]; then
    echo "You need to download dependencies before running this script." 1>&2
    echo "tensorflow/contrib/makefile/download_dependencies.sh" 1>&2
    exit 1
fi

JOB_COUNT="${JOB_COUNT:-$(get_job_count)}"

GENDIR=`pwd`/gen/protobuf_ios/
LIBDIR=${GENDIR}lib
mkdir -p ${LIBDIR}

OSX_VERSION=darwin14.0.0

IPHONEOS_PLATFORM=`xcrun --sdk iphoneos --show-sdk-platform-path`
IPHONEOS_SYSROOT=`xcrun --sdk iphoneos --show-sdk-path`
IPHONESIMULATOR_PLATFORM=`xcrun --sdk iphonesimulator --show-sdk-platform-path`
IPHONESIMULATOR_SYSROOT=`xcrun --sdk iphonesimulator --show-sdk-path`
IOS_SDK_VERSION=`xcrun --sdk iphoneos --show-sdk-version`
MIN_SDK_VERSION=8.2

CFLAGS="-DNDEBUG -Os -pipe -fPIC -fno-exceptions"
CXXFLAGS="${CFLAGS} -std=c++11 -stdlib=libc++"
LDFLAGS="-stdlib=libc++"
LIBS="-lc++ -lc++abi"

cd downloads/protobuf
PROTOC_PATH="${HOST_GENDIR}/bin/protoc"
if [[ ! -f "${PROTOC_PATH}" || ${clean} == true ]]; then
  # Try building compatible protoc first on host
  echo "protoc not found at ${PROTOC_PATH}. Build it first."
  echo make_host_protoc "${HOST_GENDIR}"
else
  echo "protoc found. Skip building host tools."
fi

#./autogen.sh
if [ $? -ne 0 ]
then
  echo "./autogen.sh command failed."
  exit 1
fi

if [[ ! " ${ARCHITECTURES[@]} " =~ " i368 " ]]; then
    echo make distclean
    echo ./configure \
    --host=i386-apple-${OSX_VERSION} \
    --disable-shared \
    --enable-cross-compile \
    --with-protoc="${PROTOC_PATH}" \
    --prefix=${LIBDIR}/ios_386 \
    --exec-prefix=${LIBDIR}/ios_386 \
    "CFLAGS=${CFLAGS} \
    -mios-simulator-version-min=${MIN_SDK_VERSION} \
    -arch i386 \
    -fembed-bitcode \
    -isysroot ${IPHONESIMULATOR_SYSROOT}" \
    "CXX=${CXX}" \
    "CXXFLAGS=${CXXFLAGS} \
    -mios-simulator-version-min=${MIN_SDK_VERSION} \
    -arch i386 \
    -fembed-bitcode \
    -isysroot \
    ${IPHONESIMULATOR_SYSROOT}" \
    LDFLAGS="-arch i386 \
    -fembed-bitcode \
    -mios-simulator-version-min=${MIN_SDK_VERSION} \
    ${LDFLAGS} \
    -L${IPHONESIMULATOR_SYSROOT}/usr/lib/ \
    -L${IPHONESIMULATOR_SYSROOT}/usr/lib/system" \
    "LIBS=${LIBS}"
    echo make -j"${JOB_COUNT}"
    echo make install
fi

if [[ ! " ${ARCHITECTURES[@]} " =~ " x86_64 " ]]; then
    echo make distclean
    echo ./configure \
    --host=x86_64-apple-${OSX_VERSION} \
    --disable-shared \
    --enable-cross-compile \
    --with-protoc="${PROTOC_PATH}" \
    --prefix=${LIBDIR}/ios_x86_64 \
    --exec-prefix=${LIBDIR}/ios_x86_64 \
    "CFLAGS=${CFLAGS} \
    -mios-simulator-version-min=${MIN_SDK_VERSION} \
    -arch x86_64 \
    -fembed-bitcode \
    -isysroot ${IPHONESIMULATOR_SYSROOT}" \
    "CXX=${CXX}" \
    "CXXFLAGS=${CXXFLAGS} \
    -mios-simulator-version-min=${MIN_SDK_VERSION} \
    -arch x86_64 \
    -fembed-bitcode \
    -isysroot \
    ${IPHONESIMULATOR_SYSROOT}" \
    LDFLAGS="-arch x86_64 \
    -fembed-bitcode \
    -mios-simulator-version-min=${MIN_SDK_VERSION} \
    ${LDFLAGS} \
    -L${IPHONESIMULATOR_SYSROOT}/usr/lib/ \
    -L${IPHONESIMULATOR_SYSROOT}/usr/lib/system" \
    "LIBS=${LIBS}"
    echo make -j"${JOB_COUNT}"
    echo make install
fi

if [[ ! " ${ARCHITECTURES[@]} " =~ " arm7 " ]]; then
    echo make distclean
    echo ./configure \
    --host=armv7-apple-${OSX_VERSION} \
    --with-protoc="${PROTOC_PATH}" \
    --disable-shared \
    --prefix=${LIBDIR}/ios_arm7 \
    --exec-prefix=${LIBDIR}/ios_arm7 \
    "CFLAGS=${CFLAGS} \
    -miphoneos-version-min=${MIN_SDK_VERSION} \
    -arch armv7 \
    -fembed-bitcode \
    -isysroot ${IPHONEOS_SYSROOT}" \
    "CXX=${CXX}" \
    "CXXFLAGS=${CXXFLAGS} \
    -miphoneos-version-min=${MIN_SDK_VERSION} \
    -arch armv7 \
    -fembed-bitcode \
    -isysroot ${IPHONEOS_SYSROOT}" \
    LDFLAGS="-arch armv7 \
    -fembed-bitcode \
    -miphoneos-version-min=${MIN_SDK_VERSION} \
    ${LDFLAGS}" \
    "LIBS=${LIBS}"
    echo make -j"${JOB_COUNT}"
    echo make install
fi

if [[ ! " ${ARCHITECTURES[@]} " =~ " arm7s " ]]; then
    echo make distclean
    echo ./configure \
    --host=armv7s-apple-${OSX_VERSION} \
    --with-protoc="${PROTOC_PATH}" \
    --disable-shared \
    --prefix=${LIBDIR}/ios_arm7s \
    --exec-prefix=${LIBDIR}/ios_arm7s \
    "CFLAGS=${CFLAGS} \
    -miphoneos-version-min=${MIN_SDK_VERSION} \
    -arch armv7s \
    -fembed-bitcode \
    -isysroot ${IPHONEOS_SYSROOT}" \
    "CXX=${CXX}" \
    "CXXFLAGS=${CXXFLAGS} \
    -miphoneos-version-min=${MIN_SDK_VERSION} \
    -arch armv7s \
    -fembed-bitcode \
    -isysroot ${IPHONEOS_SYSROOT}" \
    LDFLAGS="-arch armv7s \
    -fembed-bitcode \
    -miphoneos-version-min=${MIN_SDK_VERSION} \
    ${LDFLAGS}" \
    "LIBS=${LIBS}"
    echo make -j"${JOB_COUNT}"
    echo make install
fi

if [[ ! " ${ARCHITECTURES[@]} " =~ " arm64 " ]]; then
    echo make distclean
    echo ./configure \
    --host=arm \
    --with-protoc="${PROTOC_PATH}" \
    --disable-shared \
    --prefix=${LIBDIR}/ios_arm64 \
    --exec-prefix=${LIBDIR}/ios_arm64 \
    "CFLAGS=${CFLAGS} \
    -miphoneos-version-min=${MIN_SDK_VERSION} \
    -arch arm64 \
    -fembed-bitcode \
    -isysroot ${IPHONEOS_SYSROOT}" \
    "CXXFLAGS=${CXXFLAGS} \
    -miphoneos-version-min=${MIN_SDK_VERSION} \
    -arch arm64 \
    -fembed-bitcode \
    -isysroot ${IPHONEOS_SYSROOT}" \
    LDFLAGS="-arch arm64 \
    -fembed-bitcode \
    -miphoneos-version-min=${MIN_SDK_VERSION} \
    ${LDFLAGS}" \
    "LIBS=${LIBS}"
    echo make -j"${JOB_COUNT}"
    echo make install
fi

# Run lipo for libprotobuf and libprotobuf-lite
LIPO_CMD=lipo
for ((i=0;i<${#ARCHITECTURES[@]};++i)); do
    ARCH=${ARCHITECTURES[i]}
    LIPO_CMD+=" ${LIBDIR}/ios_$ARCH/lib/libprotobuf.a"
done
LIPO_CMD+=" -create -output ${LIBDIR}/libprotobuf.a"
echo eval $LIPO_CMD

LIPO_CMD=lipo
for ((i=0;i<${#ARCHITECTURES[@]};++i)); do
    ARCH=${ARCHITECTURES[i]}
    LIPO_CMD+=" ${LIBDIR}/ios_$ARCH/lib/libprotobuf-lite.a"
done
LIPO_CMD+=" -create -output ${LIBDIR}/libprotobuf-lite.a"
echo eval $LIPO_CMD

