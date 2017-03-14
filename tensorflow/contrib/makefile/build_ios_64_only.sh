#!/bin/bash

set -euo pipefail

DIR=$(cd $(dirname $BASH_SOURCE) && pwd)

./tensorflow/contrib/makefile/build_64_ios.sh
cd $DIR
./pack.sh
