#### Forked from minigo ####
#### https://github.com/tensorflow/minigo/blob/master/cc/tensorflow/copy_outputs.sh ####

#!/bin/bash

set -euo pipefail

if [[ $# -eq 0 ]] ; then
    echo 'Usage: build.sh dst_dir'
    exit 1
fi

src_dir=${BASH_SOURCE[0]}.runfiles
dst_dir=$1

echo "Copying from \"${src_dir}\" to \"${dst_dir}\""

for sub_dir in lib include bin; do
  rm -rfd "${dst_dir}/${sub_dir}"
  mkdir -p "${dst_dir}/${sub_dir}"
done

# TensorFlow library extensions change on different platforms:
#  - out_ext is the extension generated by the build step.
#  - dst_ext is the extension expected by the library loader at runtime.
if [ -f "${src_dir}/org_tensorflow/tensorflow/libtensorflow_cc.2.11.0.dylib" ]; then
  out_ext=2.11.0.dylib
  dst_ext=2.dylib
else
  out_ext=so.2.11.0
  dst_ext=so.2
fi

rsync -a --copy-links ${src_dir}/__main__/cc/tensorflow/*.so ${dst_dir}/lib/
rsync -a --copy-links ${src_dir}/org_tensorflow/tensorflow/*.${out_ext} ${dst_dir}/lib/
rsync -a --copy-links --exclude "*.${out_ext}" ${src_dir}/org_tensorflow/ ${dst_dir}/include/
rsync -a --copy-links ${src_dir}/eigen_archive/ ${dst_dir}/include/third_party/eigen3/
rsync -a --copy-links ${src_dir}/com_google_protobuf/src/ ${dst_dir}/include/
rsync -a --copy-links ${src_dir}/org_tensorflow/tensorflow/lite/toco/toco ${dst_dir}/bin/

mv ${dst_dir}/lib/libtensorflow_cc.${out_ext} \
   ${dst_dir}/lib/libtensorflow_cc.${dst_ext}
mv ${dst_dir}/lib/libtensorflow_framework.${out_ext} \
   ${dst_dir}/lib/libtensorflow_framework.${dst_ext}