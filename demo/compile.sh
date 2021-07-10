#!/bin/bash

# Copyright 2020 Xilinx Inc.
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

# Author: Mark Harvey, Xilinx Inc

if [ $1 = zcu102 ]; then
      ARCH=/opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU102/arch.json
      TARGET=zcu102
      echo "-----------------------------------------"
      echo "COMPILING MODEL FOR ZCU102.."
      echo "-----------------------------------------"
elif [ $1 = zcu104 ]; then
      ARCH=/opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU104/arch.json
      TARGET=zcu104
      echo "-----------------------------------------"
      echo "COMPILING MODEL FOR ZCU104.."
      echo "-----------------------------------------"
elif [ $1 = u50 ]; then
      ARCH=/opt/vitis_ai/compiler/arch/DPUCAHX8H/U50/arch.json
      TARGET=u50
      echo "-----------------------------------------"
      echo "COMPILING MODEL FOR ALVEO U50.."
      echo "-----------------------------------------"
else
      echo  "Target not found. Valid choices are: zcu102, zcu104, u50 ..exiting"
      exit 1
fi

MODEL="quant_model/${2}_int.xmodel"
NET="${2}_${TARGET}"

compile() {
  vai_c_xir \
  --xmodel      ${MODEL} \
  --arch        $ARCH \
  --net_name    ${NET} \
  --output_dir  compiled_model
}

compile 2>&1 | tee compile.log

echo "-----------------------------------------"
echo "MODEL COMPILED"
echo "-----------------------------------------"
