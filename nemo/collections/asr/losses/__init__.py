# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

from nemo.collections.asr.losses.angularloss import AngularSoftmaxLoss
from nemo.collections.asr.losses.audio_losses import SDRLoss
from nemo.collections.asr.losses.ctc import CTCLoss
from nemo.collections.asr.losses.lattice_losses import LatticeLoss
from nemo.collections.asr.losses.ssl_losses.contrastive import ContrastiveLoss
from nemo.collections.asr.losses.ssl_losses.ctc import CTCLossForSSL
from nemo.collections.asr.losses.ssl_losses.mlm import MLMLoss
from nemo.collections.asr.losses.ssl_losses.rnnt import RNNTLossForSSL

# from nemo.collections.asr.losses.rnnt_dep import RNNTLoss_dep, resolve_rnnt_default_loss_name
# from nemo.collections.asr.parts.numba.rnnt_loss.rnnt_pytorch_dep import TDTLossNumba_dep, MultiblankRNNTLossNumba_dep, RNNTLossNumba_dep