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

from nemo.collections.asr.models.asr_model import ASRModel
from nemo.collections.asr.models.audio_to_audio_model import AudioToAudioModel
from nemo.collections.asr.models.classification_models import EncDecClassificationModel
from nemo.collections.asr.models.clustering_diarizer import ClusteringDiarizer
from nemo.collections.asr.models.ctc_bpe_models import EncDecCTCModelBPE
from nemo.collections.asr.models.ctc_models import EncDecCTCModel
from nemo.collections.asr.models.enhancement_models import EncMaskDecAudioToAudioModel
from nemo.collections.asr.models.hybrid_rnnt_ctc_bpe_models import EncDecHybridRNNTCTCBPEModel
from nemo.collections.asr.models.hybrid_rnnt_ctc_models import EncDecHybridRNNTCTCModel
from nemo.collections.asr.models.k2_sequence_models import (
    EncDecK2RnntSeqModel,
    EncDecK2RnntSeqModelBPE,
    EncDecK2SeqModel,
    EncDecK2SeqModelBPE,
)
from nemo.collections.asr.models.label_models import EncDecSpeakerLabelModel
from nemo.collections.asr.models.msdd_models import EncDecDiarLabelModel, NeuralDiarizer
from nemo.collections.asr.models.rnnt_bpe_models import EncDecRNNTBPEModel
from nemo.collections.asr.models.rnnt_models import EncDecRNNTModel
from nemo.collections.asr.models.slu_models import SLUIntentSlotBPEModel
from nemo.collections.asr.models.ssl_models import SpeechEncDecSelfSupervisedModel

# jykang
# Frame Level KD
from nemo.collections.asr.models.ctc_bpe_models_KD import EncDecCTCModelBPE_KD
from nemo.collections.asr.models.ctc_models_KD import EncDecCTCModel_KD
from nemo.collections.asr.data import audio_to_text_dataset_KD

# SEQ level KD
from nemo.collections.asr.models.ctc_models_SEQ import EncDecCTCModel_SEQ
from nemo.collections.asr.models.ctc_models_SEQ_fast import EncDecCTCModel_SEQ_fast

from nemo.collections.asr.modules.beam_search_decoder import BeamSearchDecoderWithLM

# Guided CTC KD
from nemo.collections.asr.models.ctc_models_Mask import EncDecCTCModel_Mask

# Self attention extract
from nemo.collections.asr.models.ctc_models_qkv import EncDecCTCModel_qkv
from nemo.collections.asr.models.ctc_bpe_models_qkv import EncDecCTCModelBPE_qkv

# Load filename into dataloader
from nemo.collections.asr.data import audio_to_text_dataset_filename

# Self Attention KD
from nemo.collections.asr.models.ctc_models_SelfAttn import EncDecCTCModel_SelfAttn
from nemo.collections.asr.data import audio_to_text_dataset_SelfAttn
from nemo.collections.asr.data import audio_to_text_SelfAttn

# Conformer Last layer feature extract
from nemo.collections.asr.models.ctc_bpe_models_Lth_feature import EncDecCTCModelBPE_Lth_feature
from nemo.collections.asr.models.ctc_models_Lth_feature import EncDecCTCModel_Lth_feature

# Feature SKD
from nemo.collections.asr.models.ctc_bpe_models_SKD import EncDecCTCModelBPE_SKD
from nemo.collections.asr.data import audio_to_text_dataset_SKD
from nemo.collections.asr.data import audio_to_text_SKD

# Softmax SKD
from nemo.collections.asr.models.ctc_bpe_models_SKD_softmax import EncDecCTCModelBPE_SKD_softmax
from nemo.collections.asr.models.ctc_models_SKD_softmax import EncDecCTCModel_SKD_softmax