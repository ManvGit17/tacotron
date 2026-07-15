# Fixed hparams.py for TensorFlow 2.x compatibility

class HParams:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def parse(self, values):
        for k, v in values.items():
            setattr(self, k, v)

    def to_dict(self):
        return self.__dict__

# Define Tacotron hyperparameters here
hparams = HParams(
    cleaners='basic_cleaners',
    num_mels=80,
    num_freq=1025,
    sample_rate=22050,
    frame_length_ms=50,
    frame_shift_ms=12.5,
    preemphasis=0.97,
    min_level_db=-100,
    ref_level_db=20,
    outputs_per_step=5,
    batch_size=32,
    adam_beta1=0.9,
    adam_beta2=0.999,
    initial_learning_rate=0.002,
    decay_learning_rate=True,
)


'''

#import tensorflow as tf


# Default hyperparameters:
#from tensorflow.python.training import hparam
# ---- Custom HParams replacement for TensorFlow 2.x ----

class HParams:
    """Simple stand-in for tf.contrib.training.HParams"""
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
    def add_hparam(self, name, value):
        self.__dict__[name] = value
    def __getitem__(self, name):
        return getattr(self, name)
    def __setitem__(self, name, value):
        setattr(self, name, value)
    def __contains__(self, name):
        return hasattr(self, name)
# --------------------------------------------------------

# Now define hparams just like before
hparams = HParams(
    cleaners='english_cleaners',
    n_mels=80,
    n_fft=2048,
    sample_rate=22050,
    frame_shift_ms=None,
    hop_size=256,
    win_size=1024,
    fmin=125,
    fmax=7600,
    preemphasis=0.97,
    min_level_db=-100,
    ref_level_db=20,
    power=1.5,
    griffin_lim_iters=60,
    use_lws=False,
    rescaling=True,
    rescaling_max=0.999,
    allow_clipping_in_normalization=True,
    symmetric_mels=True,
    max_abs_value=4.0,
    signal_normalization=True,
    clip_mels_length=True,
    max_mel_frames=900,
    use_lws_preprocess=False
)

hparams = hparams.HParams(
  # Comma-separated list of cleaners to run on text prior to training and eval. For non-English
  # text, you may want to use "basic_cleaners" or "transliteration_cleaners" See TRAINING_DATA.md.
  cleaners='english_cleaners',

  # Audio:
  num_mels=80,
  num_freq=1025,
  sample_rate=20000,
  frame_length_ms=50,
  frame_shift_ms=12.5,
  preemphasis=0.97,
  min_level_db=-100,
  ref_level_db=20,

  # Model:
  outputs_per_step=5,
  embed_depth=256,
  prenet_depths=[256, 128],
  encoder_depth=256,
  postnet_depth=256,
  attention_depth=256,
  decoder_depth=256,

  # Training:
  batch_size=32,
  adam_beta1=0.9,
  adam_beta2=0.999,
  initial_learning_rate=0.002,
  decay_learning_rate=True,
  use_cmudict=False,  # Use CMUDict during training to learn pronunciation of ARPAbet phonemes

  # Eval:
  max_iters=200,
  griffin_lim_iters=60,
  power=1.5,              # Power to raise magnitudes to prior to Griffin-Lim
)


def hparams_debug_string():
  values = hparams.values()
  hp = ['  %s: %s' % (name, values[name]) for name in sorted(values)]
  return 'Hyperparameters:\n' + '\n'.join(hp)
'''