import argparse
import os
import tensorflow as tf

# Enable TF1 behavior in TF2
tf.compat.v1.disable_v2_behavior()

from hparams import hparams
from models import create_model
from text import text_to_sequence
from util import audio

def synthesize(model_name, checkpoint, text_input, output_dir):
    tf.compat.v1.reset_default_graph()
    with tf.compat.v1.Session() as sess:
        model = create_model(model_name, hparams)
        inputs = tf.compat.v1.placeholder(tf.int32, [1, None], 'inputs')
        input_lengths = tf.compat.v1.placeholder(tf.int32, [1], 'input_lengths')

        model.initialize(inputs, input_lengths)
        saver = tf.compat.v1.train.Saver()
        saver.restore(sess, checkpoint)

        seq = text_to_sequence(text_input, [hparams.cleaners])
        feed_dict = {inputs: [seq], input_lengths: [len(seq)]}
        linear_output = sess.run(model.linear_outputs[0], feed_dict=feed_dict)
        wav = audio.inv_spectrogram(linear_output.T)
        audio.save_wav(wav, os.path.join(output_dir, 'synthesized.wav'))
        print("✅ Saved synthesized audio to synthesized.wav")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='tacotron')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--text', required=True)
    parser.add_argument('--output_dir', default='.')
    args = parser.parse_args()
    synthesize(args.model, args.checkpoint, args.text, args.output_dir)

if __name__ == '__main__':
    main()


''' import io
import numpy as np
import tensorflow as tf
from hparams import hparams
from librosa import effects
from models import create_model
from text import text_to_sequence
from util import audio


class Synthesizer:
  def load(self, checkpoint_path, model_name='tacotron'):
    print('Constructing model: %s' % model_name)
    inputs = tf.placeholder(tf.int32, [1, None], 'inputs')
    input_lengths = tf.placeholder(tf.int32, [1], 'input_lengths')
    with tf.variable_scope('model') as scope:
      self.model = create_model(model_name, hparams)
      self.model.initialize(inputs, input_lengths)
      self.wav_output = audio.inv_spectrogram_tensorflow(self.model.linear_outputs[0])

    print('Loading checkpoint: %s' % checkpoint_path)
    self.session = tf.Session()
    self.session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(self.session, checkpoint_path)


  def synthesize(self, text):
    cleaner_names = [x.strip() for x in hparams.cleaners.split(',')]
    seq = text_to_sequence(text, cleaner_names)
    feed_dict = {
      self.model.inputs: [np.asarray(seq, dtype=np.int32)],
      self.model.input_lengths: np.asarray([len(seq)], dtype=np.int32)
    }
    wav = self.session.run(self.wav_output, feed_dict=feed_dict)
    wav = audio.inv_preemphasis(wav)
    wav = wav[:audio.find_endpoint(wav)]
    out = io.BytesIO()
    audio.save_wav(wav, out)
    return out.getvalue()
'''