import tensorflow as tf
import transforms
import multiprocessing

from board import GoBoard
from constants import *
from pathlib import Path


def main():
  ds_path = "/home/axlui/pro_dataset"
  shards = [str(path) for path in Path(ds_path).glob("*.tfrecord.zz")]
  max_concurrency = multiprocessing.cpu_count()

  ds = tf.data.Dataset.from_tensor_slices(shards)
  ds = ds.interleave(lambda x: tf.data.TFRecordDataset(
      x, compression_type='ZLIB').map(transforms.expand),
                     cycle_length=64,
                     block_length=1,
                     num_parallel_calls=max_concurrency)
  ds = ds.shuffle(1000)
  ds = ds.take(10000)
  ds = ds.prefetch(tf.data.AUTOTUNE)

  for (input, input_global_state, color, komi, score, score_one_hot, policy,
       policy_aux, own, q30, q100, q200) in ds:
    was_any_pass = input_global_state[2:] == 1
    if not (tf.reduce_any(was_any_pass)):
      continue
    input = tf.transpose(input, perm=(2, 0, 1))  # HWC -> CHW
    print('Shape: ', input.shape)
    print('-----Global-----')
    print(input_global_state)
    print('-----Board-----')
    print(GoBoard.to_string(BLACK * input[0] + WHITE * input[1]))
    print('-----Atari-----')
    print(GoBoard.to_string(BLACK * input[7] + WHITE * input[8]))
    print('-----Two Liberties-----')
    print(GoBoard.to_string(BLACK * input[9] + WHITE * input[10]))
    print('-----Three Liberties-----')
    print(GoBoard.to_string(BLACK * input[11] + WHITE * input[12]))
    print('-----Policy-----')
    print(tf.math.argmax(policy))
    print('-----Policy Aux-----')
    print(policy_aux)
    print('-----Own-----')
    print(GoBoard.to_string(own))
    # print('-----Last Moves-----')
    # for i in range(5):
    #   print(tf.math.argmax(tf.nest.flatten(input[6 - i][:, :, 0])))
    print('-----Score-----')
    print(score)
    print('-----Qs-----')
    print(f'q30: {q30}, q100: {q100}, q200: {q200}')
    print()
    print()


if __name__ == '__main__':
  main()
