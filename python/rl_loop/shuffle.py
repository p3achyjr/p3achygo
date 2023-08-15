'''
Shuffler wrapper.

Invokes a C++ binary and waits until a new model is uploaded, before stopping.
'''

from __future__ import annotations

import os, shlex, signal, sys, time
import gcs_utils as gcs
import rl_loop.config as config
import rl_loop.fs_utils as fs_utils
import rl_loop.shuffle_metadata as shuffle_metadata

from absl import app, flags, logging
from pathlib import Path
from subprocess import Popen, PIPE, STDOUT
from threading import Thread

FLAGS = flags.FLAGS
POLL_INTERVAL_S = 10

# How many times to use each sample, on average.
AVG_NUM_SYMMETRIES = 3

running = True

flags.DEFINE_string('bin_path', '', 'Local path to shuffler binary.')
flags.DEFINE_string('run_id', '', 'ID corresponding to the current run.')
flags.DEFINE_string('local_run_dir', '/tmp/p3achygo',
                    'Local path for training data')


def handle_shutdown(signum, _):
  logging.info(f'Received Shutdown Signal {signum}')
  global running
  running = False


def print_stdout(out: Popen.stdout):  # pytype : disable=unbound-type-param
  for line in out:
    print(line.rstrip())


def download_chunks(local_sp_chunk_dir: str, sp_chunks: set[str]):
  for sp_chunk in sp_chunks:
    chunk_filename = str(Path(sp_chunk).name)
    local_chunk_path = str(Path(local_sp_chunk_dir, chunk_filename))
    gcs._download(local_chunk_path, sp_chunk)


def num_samples_in_chunks(gcs_sp_chunks: set[str]) -> int:
  '''
  Calculates total number of samples in a set of self-play chunks. The number
  of examples in each chunk is embedded in the filename.
  '''

  def find_match(path: str):
    p = Path(path)
    for part in p.parts:
      match = gcs.SP_CHUNK_RE.fullmatch(part)
      if match != None:
        return match

    return None

  num_samples = 0
  for sp_chunk in gcs_sp_chunks:
    match = find_match(sp_chunk)
    if not match:
      logging.error(f'No regex match for chunk file: {sp_chunk}')
      continue

    if len(match.groups()) != 6:
      logging.error(f'Wrong number of matches for chunk file: {sp_chunk}')
      continue

    _, _, _, num_samples_in_chunk, _, _ = match.groups()
    num_samples += int(num_samples_in_chunk)

  return num_samples


def loop(bin_path: str, run_id: str, local_run_dir: str,
         config: config.RunConfig):
  '''
  Continually produces new training chunks until reaching a specified number of
  generations.
  '''

  def generate_one_chunk(chunk_gen: int,
                         train_window_size: int,
                         select_sample_prob: float,
                         local_sp_chunk_dir: str,
                         gcs_sp_chunks: set[str],
                         in_continuous_mode=True):
    num_new_samples = 0
    # run shuffler.
    env = os.environ.copy()
    env['LD_PRELOAD'] = '/usr/local/lib/libmimalloc.so'
    num_games_to_play = config.games_per_gen if chunk_gen > 1 else config.games_first_gen
    cmd = shlex.split(
        f'{bin_path} --data_path={local_sp_chunk_dir}' + f' --gen={chunk_gen}' +
        f' --games_this_gen={num_games_to_play}' +
        f' --train_window_size={train_window_size}' +
        f' --p={select_sample_prob}' +
        f' --run_continuously={"true" if in_continuous_mode else "false"}')
    shuf_proc = Popen(cmd,
                      stdin=PIPE,
                      stdout=PIPE,
                      stderr=STDOUT,
                      universal_newlines=True,
                      env=env)
    t = Thread(target=print_stdout, args=(shuf_proc.stdout,), daemon=True)
    t.start()

    while running and shuf_proc.poll() is None:
      time.sleep(POLL_INTERVAL_S)

      if in_continuous_mode:
        # download new chunks
        gcs_sp_chunks_now = set(gcs.list_sp_chunks(run_id))
        gcs_sp_chunks, new_sp_chunks = gcs_sp_chunks_now, gcs_sp_chunks_now.difference(
            gcs_sp_chunks)

        download_chunks(local_sp_chunk_dir, new_sp_chunks)
        num_new_samples += num_samples_in_chunks(new_sp_chunks)

    if shuf_proc.poll() is None:
      shuf_proc.communicate('\n')  # force a flush just to be safe.

    logging.info(f'Shuffler exited with status {shuf_proc.poll()}')

    # Upload chunk.
    gcs.upload_chunk(run_id, gcs.local_chunk_dir(local_sp_chunk_dir), chunk_gen)
    gcs.upload_chunk_size(run_id, gcs.local_chunk_dir(local_sp_chunk_dir),
                          chunk_gen)
    logging.info(f'Uploaded chunk gen {chunk_gen} to gs://p3achygo/{run_id}')

    return num_new_samples, gcs_sp_chunks

  (_, _, local_sp_chunk_dir, _) = fs_utils.ensure_local_dirs(local_run_dir)
  logging.info(f'Using {local_sp_chunk_dir} to store self-play chunks.')

  gcs_sp_chunks = set(gcs.list_sp_chunks(run_id))
  download_chunks(local_sp_chunk_dir, gcs_sp_chunks)

  num_samples_generated = num_samples_in_chunks(gcs_sp_chunks)

  chunk_gen = gcs.get_most_recent_chunk(run_id) + 1
  while chunk_gen <= config.num_generations:
    # calculate metadata.
    train_window_size = shuffle_metadata.training_window_size(
        num_samples_generated)
    select_sample_prob = shuffle_metadata.select_sample_probability(
        train_window_size, config.games_per_gen, AVG_NUM_SYMMETRIES)
    num_new_samples, gcs_sp_chunks = generate_one_chunk(chunk_gen,
                                                        train_window_size,
                                                        select_sample_prob,
                                                        local_sp_chunk_dir,
                                                        gcs_sp_chunks,
                                                        in_continuous_mode=True)
    chunk_gen += 1
    num_samples_generated += num_new_samples

  # We have now generated `config.num_generations` number of chunks. However, we
  # still have not consumed our later self-play data as much as we should have.
  # In the pathological case, for our last batch of self-play, we sample the data
  # in the batch once, leaving each sample used an average of 1 / `generation_window``
  # number of times.
  #
  # We want to sample each sample `AVG_NUM_SYMMETRIES` number of times, so we
  # should continue training past the end of self-play. We provide the number
  # of generations in our config. At each extra generation, we will simulate
  # as if we have played a new generation, and calculate our training window
  # accordingly.
  num_samples_per_gen_est = config.games_per_gen * 75
  total_gens = config.num_generations + config.extra_train_gens
  extra_gen = 0
  while chunk_gen <= total_gens:
    # pretend as if we have actually played `extra_gen` number of generations, to
    # calculate sample window. Then subtract `extra_gen * num_samples_per_gen_est` to
    # get the actual sample window from our training data.
    num_samples_generated_sim = (num_samples_generated +
                                 extra_gen * num_samples_per_gen_est)
    train_window_size = shuffle_metadata.training_window_size(
        num_samples_generated_sim)
    # compute sample probability as if we had played extra generations
    select_sample_prob = shuffle_metadata.select_sample_probability(
        train_window_size, config.games_per_gen, AVG_NUM_SYMMETRIES)

    # then normalize training window so that we only look at samples that lie
    # in our training window, had we continued playing those generations.
    train_window_size -= (extra_gen * num_samples_per_gen_est)
    train_window_size = max(100000, train_window_size)
    _, _ = generate_one_chunk(chunk_gen,
                              train_window_size,
                              select_sample_prob,
                              local_sp_chunk_dir,
                              gcs_sp_chunks,
                              in_continuous_mode=False)
    chunk_gen += 1
    extra_gen += 1

  logging.info(f'Chunk gen: {chunk_gen}. Shutting down.')


def main(_):
  if FLAGS.bin_path == '':
    logging.error('No --bin_path specified.')
    return

  if FLAGS.run_id == '':
    logging.error('No --run_id specified.')
    return

  run_config = config.parse(FLAGS.run_id)
  loop(FLAGS.bin_path, FLAGS.run_id, FLAGS.local_run_dir, run_config)


if __name__ == '__main__':
  signal.signal(signal.SIGINT, handle_shutdown)
  signal.signal(signal.SIGTERM, handle_shutdown)
  sys.stdout.reconfigure(line_buffering=True)  # pytype: disable=attribute-error
  sys.stderr.reconfigure(line_buffering=True)  # pytype: disable=attribute-error
  app.run(main)
