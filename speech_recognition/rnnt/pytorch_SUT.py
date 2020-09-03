# Copyright (c) 2020, Cerebras Systems, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#           http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import os
sys.path.insert(0, os.path.join(os.getcwd(), "pytorch"))

import array
import numpy as np
import toml
import mlperf_loadgen as lg
from tqdm import tqdm

from QSL import AudioQSL, AudioQSLInMemory
from decoders import ScriptGreedyDecoder
from helpers import add_blank_label
from preprocessing import AudioPreprocessing
from model_separable_rnnt import RNNT

import multiprocessing as mp
import threading
import time

def get_num_cores():
    cmd = "lscpu | awk '/^Core\(s\) per socket:/ {cores=$NF}; /^Socket\(s\):/ {sockets=$NF}; END{print cores*sockets}'"
    lscpu = os.popen(cmd).readlines()
    lscpu = int(lscpu[0])
    return lscpu

def block_until(counter, num_ins, t):
    while counter.value < num_ins:
        time.sleep(t)

class Input(object):
    def __init__(self, id_list, idx_list):
        assert isinstance(id_list, list)
        assert isinstance(idx_list, list)
        assert len(id_list) == len(idx_list)
        self.query_id_list = id_list
        self.query_idx_list = idx_list


class Output(object):
    def __init__(self, query_id, transcript):
        self.query_id = query_id
        self.transcript = transcript


class InQueue():
    def __init__(self, in_queue, batch_size=1):
        self.in_queue = in_queue
        self.batch_size = batch_size

    def put(self, query_samples):
        query_len = len(query_samples)
        query_idx = [q.index for q in query_samples]
        query_id = [q.id for q in query_samples]

        bs = self.batch_size
        for i in range(0, query_len, bs):
            i_end = min(i + bs, query_len)
            current_batch_size = i_end - i
            input_item = Input(query_id[i:i_end], query_idx[i:i_end])
            self.in_queue.put(input_item)


class Consumer(mp.Process):
    def __init__(self, task_queue, result_queue, lock, init_counter,
                 rank, world_size, num_cores,
                 qsl, config_toml, checkpoint_path, dataset_dir, manifest_filepath, perf_count):

        mp.Process.__init__(self)

        ### sub process
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.lock = lock
        self.init_counter = init_counter
        self.rank = rank
        if rank == 0:
            self.start_core = rank * num_cores // world_size + 2
        else:
            self.start_core = rank * num_cores // world_size
        self.end_core = (rank + 1) * num_cores // world_size - 1

        self.qsl = qsl
        self.config_toml = config_toml
        self.checkpoint_path = checkpoint_path
        self.dataset_dir = dataset_dir
        self.manifest_filepath = manifest_filepath
        self.perf_count = perf_count

        self.model_init = False


    def run(self):
        import torch

        core_list = range(self.start_core, self.end_core + 1)
        num_cores = len(core_list)
        os.sched_setaffinity(self.pid, core_list)
        print("### set rank {} to cores [{}:{}]; omp num threads = {}"
            .format(self.rank, self.start_core, self.end_core, num_cores))

        self.lock.acquire()
        self.init_counter.value += 1
        self.lock.release()

        torch.set_num_threads(num_cores)

        if not self.model_init:
            print("lazy_init rank {}".format(self.rank))
            config = toml.load(self.config_toml)
            dataset_vocab = config['labels']['labels']
            rnnt_vocab = add_blank_label(dataset_vocab)
            featurizer_config = config['input_eval']
            self.audio_preprocessor = AudioPreprocessing(**featurizer_config)
            self.audio_preprocessor.eval()
            self.audio_preprocessor = torch.jit.script(self.audio_preprocessor)
            self.audio_preprocessor = torch.jit._recursive.wrap_cpp_module(
                torch._C._freeze_module(self.audio_preprocessor._c))

            model = RNNT(
                feature_config=featurizer_config,
                rnnt=config['rnnt'],
                num_classes=len(rnnt_vocab)
            )
            checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
            migrated_state_dict = {}
            for key, value in checkpoint['state_dict'].items():
                key = key.replace("joint_net", "joint.net")
                migrated_state_dict[key] = value
            del migrated_state_dict["audio_preprocessor.featurizer.fb"]
            del migrated_state_dict["audio_preprocessor.featurizer.window"]
            model.load_state_dict(migrated_state_dict, strict=True)

            model.eval()
            model.encoder = torch.jit.script(model.encoder)
            model.encoder = torch.jit._recursive.wrap_cpp_module(
                torch._C._freeze_module(model.encoder._c))
            model.prediction = torch.jit.script(model.prediction)
            model.prediction = torch.jit._recursive.wrap_cpp_module(
                torch._C._freeze_module(model.prediction._c))
            model.joint = torch.jit.script(model.joint)
            model.joint = torch.jit._recursive.wrap_cpp_module(
                torch._C._freeze_module(model.joint._c))
            model = torch.jit.script(model)

            self.greedy_decoder = ScriptGreedyDecoder(len(rnnt_vocab) - 1, model)

            self.model_init = True

        while True:
            next_task = self.task_queue.get()
            if next_task is None:
                print("### rank {} finished".format(self.rank))
                self.task_queue.task_done()
                break

            query_id_list = next_task.query_id_list
            query_idx_list = next_task.query_idx_list
            query_len = len(query_id_list)
            for idx, id in zip(query_idx_list, query_id_list):
                waveform = self.qsl[idx]
                assert waveform.ndim == 1
                waveform_length = np.array(waveform.shape[0], dtype=np.int64)
                waveform = np.expand_dims(waveform, 0)
                waveform_length = np.expand_dims(waveform_length, 0)

                with torch.no_grad():
                    waveform = torch.from_numpy(waveform)
                    waveform_length = torch.from_numpy(waveform_length)
                    feature, feature_length = self.audio_preprocessor.forward((waveform, waveform_length))
                    assert feature.ndim == 3
                    assert feature_length.ndim == 1
                    feature = feature.permute(2, 0, 1)

                    _, _, transcript = self.greedy_decoder.forward(feature, feature_length)

                assert len(transcript) == 1
                self.result_queue.put(Output(id, transcript))

            self.task_queue.task_done()


def response_loadgen(out_queue):
    out_queue_cnt = 0
    while True:
        next_task = out_queue.get()
        if next_task is None:
            print("Exiting response thread")
            break

        query_id = next_task.query_id
        transcript = next_task.transcript
        assert len(transcript) == 1
        response_array = array.array('q', transcript[0])
        bi = response_array.buffer_info()
        response = lg.QuerySampleResponse(query_id, bi[0],
                                          bi[1] * response_array.itemsize)
        lg.QuerySamplesComplete([response])
        out_queue_cnt += 1

    print("Finish processing {} samples: ".format(out_queue_cnt))


class PytorchSUT:
    def __init__(self, config_toml, checkpoint_path, dataset_dir,
                 manifest_filepath, perf_count, batch_size=1, num_instances=2):
        ### multi instance attributes
        self.num_instances = num_instances
        self.num_cores = get_num_cores()
        self.lock = mp.Lock()
        self.init_counter = mp.Value("i", 0)
        self.output_queue = mp.Queue()
        self.input_queue = mp.JoinableQueue()
        self.issue_queue = InQueue(self.input_queue, batch_size)

        config = toml.load(config_toml)

        dataset_vocab = config['labels']['labels']
        rnnt_vocab = add_blank_label(dataset_vocab)
        featurizer_config = config['input_eval']

        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries,
                                   self.process_latencies)
        self.qsl = AudioQSLInMemory(dataset_dir,
                                    manifest_filepath,
                                    dataset_vocab,
                                    featurizer_config["sample_rate"],
                                    perf_count)

        ### worker process
        self.consumers = [Consumer(self.input_queue, self.output_queue,
                                   self.lock, self.init_counter, i, self.num_instances, self.num_cores,
                                   self.qsl, config_toml, checkpoint_path, dataset_dir, manifest_filepath, perf_count)
                          for i in range(self.num_instances)]

        ### start worker process
        for c in self.consumers:
            c.start()

        ### wait until all sub processes are ready
        block_until(self.init_counter, self.num_instances, 2)

        ### start response thread
        self.response_worker = threading.Thread(
            target=response_loadgen, args=(self.output_queue,))
        self.response_worker.daemon = True
        self.response_worker.start()

    def issue_queries(self, query_samples):
        self.issue_queue.put(query_samples)

    def flush_queries(self):
        pass

    def process_latencies(self, latencies_ns):
        print("Average latency (ms) per query:")
        print(np.mean(latencies_ns)/1000000.0)
        print("Median latency (ms): ")
        print(np.percentile(latencies_ns, 50)/1000000.0)
        print("90 percentile latency (ms): ")
        print(np.percentile(latencies_ns, 90)/1000000.0)

    def __del__(self):
        ### clear up sub processes
        self.input_queue.join()
        for i in range(self.num_instances):
            self.input_queue.put(None)
        for c in self.consumers:
            c.join()
        self.output_queue.put(None)

        print("Finished destroying SUT.")
