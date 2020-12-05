import os
import random
import copy
import numpy as np
import torch


class SpectrumLoader():
    def __init__(self,  recpetive_field, downsampling=1, target_regional_supression=0,
                 datafolder="./output", filename_suffix="_ex", AWGN=[0, 0],
                 centralize=[True, True], source_scale=[1, 1],
                 source_bias=[0, 0], data_split=[0, 1.0], data_kept=[0, 0]):
        self.generated_dataset_atlas = []
        self.batch_index = 0
        self.downsampling = downsampling
        self.recpetive_field = recpetive_field
        self.AWGN = AWGN
        self.centralize = centralize
        self.target_regional_supression = target_regional_supression
        self.target_smooting = False
        self.source_scale = source_scale
        self.source_bias = source_bias

        dataset_data = self.read_folder(datafolder, ".npy", filename_suffix)
        firstdata = int(len(dataset_data) *
                        data_split[0])
        lastdata = int(len(dataset_data)*data_split[1])
        if data_kept[0] != 0:
            firstdata = data_kept[0]
        if data_kept[1] != 0:
            lastdata = min(len(dataset_data), data_kept[1])
        print("data_split:", firstdata, lastdata)
        dataset_data = dataset_data[firstdata:lastdata]
        self.size = len(dataset_data)
        for each in dataset_data:
            if not each in self.generated_dataset_atlas:
                self.generated_dataset_atlas.append(each)
        self.reset(randomize=True)

    def read_folder(self, path, endswith, type):
        return [os.path.join(root, name)
                for root, dirs, files in os.walk(path)
                for name in files
                if (name.endswith(endswith) and name.find(type) >= 0)]

    def reset(self, randomize=False):
        self.batch_index = 0
        if randomize:
            self.random_lut = np.random.permutation(
                len(self.generated_dataset_atlas))
        else:
            self.random_lut = np.arange(len(self.generated_dataset_atlas))

    def _get_one_sample(self, batch_index=None):
        if batch_index is None:
            self.batch_index += 1
            self.batch_index %= len(self.random_lut)
        else:
            self.batch_index = batch_index
        filename_raw = self.generated_dataset_atlas[self.random_lut[self.batch_index]]
        loaded_npy = np.load(filename_raw)
        target = loaded_npy[2]
        source = loaded_npy[0:2]
        for i, noise in enumerate(self.AWGN):
            if noise > 0:
                source[i] = source[i] + \
                    np.random.normal(0, noise, source[i].shape)
        source[0] /= self.source_scale[0]
        source[1] /= self.source_scale[1]
        source[0] += self.source_bias[0]
        source[1] += self.source_bias[1]
        if self.downsampling > 1:
            N = self.downsampling
            end = N * int(len(source[0])/N)
            start = 0
            v1 = np.mean(source[0][start:end].copy(
            ).reshape(-1, self.downsampling), 1)
            v2 = np.mean(source[1][start:end].copy(
            ).reshape(-1, self.downsampling), 1)
            source = np.vstack((v1, v2))
            target = np.mean(
                target[start:end].reshape(-1, self.downsampling), 1)
            target = (target > 0)

        self.regional_supression(target)

        return source, target, filename_raw

    def regional_supression(self, target):
        if self.target_regional_supression > 0:
            i = 0
            while i < len(target):
                if target[i] > 0:
                    pos = 0
                    accumulator = 0
                    for j in range(0, self.target_regional_supression):
                        if i+j < len(target):
                            if target[i+j] > 0:
                                pos += j
                                accumulator += 1
                            target[i+j] = 0
                    pos = pos//accumulator
                    target[i+pos] = 1
                    i += j
                i += 1

    def _get_batches_of_transformed_samples(self, sample_per_file=-1):
        while True:
            source_raw, target_raw, filename_raw = self._get_one_sample()
            if len(target_raw) > self.recpetive_field*2:
                break
        samples = []
        if sample_per_file > 0:
            event_center = []
            for x in range(len(target_raw)):
                if target_raw[x] >= 1:
                    event_center.append(x)
            i = 0
            for each in event_center:
                for _ in range(sample_per_file//(2*len(event_center))):
                    head = each-self.recpetive_field//2 + \
                        random.randint(0, self.recpetive_field//4)
                    end = head + self.recpetive_field
                    if (head >= 0) and (end < len(target_raw)):
                        i += 1
                        samples.append([head, end])
            while i < sample_per_file:
                head = random.randint(0, len(target_raw) -
                                      self.recpetive_field)
                end = head + self.recpetive_field
                if (head >= 0) and (end < len(target_raw)):
                    i += 1
                    samples.append([head, end])
        else:
            for head in range(len(target_raw)-self.recpetive_field+1):
                samples.append([head, head+self.recpetive_field])
            sample_per_file = len(samples)

        event_list = []
        for pos in range(len(target_raw)):
            if target_raw[pos] > 0:
                event_list.append(pos)

        target_raw_reg = np.zeros(len(target_raw))
        target_raw_invreg = np.zeros(len(target_raw))
        event_list_rhs = copy.copy(event_list)
        event_list_lhs = []
        for pos in range(len(target_raw)):
            to_event_dist = self._to_nearest_event(
                event_list_rhs, pos, event_list_lhs)
            if abs(to_event_dist) < self.recpetive_field//2:
                target_raw_reg[pos] = 0.5 + to_event_dist/self.recpetive_field
                target_raw_invreg[pos] = 0.5 - \
                    to_event_dist/self.recpetive_field
        source = np.zeros((sample_per_file, 2, self.recpetive_field))
        target_regress = np.zeros((sample_per_file, 1))
        target_invreg = np.zeros((sample_per_file, 1))
        for sample_cnt in range(sample_per_file):
            each = samples[min(sample_cnt, len(samples))]
            source[sample_cnt][0][:] = source_raw[0][each[0]:each[1]]
            source[sample_cnt][1][:] = source_raw[1][each[0]:each[1]]
            target_regress[sample_cnt] = target_raw_reg[(each[0]+each[1])//2]
            target_invreg[sample_cnt] = target_raw_invreg[(
                each[0]+each[1])//2]
        return source, target_regress, target_invreg, (source_raw, target_raw, event_list, filename_raw)

    def _to_nearest_event(self, event_list_rhs, pos, event_list_lhs):
        while event_list_rhs and pos > event_list_rhs[0]:
            event_list_lhs.append(event_list_rhs.pop(0))
        to_right_event = self.recpetive_field+1
        if event_list_rhs:
            to_right_event = min(to_right_event, event_list_rhs[0]-pos)
        to_left_event = self.recpetive_field+1
        if event_list_lhs:
            to_left_event = min(to_left_event, pos-event_list_lhs[-1])
        if to_left_event > to_right_event:
            to_event_dist = -to_right_event
        else:
            to_event_dist = to_left_event
        return to_event_dist

    def get_distances(self, target_events, predict_events):
        detection_to_target_distances = []
        event_list_rhs = copy.copy(target_events)
        event_list_lhs = []
        for pos in predict_events:
            detection_to_target_distances.append(self._to_nearest_event(
                event_list_rhs, pos, event_list_lhs))
        return detection_to_target_distances

    def centralize_transform(self, sourceold):
        source = sourceold.clone().permute(2, 0, 1)
        center_value = source[source.size(0)//2, :, :]
        if self.centralize[0]:
            source[:, :, 0] -= center_value[:, 0]
        if self.centralize[1]:
            source[:, :, 1] -= center_value[:, 1]
        sourceold = source.permute(1, 2, 0)
        return sourceold

    def get_samples(self, *args, device, **vargs):
        source_cpu, target_regress_cpu, target_invreg_cpu, rawinfo = self._get_batches_of_transformed_samples(
            *args, **vargs)

        source = torch.from_numpy(source_cpu).to(device)

        target_invreg = torch.from_numpy(
            target_invreg_cpu.astype(np.double)).to(device)
        target_regress = torch.from_numpy(
            target_regress_cpu.astype(np.double)).to(device)

        source = self.centralize_transform(source)
        return source, target_regress, target_invreg, rawinfo


def main():
    spLoader = SpectrumLoader(
        datafolder="./output", filename_suffix="", recpetive_field=224)
    device = torch.device('cpu')
    source, target_regress, target_invreg, rawinfo = spLoader.get_samples(
        device=device, sample_per_file=-1)
    (source_raw, target_raw, event_list, filename_raw) = rawinfo
    print(filename_raw)
    import matplotlib
    import matplotlib.pyplot as plt

    plt.figure(figsize=(30, 10))

    def draw(data, color, offset=0):
        plt.plot(np.arange(data.shape[0]) +
                 offset, data, color, linewidth=1.0)
    for id in range(min(source.shape[0], 50)):
        draw(source[id][0], 'y')
        draw(source[id][1], 'b')

    plt.figure(figsize=(30, 10))
    draw(source_raw[0], 'y')
    draw(source_raw[1], 'b')
    draw(target_raw, 'g')
    draw(target_regress, 'black', offset=224/2)
    draw(target_invreg, 'black', offset=224/2)
    plt.show()


if __name__ == '__main__':
    main()
