import sys
import time
import torch
from torch import nn
from torch.utils.data import Dataset, random_split, DataLoader


def load_file_as_tensor_list(str__file_path):
    with open(str__file_path) as f:
        int__len, int__tensor_size = map(int, f.readline().split())
        list_ndarray_float__result = []
        for _ in range(int__len):
            # sparse_tensor_float__tensor = torch.sparse_coo_tensor(size=(int__tensor_size,), dtype=torch.float32)
            list_tuple2_str_str__sparse_tensor = map(lambda str_index_and_value: tuple(str_index_and_value.split(':')),
                                                     f.readline().split())
            list_int__indexes = []
            list_float__values = []
            for tuple2_str_str__index_and_value in list_tuple2_str_str__sparse_tensor:
                list_int__indexes.append(int(tuple2_str_str__index_and_value[0]))
                list_float__values.append(float(tuple2_str_str__index_and_value[1]))

                # int__index = int(tuple2_str_str__index_and_value[0])
                # float__value = float(tuple2_str_str__index_and_value[1])
                # sparse_tensor_float__tensor[int__index] = float__value
            sparse_tensor_float__tensor = torch.sparse.FloatTensor(
                torch.LongTensor([[0] * len(list_int__indexes), list_int__indexes]),
                torch.FloatTensor(list_float__values),
                torch.Size([1, int__tensor_size]))
            # print(sparse_tensor_float__tensor)
            list_ndarray_float__result.append(sparse_tensor_float__tensor)
        return list_ndarray_float__result


class Amazon670KDataset(Dataset):
    def __init__(self, str__feature_file, str__label_file):
        list_ndarray_float__feature_list = load_file_as_tensor_list(str__feature_file)
        print(f"load feature list len:{len(list_ndarray_float__feature_list)}", file=sys.stderr)
        list_ndarray_float__label_list = load_file_as_tensor_list(str__label_file)
        print(f"load label list len:{len(list_ndarray_float__label_list)}", file=sys.stderr)
        assert len(list_ndarray_float__feature_list) == len(list_ndarray_float__label_list)
        self.list_tuple2_tensor_float_tensor_float__values = list(
            map(lambda v: (v[0], v[1]._indices()[1][0]),
                filter(lambda v: v[1]._nnz() == 1,
                       zip(list_ndarray_float__feature_list, list_ndarray_float__label_list))))

    def __len__(self):
        return len(self.list_tuple2_tensor_float_tensor_float__values)

    def __getitem__(self, int__index):
        tensor_float__input, tensor_float__output = self.list_tuple2_tensor_float_tensor_float__values[int__index]
        dict_str_ndarray_float__result = {
            "feature": tensor_float__input,
            "label": tensor_float__output
        }
        return dict_str_ndarray_float__result


class NeuralNetwork(nn.Module):
    def __init__(self, int__input_size, int__output_size):
        super(NeuralNetwork, self).__init__()
        self.operation_flatten = nn.Flatten()
        self.operation_sequential = nn.Sequential(
            nn.Linear(int__input_size, 128),
            nn.ReLU(),
            nn.Linear(128, int__output_size),
            nn.Softmax(dim=1)
        )
        print("construct network", file=sys.stderr)

    def forward(self, tensor__input):
        tensor__mid = self.operation_flatten(tensor__input)
        tensor__output = self.operation_sequential(tensor__mid)
        return tensor__output


def train(dataset):
    # device = 'cuda:0'
    device = 'cpu'
    list_dataset__split_dataset = random_split(dataset,
                                               [int(len(dataset) * 0.95), len(dataset) - int(len(dataset) * 0.95)])
    dataset__train = DataLoader(list_dataset__split_dataset[0], batch_size=256, shuffle=True)
    dataset__test = DataLoader(list_dataset__split_dataset[1], batch_size=256, shuffle=True)
    model = NeuralNetwork(135909, 670091)
    # model.cuda()
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    int__train_iteration = 0
    print('log_type,iteration,time_ms,accuracy,loss')
    float__train_start_time = time.perf_counter()
    for e in range(10):
        for dict_str_tensor_float__current_data in dataset__train:
            print(f'e={e} i={int__train_iteration}', file=sys.stderr)
            tensor_float__input = dict_str_tensor_float__current_data["feature"].to_dense().to(device)
            tensor_float__expect_output = dict_str_tensor_float__current_data["label"].to(device)
            tensor_float__output = model(tensor_float__input)
            tensor_float__loss = loss(tensor_float__output, tensor_float__expect_output)

            float__loss = tensor_float__loss.item() / len(tensor_float__output)
            tensor_bool__accuracy = tensor_float__output.argmax(dim=1) == tensor_float__expect_output
            float__accuracy = float(sum(tensor_bool__accuracy).item() / len(tensor_bool__accuracy))
            print(
                f'train_log,{int__train_iteration},{int((time.perf_counter() - float__train_start_time) * 1000)},{float__accuracy},{float__loss}',
                flush=True)

            optimizer.zero_grad()
            tensor_float__loss.backward()
            optimizer.step()

            int__train_iteration += 1

        int__data_length = 0
        float__sum_of_loss = 0
        int__correct_count = 0
        with torch.no_grad():
            for dict_str_tensor_float__current_data in dataset__test:
                tensor_float__input = dict_str_tensor_float__current_data["feature"].to_dense().to(device)
                tensor_float__expect_output = dict_str_tensor_float__current_data["label"].to(device)
                tensor_float__output = model(tensor_float__input)
                tensor_float__loss = loss(tensor_float__output, tensor_float__expect_output)
                int__data_length += len(tensor_float__output)
                float__sum_of_loss += tensor_float__loss.item()
                int__correct_count += int(sum(tensor_float__output.argmax(dim=1) == tensor_float__expect_output))
        print(
            f'test_log,{int__train_iteration},{int((time.perf_counter() - float__train_start_time) * 1000)},{int__correct_count / int__data_length},{float__sum_of_loss / int__data_length}',
            flush=True)


train(Amazon670KDataset("../Amazon-670K/trn_ft_mat.txt", "../Amazon-670K/trn_lbl_mat.txt"))
