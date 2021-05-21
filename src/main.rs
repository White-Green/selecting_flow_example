use std::convert::TryInto;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::time::Instant;

use clap::{App, Arg};
use rand::prelude::SliceRandom;
use rand::thread_rng;
use rayon::iter::IndexedParallelIterator;
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;
use selecting_flow::compute_graph::activation::rectified_linear::ReLU;
use selecting_flow::compute_graph::activation::softmax_with_loss::SoftmaxWithLoss;
use selecting_flow::compute_graph::fully_connected_layer::{ApplyFullyConnectedLayer, FullyConnectedLayer};
use selecting_flow::compute_graph::input_box::InputBox;
use selecting_flow::compute_graph::{ExactDimensionComputeGraphNode, GraphNode};
use selecting_flow::data_types::{Sparse, TensorEitherOwned};
use selecting_flow::hasher::sim_hash::SimHash;
use selecting_flow::hasher::FullyConnectedHasher;
use selecting_flow::optimizer::adam::Adam;

fn main() {
    let arg = App::new("shield_example")
        .arg(Arg::with_name("label").help("path to trn_lbl_mat.txt").long("label").short("l").takes_value(true).required(true))
        .arg(Arg::with_name("feature").help("path to trn_ft_mat.txt").long("feature").short("f").takes_value(true).required(true))
        .get_matches();
    let labels = arg.value_of("label").unwrap();
    let features = arg.value_of("feature").unwrap();
    eprintln!("boot");
    let train_data = read_train_data(labels, features);
    eprintln!("load train_data");
    train(train_data);
}

fn train(train_data: TrainData) {
    let TrainData {
        input_size,
        output_size,
        mut data_pair,
    } = train_data;
    const NUM_ITERATION: usize = 5;
    const MINI_BATCH_SIZE: usize = 256;
    const REBUILD_DELTA_INC: f64 = 1.05;
    const TRAIN_DATA_RATIO: f64 = 0.95;
    let (data_pair_train, data_pair_test) = {
        let mid = (data_pair.len() as f64 * TRAIN_DATA_RATIO) as usize;
        data_pair.shuffle(&mut thread_rng());
        data_pair.split_at_mut(mid)
    };
    let time = Instant::now();
    let mut layer1 = FullyConnectedLayer::new_random_param(input_size, 128, SimHash::new(50, 6, 128, 1, 0.333), Adam::new(0.9, 0.999, 0.001));
    eprintln!("construct layer1 in {}ms", time.elapsed().as_millis());
    let time = Instant::now();
    let mut layer2 = FullyConnectedLayer::new_random_param(128, output_size, SimHash::new(50, 8, 4096, 3, 0.333), Adam::new(0.9, 0.999, 0.001));
    eprintln!("construct layer2 in {}ms", time.elapsed().as_millis());
    let mut next_rebuild = 49;
    let mut rebuild_delta = 50;
    let parallel_num = num_cpus::get();
    dbg!(parallel_num);
    eprintln!("start training");
    let time = Instant::now();
    let mini_batch_count = (data_pair_train.len() + MINI_BATCH_SIZE - 1) / MINI_BATCH_SIZE;
    dbg!(data_pair_train.len());
    dbg!(mini_batch_count);
    println!("log_type,iteration,time_ms,accuracy,loss");
    for e in 0..NUM_ITERATION {
        data_pair_train.shuffle(&mut thread_rng());
        for i in 0..mini_batch_count {
            dbg!(e);
            dbg!(i);
            let batch_range = i * MINI_BATCH_SIZE..((i + 1) * MINI_BATCH_SIZE).min(data_pair_train.len());
            let (sum_of_loss, sum_of_accuracy) = process_mini_batch(&data_pair_train[batch_range.clone()], parallel_num, true, || {
                let input = InputBox::new([input_size]);
                let mid = layer1.apply_to(input.clone(), ReLU::new());
                let output = layer2.apply_to(mid, SoftmaxWithLoss::new());
                (input, output)
            });
            println!(
                "train_log,{},{},{},{}",
                e * mini_batch_count + i,
                time.elapsed().as_millis(),
                sum_of_accuracy / batch_range.len() as f64,
                sum_of_loss / batch_range.len() as f64,
            );
            layer1.update_parameter();
            layer2.update_parameter();
            if e * mini_batch_count + i >= next_rebuild {
                layer1.rebuild_hash();
                layer2.rebuild_hash();
                rebuild_delta = (rebuild_delta as f64 * REBUILD_DELTA_INC) as usize;
                next_rebuild += rebuild_delta;
            }
        }
        let (sum_of_loss, sum_of_accuracy) = process_mini_batch(data_pair_test, parallel_num, false, || {
            let input = InputBox::new([input_size]);
            let mid = layer1.apply_to(input.clone(), ReLU::new());
            let output = layer2.apply_unhash_to(mid, SoftmaxWithLoss::new());
            (input, output)
        });
        println!(
            "test_log,{},{},{},{}",
            (e + 1) * mini_batch_count,
            time.elapsed().as_millis(),
            sum_of_accuracy / data_pair_test.len() as f64,
            sum_of_loss / data_pair_test.len() as f64,
        );
    }
}

fn process_mini_batch<I: 'static + ExactDimensionComputeGraphNode<1, Item = f32>, H: FullyConnectedHasher<f32, f32>>(
    data_pair: &[TrainDataPair],
    parallel_num: usize,
    back_propagate: bool,
    construct_layers: impl Sync + Fn() -> (GraphNode<InputBox<f32, 1>, 1>, GraphNode<ApplyFullyConnectedLayer<I, f32, H, SoftmaxWithLoss<f32>, 0>, 0>),
) -> (f64, f64) {
    crossbeam::scope(|scope| {
        let mut threads = Vec::with_capacity(parallel_num);
        for t in 0..parallel_num {
            let range = t * data_pair.len() / parallel_num..(t + 1) * data_pair.len() / parallel_num;
            threads.push(scope.spawn(|_| {
                let (mut input, mut output) = construct_layers();
                let mut sum_of_loss = 0f64;
                let mut accuracy = 0f64;
                for data in &data_pair[range] {
                    let TrainDataPair {
                        input: input_value,
                        output: output_value,
                    } = &data;
                    assert_eq!(output_value.value_count(), 1);
                    input.set_value(input_value.clone().into());
                    output.set_expect_output(output_value.clone());
                    let output_loss = output.get_output_value();
                    let output_without_loss = output.get_output_without_loss();
                    accuracy += match &output_without_loss {
                        TensorEitherOwned::Dense(tensor) => {
                            let ([correct_index], _) = output_value.iter().next().unwrap();
                            let correct = *tensor.get([correct_index]).unwrap();
                            if tensor.as_all_slice().iter().enumerate().all(|(i, v)| i == correct_index || *v < correct) {
                                1.
                            } else {
                                0.
                            }
                        }
                        TensorEitherOwned::Sparse(tensor) => {
                            let ([correct_index], _) = output_value.iter().next().unwrap();
                            let correct = *tensor.get([correct_index]).unwrap();
                            if tensor.iter().all(|([i], v)| i == correct_index || *v < correct) {
                                1.
                            } else {
                                0.
                            }
                        }
                    };
                    sum_of_loss += *output_loss.get([]).unwrap() as f64;
                    if back_propagate {
                        output.clear_gradient_all();
                        output.back_propagate_all();
                    }
                }
                (sum_of_loss, accuracy)
            }));
        }
        threads.into_iter().fold((0f64, 0f64), |(sum_loss, sum_accuracy), handle| {
            let (loss, accuracy) = handle.join().unwrap();
            (sum_loss + loss, sum_accuracy + accuracy)
        })
    })
    .expect("failed to use thread")
}

struct TrainData {
    input_size: usize,
    output_size: usize,
    data_pair: Vec<TrainDataPair>,
}

struct TrainDataPair {
    input: Sparse<f32, 1>,
    output: Sparse<f32, 1>,
}

impl TrainDataPair {
    fn new(input: Sparse<f32, 1>, output: Sparse<f32, 1>) -> Self {
        TrainDataPair { input, output }
    }
}

fn read_train_data(labels: impl AsRef<Path>, features: impl AsRef<Path>) -> TrainData {
    let (output_size, labels) = read_file_as_tensors(labels);
    let (input_size, features) = read_file_as_tensors(features);
    let data_pair = labels
        .into_par_iter()
        .zip_eq(features)
        .filter(|(output, _)| output.value_count() == 1)
        .map(|(output, input)| TrainDataPair::new(input, output))
        .collect();
    TrainData { input_size, output_size, data_pair }
}

fn read_file_as_tensors(path: impl AsRef<Path>) -> (usize, Vec<Sparse<f32, 1>>) {
    let path = path.as_ref();
    let failed_read = &format!("failed to read file {}", path.display());
    let invalid_format = &format!("invalid file format {}", path.display());
    let file = File::open(path).expect("failed to open feature file");
    let file = BufReader::new(file);
    let mut file = file.lines();
    let [len, tensor_width]: [usize; 2] = file
        .next()
        .expect(failed_read)
        .expect(failed_read)
        .trim()
        .split_whitespace()
        .map(|s| s.parse().expect(invalid_format))
        .collect::<Vec<_>>()
        .try_into()
        .expect(invalid_format);
    let tensor_list = file
        .map(|s| {
            let s = s.expect(failed_read);
            let mut input = Sparse::new([tensor_width]);
            s.trim().split_whitespace().for_each(|s| {
                let [index, value]: [_; 2] = s.trim().split(':').collect::<Vec<_>>().try_into().expect(invalid_format);
                let index = index.parse().expect(invalid_format);
                assert!(index < tensor_width, "{}", invalid_format);
                input.set([index], value.parse().expect(invalid_format));
            });
            input
        })
        .collect::<Vec<_>>();
    assert_eq!(tensor_list.len(), len, "{}", invalid_format);
    (tensor_width, tensor_list)
}
