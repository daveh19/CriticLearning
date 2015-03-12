########## Parameters #############
random_seed = 4::Int;#3;

# network parameters
#no_pre_neurons = 100;
no_pre_neurons_per_task = 50::Int;
no_post_neurons = 2::Int;
no_input_tasks = 2::Int;

# input tuning function
# empty constructors utilise multiple dispatch in selecting tuning function
abstract TuningSelector
type gaussian_tc <: TuningSelector end
type linear_tc <: TuningSelector end
no_tuning_curves_per_input_neuron = 2::Int; # as this increases the inputs increase in magnitude!
use_gaussian_tuning_function = true::Bool;
use_linear_tuning_function = false::Bool;
normalise_height_of_multiple_gaussian_inputs = true::Bool;
gaussian_tuning_mu_lower_bound = (-1.0) :: Float64;
gaussian_tuning_mu_upper_bound = 1.0 :: Float64;
fix_tuning_gaussian_width = true :: Bool;
gaussian_tuning_sigma = 0.25 :: Float64;
gaussian_tuning_sigma_width = 0.1 :: Float64;
gaussian_tuning_height = 2.0 :: Float64;
gaussian_tuning_height_variance = 0.25 :: Float64;

# weight constraints
weights_upper_bound = 10.0 :: Float64;
weights_lower_bound = (-10.0) :: Float64;

# trial length parameters
no_trials_in_block = 80::Int; #80;
no_blocks_in_experiment = 5::Int; #20::Int; #14;
no_subjects = 10::Int; #10;
double_no_of_trials_in_alternating_experiment = false::Bool;

# critic parameters
no_task_critics = 2 :: Int;
no_choices_per_task_critics = 1 :: Int;
use_multi_critic = false :: Bool;
use_single_global_critic = true :: Bool;
reset_average_reward_on_each_block = true :: Bool;

# problem difficulty parameters
problem_left_bound = (-1.) :: Float64; #-0.5;
problem_right_bound = 1. :: Float64; #0.5;

running_av_window_length = 50::Int; #50::Int;

learning_rate = 0.00012 :: Float64; #0.00001 for debugging # 0.00012 was pretty good with Henning # 0.001; #0.002;
output_noise = 10.0 :: Float64; #10.0;

initial_weight_bias = (2.0) :: Float64; # 2.0
gaussian_weight_bias = (0.5) :: Float64;

# choose input sequence
use_cts_random_inputs = true :: Bool;
use_binary_alternating_inputs = false :: Bool;
use_binary_random_inputs = false :: Bool;

# selective tuning of input
input_baseline = 2.0 :: Float64; #2.0;
input_baseline_variance = 0.5 :: Float64; #0.5;
task_tuning_slope_variance = zeros(no_input_tasks) :: Array{Float64,1};
task_tuning_slope_variance[1] = 0.375 :: Float64; # easy task
task_tuning_slope_variance[2] = 0.25 :: Float64; # hard task


# discrimination threshold calculation
perform_detection_threshold = false :: Bool;
detection_threshold = 0.25 :: Float64;

# this mimics the same subject in each experiment, rather than new subjects throughout
use_ab_persistence = false :: Bool; 

# Reward can be saved from the state of the running average or on a per block basis
#	reward averaging on a per task basis is only on a per block basis
save_reward_from_running_average = true :: Bool;

# first block of each experiment is just used to build up a running average
const disable_learning_on_first_block = false :: Bool;

# Verbosity of console output:
#   (-1) : You only see the beginning of each experiment headers
#   0 : You just see the passing of the block beginnings
#   1 : You see the individual trial choices
#   2 : You see the interpretation of the trial choices
#   3 : You see the summary output at the end of each block
#   4 : You see the individual weight update norms
verbosity = (-1) :: Int;

# plotting options
plotting_scatter_plot_on = true :: Bool; # dots from scatter plot showing individual subject results 
plotting_individual_subjects_on = true :: Bool; # lines joining the individual subject data points
plotting_error_bars_on = false :: Bool; # standard deviation from the mean
use_plot_mean = false :: Bool; # plot mean rather than median for proportions correct