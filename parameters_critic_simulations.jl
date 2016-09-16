########## Parameters  for WTA-Linear outputs, which learns #############
random_seed = 3::Int;#3;

# network parameters
#no_pre_neurons = 100;
no_pre_neurons_per_task = 50::Int;
no_post_neurons = 2::Int; # corresponds to number of output decision categories
no_input_tasks = 2::Int;
no_pop_scaling_post_neurons = 1 :: Int; # per output decision category


# trial length parameters
no_trials_in_block = 80::Int; #80;
no_blocks_in_experiment = 14::Int; #20::Int; #14;
no_subjects = 10::Int; #10;
double_no_of_trials_in_alternating_experiment = true ::Bool;

# critic parameters
no_task_critics = 1 :: Int;
no_choices_per_task_critics = 1 :: Int;
use_multi_critic = true :: Bool;
use_single_global_critic = false :: Bool;
reset_average_reward_on_each_block = false :: Bool;
#use_fixed_external_bias = false :: Bool; # default to off
fixed_external_bias_value = (1.2) :: Float64;


# changing the post part of the weight update rule
floor_on_post = (-Inf) :: Float64; # applied in post()
disable_winner_takes_all = false :: Bool; # applied in post()
binary_outputs_mode = false :: Bool; # applied to dw
rescaled_outputs_mode = false :: Bool; # applied to dw
if (binary_outputs_mode) disable_winner_takes_all = true; end # binary outputs and winner takes all are mutually exclusive in weight update code
use_intrinsic_plasticity = false :: Bool; #leave OFF for now! #enable updating, and subtraction of an intrinsic plasticity factor from post
use_weight_normalisation = true :: Bool; # weight normalisation using quadratic norm, multiplicative rule
use_decision_criterion_learner = true :: Bool;
use_pooled_scaling_of_post_population_for_decisions = true :: Bool;
if (!use_pooled_scaling_of_post_population_for_decisions) no_pop_scaling_post_neurons = 1; end # faster way to make sure that post applies noise correctly

# problem difficulty parameters
problem_left_bound = (-1.) :: Float64; #-0.5;
problem_right_bound = (1.) :: Float64; #0.5;

running_av_window_length = 50 :: Int; #50::Int;

learning_rate = (0.002) :: Float64; #(0.000002) #(0.00000001) #(0.0001) linear #(0.02) binary #(0.0001) #(0.0025) #(0.00008); #(0.001); #(0.0001); #0.00012 :: Float64; #0.00001 for debugging # 0.00012 was pretty good with Henning # 0.001; #0.002;
learning_rate /= sqrt(no_pop_scaling_post_neurons) :: Float64; # rescale for population noise, try to keep signal-to-noise ratio constant while maintaining same rate of learning
output_noise_variance = 10.0^2; #3.5; #sqrt(10.0) :: Float64; #10.0;

initial_weight_bias = (2.0); #(2.0) :: Float64; # 2.0
gaussian_weight_bias = (0.5) :: Float64;

# weight constraints
weights_upper_bound = (1e3) #(10.0) #(1e10) #(Inf) #(10.0) #(Inf) :: Float64;
weights_lower_bound = (-1e3) #(-10.0) #(-1e10) #(-10.0) #(-Inf) :: Float64;


# criterion learning
# as a simplification of both intrinsic plasticity and weight normalisation
#   here we use a simple decision bias measure
#   associate 1 with output 2 and 0 with output 1
decision_criterion_timescale = 10.0 :: Float64; # (dt/tau for updating of decision_criterion_monitor)
reset_decision_criterion_monitor_on_each_block = false :: Bool;
use_reset_decision_criterion_monitor_each_subject = true :: Bool;


# intrinsic plasticity
intrinsic_baseline = [0.0, 0.0] :: Array{Float64,1};
intrinsic_plasticity_window_length = 10 :: Int;
use_intrinsic_plasticity_with_noise = true :: Bool;
use_intrinsic_plasticity_with_wta_form = true :: Bool; #if WTA is on you still need to decide whether to use it in the intrinsic plasticity or not
use_reset_average_post_on_each_block = false :: Bool;


# choose input sequence
use_cts_random_inputs = false :: Bool;
use_binary_alternating_inputs = false :: Bool;
use_binary_random_inputs = true :: Bool;
use_biased_cts_random_inputs = false :: Bool;
# (criterion learning bias)
input_sequence_bias = (-0.250) :: Float64; # should be between -0.5 (L) and +0.5 (R), this is the x value bias
criterion_learner_expectation = (0.5) :: Float64; # between 0 (L) and 1 (R), expectation of right with respect to left presentations
print("Stimulus sequence (dx) ratio (L:R): $(1-(0.5+input_sequence_bias)):$(0.5+input_sequence_bias)\n");

# task sequence
task_sequence_bias = (0.0) :: Float64; # should be between -0.5 and +0.5, gives (1-(0.5+bias)):(0.5+bias) ratio of tasks, this is the task_id value
print("Bisection task ratio (1:2): $(1-(0.5+task_sequence_bias)):$(0.5+task_sequence_bias)\n");

# selective tuning of input
input_baseline = 2.0 :: Float64; #2.0;
input_baseline_variance = 0.5^2; #0.25; #0.5 :: Float64; #0.5;
task_tuning_slope_variance = zeros(no_input_tasks) :: Array{Float64,1};
task_tuning_slope_variance[1] = 0.5^2; #0.5^2; #0.5^2; #0.7^2; #0.5^2; #0.4; #0.375; #0.25; #0.375 :: Float64; # easy task
task_tuning_slope_variance[2] = 0.2^2; #0.2^2; #0.2^2; #0.25^2; #0.25; #0.0625; #0.25 :: Float64; # hard task


# input tuning function
# empty constructors utilise multiple dispatch in selecting tuning function
abstract TuningSelector
type gaussian_tc <: TuningSelector end
type linear_tc <: TuningSelector end
use_gaussian_tuning_function = false ::Bool;
use_linear_tuning_function = true ::Bool;
no_tuning_curves_per_input_neuron = 1::Int; # as this increases the inputs increase in magnitude! (consider using normalise option)
normalise_height_of_multiple_gaussian_inputs = true::Bool;
gaussian_tuning_mu_lower_bound = (-1.0) :: Float64;
gaussian_tuning_mu_upper_bound = (1.0) :: Float64;
fix_tuning_gaussian_width = true :: Bool;
gaussian_tuning_sigma_mean = 0.25 :: Float64;
gaussian_tuning_sigma_variance = 0.1 :: Float64;
gaussian_tuning_height_mean = 2.0 :: Float64;
gaussian_tuning_height_variance = 0.25 :: Float64;

# discrimination threshold calculation
perform_detection_threshold = false :: Bool;
perform_post_hoc_detection_threshold = false :: Bool;
detection_threshold = 0.25 :: Float64;

# this mimics the same subject in each experiment, rather than new subjects throughout
use_ab_persistence = false :: Bool;

# Reward can be saved from the state of the running average or on a per block basis
#	reward averaging on a per task basis is only on a per block basis
save_reward_from_running_average = true :: Bool;

# first block of each experiment is just used to build up a running average
const disable_learning_on_first_block = true :: Bool;

# used in high_dim_array2 setup, might be useful to keep generic rather than putting the int into code directly
const no_classifications_per_task = 2::Int;


# the following is development code, it does not currently work
use_defined_performance_setup = false :: Bool;
defined_performance_task_1 = 0.6 :: Float64;
defined_performance_task_2 = 0.3 :: Float64;


# Verbosity of console output:
#   (-1) : You only see the beginning of each experiment headers
#   0 : You just see the passing of the block beginnings
#   1 : You see the individual trial choices
#   2 : You see the interpretation of the trial choices
#   3 : You see the summary output at the end of each block
#   4 : You see the individual weight update norms
verbosity = (-1) :: Int;

# plotting options
plotting_scatter_plot_on = true; #true :: Bool; # dots from scatter plot showing individual subject results
plotting_individual_subjects_on = true; #true :: Bool; # lines joining the individual subject data points
plotting_error_bars_on = false :: Bool; # standard deviation from the mean
use_plot_mean = false :: Bool; # plot mean rather than median for proportions correct
# The following two are not equivalent
plotting_task_by_task_on = true :: Bool; # Separated task plotting in roving experiment
plotting_separate_choices_on = true :: Bool; # Separated proportion correct plotting in each experiment, eg. for left versus right
