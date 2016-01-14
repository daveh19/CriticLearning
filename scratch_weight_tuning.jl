using Distributions;

no_pre_neurons_per_task = 50;
no_post_neurons = 2;
no_input_tasks = 2;

output_noise_variance = 10^2;

input_baseline = 2.0 :: Float64; #2.0;
input_baseline_variance = 0.5^2; #0.25; #0.5 :: Float64; #0.5;
task_tuning_slope_variance = zeros(no_input_tasks) :: Array{Float64,1};
task_tuning_slope_variance[1] = 0.5^2; #0.5^2; #0.5^2; #0.7^2; #0.5^2; #0.4; #0.375; #0.25; #0.375 :: Float64; # easy task
task_tuning_slope_variance[2] = 0.2^2;

defined_performance_task_1 = 0.6;
defined_performance_task_2 = 0.4;


abstract TuningSelector
type gaussian_tc <: TuningSelector end
type linear_tc <: TuningSelector end

global a = rand(Normal(0,1), (no_pre_neurons_per_task,no_input_tasks)) .* sqrt(input_baseline_variance) + input_baseline;
global b = zeros(no_pre_neurons_per_task, no_input_tasks);

for i = 1:no_input_tasks
  #b[:,i] = rand(Normal(0, task_tuning_slope_variance[i]), no_pre_neurons_per_task);
  b[:,i] = rand(Normal(0,1), no_pre_neurons_per_task) .* sqrt(task_tuning_slope_variance[i]);
end

function pre(x::Float64, task_id::Int, tuning_type::linear_tc)
  local_pre = zeros(no_pre_neurons_per_task, no_input_tasks);
  local_pre[:,task_id] = collect(a[:,task_id] + b[:,task_id] .* x); # weird vcat error in Julia v0.4
  return local_pre;
end



w = ones(no_pre_neurons_per_task, no_post_neurons, no_input_tasks);
w *= 0.5;
D_rev(p) = 2.0 * sqrt(output_noise_variance) * erfinv(2 * p - 1.0);
positive_difference_in_outputs_1 = D_rev(defined_performance_task_1);
positive_difference_in_outputs_2 = D_rev(defined_performance_task_2);
local_bias = zeros(no_pre_neurons_per_task, no_post_neurons, no_input_tasks);


for i = 1:no_input_tasks
  # currently assuming only two input tasks (+/-1)
  local_pre_1 = pre(-1.0, i, linear_tc());
  local_pre_2 = pre(1.0, i, linear_tc());

  #TODO: dimension mismatch, need to realign these matrices
  local_bias[:,1,i] = positive_difference_in_outputs_1 ./ ( 2 * local_pre_1[:,i] .* b[:,i] );
  local_bias[:,2,i] = positive_difference_in_outputs_1 ./ ( 2 * local_pre_2[:,i] .* b[:,i] ) ;

  w[:,1,i] += local_bias[:,1,i] .* b[:,i];
  w[:,2,i] += local_bias[:,2,i] .* b[:,i];
end
