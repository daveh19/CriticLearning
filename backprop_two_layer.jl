module backprop_two_layer

export update_critic_representation, get_reward_prediction

using PyPlot
using Distributions

function initialise(no_trials::Int, no_tasks=2::Int64)
  srand(1);
  task_sequence = zeros(Int, no_trials, 1);
  if no_tasks == 2
    for i = 1:no_trials
      task_sequence[i] = (rand(Uniform(0,1)) < 0.5 ? 1 : 2);
    end
  else
    for i = 1:no_trials
      task_sequence[i] = 1;
    end
  end

  W1 = ones(2,2) * 0.5;
  stdev_weight_noise = 0.; #0.001;
  weight_noise = rand(Normal(0,1),2,2) * stdev_weight_noise;
  # weight_noise = zeros(2,2);
  # weight_noise[1,1] = 0.001;
  # weight_noise[1,2] = -0.001;
  # weight_noise[2,1] = -0.001;
  # weight_noise[2,2] = 0.001;
  W1 += weight_noise;

  W2 = ones(2,1) #+ rand(Normal(0,1),2,1) * 0.001;

  return (task_sequence, W1, W2);
end


function get_inputs(task_id::Int)
  x = zeros(Float64, 2, 1);

  x[task_id] = 1.;

  return x;
end


function get_output(x, W)
  return (x' * W)' # + rand(Normal(0,1),2,1) * 0.01;
end

function get_output(x, W1, W2)
  return (x' * W1 * W2) # + rand(Normal(0,1)) * 0.01;
end


function modify_W!(x, y, z, target, W1, W2, use_realistic_feedback::Bool=false)
  alpha = [1.; 1.];
  tau = [500; 5];
  tau = [500; 30];

  if use_realistic_feedback
    # use contingency to generate probabilistic feedback signal
    probability_target = target;
    feedback = ( rand(Uniform(0,1)) < probability_target ? 1 : 0);
    # @show feedback
    target = feedback;
  end
  # error = (1./tau) * (target - y);

  # Backpropagation algorithm: two layers
  #   inputs x, middle y, output z
  #   W1 is inputs to middle layer weights
  #   W2 is middle to output layer weights
  error = zeros(2,1);
  error[1] = (1./tau[1]) * (target - z[1]);
  error[2] = (1./tau[2]) * (target - z[1]);

  δW1 = zeros(2,2);
  # backprop gradient (assume linear transfer functions)
  δW1[1,1] = x[1] * W2[1];
  δW1[1,2] = x[1] * W2[2];
  δW1[2,1] = x[2] * W2[1];
  δW1[2,2] = x[2] * W2[2];
  δW1 *= alpha[1] * error[1];

  δW2 = alpha[2] * error[2] .* y;

  for i = 1:2, j = 1:2
    W1[i,j] += δW1[i,j];
  end
  W2[1] += δW2[1];
  W2[2] += δW2[2];

  # @show W
end


function update_critic_representation(task_id::Int, local_reward::Int)
  # use the Float64 version of this function for now
  local_reward = local_reward * 1.0;
  update_critic_representation(task_id, local_reward);
end

function update_critic_representation(task_id::Int, local_reward::Float64) # later can make Int of local_reward
  #TODO: fill out update_critic_representation function
  #   needs access to x (from get_inputs(task_id)), W1, W2
end

function get_reward_prediction(task_id::Int)
  #TODO: fill out get_reward_prediction function
  #   needs access to x (from get_inputs(task_id)) and W1, W2
  #get_output(x, W1, W2);
  return 0.;
end


function run_matrix(realistic_feedback::Bool=false)
  no_trials = 6000;
  initial_contingency = [0.8; 0.5];
  switch_point = 3000;
  second_contingencies = [1.0; 0.2];

  (task_sequence, W1, W2) = initialise(no_trials);

  outputs = zeros(no_trials, 1);

  outputs_1 = zeros(no_trials,1);
  outputs_2 = zeros(no_trials,1);

  for i = 1:no_trials
    x = get_inputs(task_sequence[i]);
    y = get_output(x, W1);
    z = get_output(x, W1, W2);

    # actual performance on the desired task
    outputs[i] = z[1];
    # monitors of potential performance on the two underlying tasks
    #   ie. had I been asked to do task i how would I have done?
    outputs_1[i] = get_output(get_inputs(1), W1, W2)[1];
    outputs_2[i] = get_output(get_inputs(2), W1, W2)[1];


    if i == switch_point
      print("Switching contingencies\n");
    end

    if i < switch_point
      modify_W!(x,y,z,initial_contingency[task_sequence[i]],W1,W2,realistic_feedback);
    else
      modify_W!(x,y,z,second_contingencies[task_sequence[i]],W1,W2,realistic_feedback);
    end

    @show W1 W2 task_sequence[i]
  end

  figure()
  plot(linspace(1,no_trials,no_trials), outputs, "b", linewidth=3);
  plot(linspace(1,no_trials,no_trials), outputs_1, "r", label="Task 1");
  plot(linspace(1,no_trials,no_trials), outputs_2, "g", label="Task 2");

  title("Contingencies {0.8,0.5} then {1,0.2}. Two-layer using Backprop")
  ylabel("abstract reward/performance unit")
  xlabel("trial number")
  savefig("backprop_two_layer.pdf")
end


function single_task_run_matrix(realistic_feedback::Bool=false)
  no_trials = 6000;
  initial_contingency = [0.8; 0.5];
  switch_point = 3000;
  second_contingencies = [0.5; 0.2];

  (task_sequence, W1, W2) = initialise(no_trials, 1);

  outputs = zeros(no_trials, 1);

  outputs_1 = zeros(no_trials,1);
  outputs_2 = zeros(no_trials,1);

  for i = 1:no_trials
    x = get_inputs(task_sequence[i]);
    y = get_output(x, W1);
    z = get_output(x, W1, W2);

    # actual performance on the desired task
    outputs[i] = z[1];
    # monitors of potential performance on the two underlying tasks
    #   ie. had I been asked to do task i how would I have done?
    outputs_1[i] = get_output(get_inputs(1), W1, W2)[1];
    outputs_2[i] = get_output(get_inputs(2), W1, W2)[1];


    if i == switch_point
      print("Switching contingencies\n");
    end

    if i < switch_point
      modify_W!(x,y,z,initial_contingency[task_sequence[i]],W1,W2,realistic_feedback);
    else
      modify_W!(x,y,z,second_contingencies[task_sequence[i]],W1,W2,realistic_feedback);
    end

    @show W1 W2 task_sequence[i]
  end

  figure()
  plot(linspace(1,no_trials,no_trials), outputs, "b", linewidth=3);
  plot(linspace(1,no_trials,no_trials), outputs_1, "r", label="Task 1");
  # plot(linspace(1,no_trials,no_trials), outputs_2, "g", label="Task 2");

  title("Single task. Contingencies 0.8 and 0.5. Two-layer using Backprop")
  ylabel("abstract reward/performance unit")
  xlabel("trial number")
  legend()
  savefig("backprop_two_layer_single_task.pdf")
end


function crossover_run_matrix(realistic_feedback::Bool=false)
  no_trials = 8000;
  initial_contingency = [0.8; 0.5];
  switch_point = 3000;
  second_contingencies = [0.3; 0.7];

  (task_sequence, W1, W2) = initialise(no_trials);

  outputs = zeros(no_trials, 1);

  outputs_1 = zeros(no_trials,1);
  outputs_2 = zeros(no_trials,1);

  for i = 1:no_trials
    x = get_inputs(task_sequence[i]);
    y = get_output(x, W1);
    z = get_output(x, W1, W2);

    # actual performance on the desired task
    outputs[i] = z[1];
    # monitors of potential performance on the two underlying tasks
    #   ie. had I been asked to do task i how would I have done?
    outputs_1[i] = get_output(get_inputs(1), W1, W2)[1];
    outputs_2[i] = get_output(get_inputs(2), W1, W2)[1];


    if i == switch_point
      print("Switching contingencies\n");
    end

    if i < switch_point
      modify_W!(x,y,z,initial_contingency[task_sequence[i]],W1,W2,realistic_feedback);
    else
      modify_W!(x,y,z,second_contingencies[task_sequence[i]],W1,W2,realistic_feedback);
    end

    @show W1 W2 task_sequence[i]
  end

  figure()
  plot(linspace(1,no_trials,no_trials), outputs, "b", linewidth=3);
  plot(linspace(1,no_trials,no_trials), outputs_1, "r", label="Task 1");
  plot(linspace(1,no_trials,no_trials), outputs_2, "g", label="Task 2");

  title("Contingencies {0.8,0.5} then {0.3,0.7}. Two-layer using Backprop")
  ylabel("abstract reward/performance unit")
  xlabel("trial number")
  savefig("backprop_two_layer_crossover.pdf")
end


function reverse_run_matrix(realistic_feedback::Bool=false)
  no_trials = 10000;
  initial_contingency = [0.8; 0.5];
  switch_point = 3000;
  second_contingencies = [0.6; 0.6];

  (task_sequence, W1, W2) = initialise(no_trials);

  outputs = zeros(no_trials, 1);

  outputs_1 = zeros(no_trials,1);
  outputs_2 = zeros(no_trials,1);

  for i = 1:no_trials
    x = get_inputs(task_sequence[i]);
    y = get_output(x, W1);
    z = get_output(x, W1, W2);

    # actual performance on the desired task
    outputs[i] = z[1];
    # monitors of potential performance on the two underlying tasks
    #   ie. had I been asked to do task i how would I have done?
    outputs_1[i] = get_output(get_inputs(1), W1, W2)[1];
    outputs_2[i] = get_output(get_inputs(2), W1, W2)[1];


    if i == switch_point
      print("Switching contingencies\n");
    end

    if i < switch_point
      modify_W!(x,y,z,initial_contingency[task_sequence[i]],W1,W2,realistic_feedback);
    else
      modify_W!(x,y,z,second_contingencies[task_sequence[i]],W1,W2,realistic_feedback);
    end

    @show W1 W2 task_sequence[i]
  end

  figure()
  plot(linspace(1,no_trials,no_trials), outputs, "b", linewidth=3);
  plot(linspace(1,no_trials,no_trials), outputs_1, "r", label="Task 1");
  plot(linspace(1,no_trials,no_trials), outputs_2, "g", label="Task 2");

  title("Contingencies {0.8,0.5} then {0.6,0.6}. Two-layer using Backprop")
  ylabel("abstract reward/performance unit")
  xlabel("trial number")
  savefig("backprop_two_layer_reverse.pdf")
end


end # module backprop_two_layer
