using PyPlot
using Distributions

function initialise(no_trials::Int, no_tasks=2::Int64)
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

  W = ones(3,1) * 0.75;

  return (task_sequence, W);
end


function get_inputs(task_id)
  x = zeros(Float64, 3, 1);

  if task_id == 1
    x[1] = 1;
  else
    x[3] = 1;
  end
  x[2] = 0.5;

  return x;
end


function get_output(x, W)
  return x' * W;
end


function modify_W!(x, y, target, W)
  alpha = 1.0;
  tau = 10;
  error = (1./tau) * (target - y);

  # δW = zeros(3,1);
  δW = alpha * error .* x;
  W[1] += δW[1];
  W[2] += δW[2];
  W[3] += δW[3];
  # @show W
end


function run_matrix()
  no_trials = 200;
  initial_contingency = 0.75;
  switch_point = 100;
  second_contingencies = [1.0; 0.5];
  (task_sequence, W) = initialise(no_trials);

  (single_sequence, W_single) = initialise(no_trials,1);

  outputs = zeros(no_trials, 1);

  outputs_1 = zeros(no_trials,1);
  outputs_2 = zeros(no_trials,1);

  outputs_single = zeros(round(Int, no_trials/2), 1);

  for i = 1:no_trials
    x = get_inputs(task_sequence[i]);
    y = get_output(x, W);
    outputs[i] = y[1];

    outputs_1[i] = get_output(get_inputs(1), W)[1];
    outputs_2[i] = get_output(get_inputs(2), W)[1];

    if i % 2 == 0
      outputs_single[round(Int,i/2)] = get_output(get_inputs(single_sequence[i]), W_single)[1];
      if i < switch_point
        modify_W!(get_inputs(single_sequence[i]),outputs_single[round(Int,i/2)],initial_contingency,W_single);
      else
        modify_W!(get_inputs(single_sequence[i]),outputs_single[round(Int,i/2)],second_contingencies[single_sequence[i]],W_single);
      end
    end

    if i == switch_point
      print("Switching contingencies\n");
    end

    if i < switch_point
      modify_W!(x,y,initial_contingency,W);
    else
      modify_W!(x,y,second_contingencies[task_sequence[i]],W);
    end

    @show W task_sequence[i]
  end

  figure()
  # plot(linspace(1,no_trials,no_trials), outputs, "b", linewidth=3);
  plot(linspace(1,no_trials,no_trials), outputs_1, "r", label="Task 1");
  plot(linspace(1,no_trials,no_trials), outputs_2, "g", label="Task 2");

  plot(linspace(1,no_trials,no_trials/2), outputs_single, "k", label="Only learning Task 1, every second step")

  title("Matrix critic, overlapping inputs")
  ylabel("abstract reward/performance unit")
  xlabel("trial number")
  savefig("matrix_critic_overlapping_inputs.pdf")
end
