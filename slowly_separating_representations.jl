# module slowly_separating_representations

using PyPlot
using Distributions

export update_critic_representation, get_reward_prediction, initialise_critic_parameters, set_phase_id

type Critic_Representation
  W :: Array{Float64,2}
  alpha :: Float64
  tau :: Int64
  phase_id :: Int64
  phase_counter :: Int64
end

global my_critic = Critic_Representation(Array{Float64,2}(0,0), 0., 0, 0, 0);

function initialise_critic_parameters()
  global my_critic;

  ## Learned weights initialisation
  W = ones(3,1) * 0.5;

  ## Learning parameters
  alpha = 1;
  tau = 30;

  # Phase must be pased around for compatibility with other methods
  phase_id = 1;
  phase_counter = 0;

  my_critic.W = W;
  my_critic.alpha = alpha;
  my_critic.tau = tau;
  my_critic.phase_id = phase_id;
  my_critic.phase_counter = phase_counter;
end


function initialise_critic_sim(no_trials::Int64, no_tasks=2::Int64)
#function initialise(no_trials::Int, no_tasks=2::Int64)
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

  initialise_critic_parameters();
  W = my_critic.W;

  return (task_sequence, W);
end


function increment_phase_id(phase_length::Int=1000)
  global my_critic;

  phase_id = my_critic.phase_id;
  phase_counter = my_critic.phase_counter;

  # count through the three phases
  phase_counter += 1;
  if phase_counter > phase_length
    phase_id += 1;
    phase_counter = 1;
    # if phase_id == 2
    #   phase_id += 1;
    # elseif phase_id > 3
    #   phase_id = 3;
    # end
    if phase_id > 3
      phase_id = 3;
    end
    print("Incrementing phase_id now\n")
  end

  # @show phase_id phase_counter

  my_critic.phase_id = phase_id;
  my_critic.phase_counter = phase_counter;

  # @show phase_id my_critic.phase_id phase_counter
end


function set_phase_id(phase_id::Int)
  global my_critic;
  my_critic.phase_id = phase_id;
  my_critic.phase_counter = 1;
end


function get_inputs(task_id::Int)
  phase_id = my_critic.phase_id;
  x = zeros(Float64, 3, 1);

  if phase_id == 1
    x = ones(3,1);
    x /= sum(x);
  elseif phase_id == 2
    if task_id == 1
      x[1] = 1;
    else
      x[3] = 1;
    end
    x[2] = 0.5;
    x /= sum(x);
  elseif phase_id == 3
    if task_id == 1
      x[1] = 1;
      x[3] = 0;
    else
      x[1] = 0;
      x[3] = 1;
    end
    x[2] = 0;
  else
    print("Error\n");
  end

  x ./ norm(x);

  return x;
end


function get_output(x, W)
  return x' * W;
end


function modify_W!(x, y, target, W, use_realistic_feedback::Bool=false, change_reward_range::Bool=false)
  # alpha = 1;
  # tau = 50;
  alpha = my_critic.alpha;
  tau = my_critic.tau;

  if change_reward_range
    # convert reward from [-1,+1] to [0,1] internal representation (it's smoother)
    target = (target / 2.) + 0.5;
  end

  if use_realistic_feedback
    # use contingency to generate probabilistic feedback signal
    probability_target = target;
    feedback = ( rand(Uniform(0,1)) < probability_target ? 1 : 0);
    # this code is for debugging, in the backprop host code it originally generated locally a
    #   feedback of {0,1}, in the sim we typically provide feedback of {-1,+1} so
    #   we're briefly going to generate that locally here... for debugging.
    # feedback = ( rand(Uniform(0,1)) < probability_target ? 1 : -1);
    # @show feedback
    target = feedback;
  end

  # delta error algorithm
  error = (1./tau) * (target - y);

  # δW = zeros(3,1);
  δW = alpha * error .* x;
  W[1] += δW[1];
  W[2] += δW[2];
  W[3] += δW[3];

  # Playing with weight normalisation
  # W1[:,1] = W1[:,1] / norm(W1[:,1]) # inputs to neuron 1 in layer 1
  # W1[:,2] = W1[:,2] / norm(W1[:,2]) # inputs to neuron 2 in layer 1
  #
  # W2 = W2 ./ norm(W2)
  # @show W
end


function update_critic_representation(task_id::Int, local_reward::Float64, change_reward_range::Bool=false) # later can make Int of local_reward
  # needs access to x (from get_inputs(task_id)), W, phase_id
  x = get_inputs(task_id);
  y = get_output(x, my_critic.W);

  if change_reward_range
    # convert reward from [-1,+1] to [0,1] internal representation (it's smoother)
    local_reward = (local_reward / 2.) + 0.5;
  end

  modify_W!(x, y, local_reward, my_critic.W, false);
end

function update_critic_representation(task_id::Int, local_reward::Int, change_reward_range::Bool=false)
  # use the Float64 version of this function for now
  update_critic_representation(task_id, float(local_reward), change_reward_range);
end


function get_reward_prediction(task_id::Int, change_reward_range::Bool=false)
  # needs access to x (from get_inputs(task_id)) and W1, W2
  x = get_inputs(task_id);
  rp = get_output(x, my_critic.W);

  if change_reward_range
    # convert to [-1,+1] scale
    ret_val = (rp - 0.5) * 2;
  else
    ret_val = rp;
  end

  return ret_val;
end


function run_matrix(realistic_feedback::Bool=false, change_reward_range::Bool=false)
  global my_critic;
  no_trials = 3000;
  initial_contingency = [0.8, 0.]; #[-0.2; 0.4]; #[0.8; 0.5];
  # switch_point = 100;
  # second_contingencies = [1.0; 0.5];
  phase_length = 1000;

  (task_sequence, W) = initialise_critic_sim(no_trials);

  # (single_sequence, W_single) = initialise(no_trials,1);

  outputs = zeros(no_trials, 1);

  outputs_1 = zeros(no_trials,1);
  outputs_2 = zeros(no_trials,1);

  # outputs_single = zeros(round(Int, no_trials/2), 1);

  for i = 1:no_trials
    # setting phase_id via a global variable is a workaround, to maintain
    #   api compatibility with the backprop_two_layer module
    increment_phase_id(phase_length);

    x = get_inputs(task_sequence[i]);
    y = get_output(x, W);
    # actual performance on the desired task
    outputs[i] = y[1];
    # monitors of potential performance on the two underlying tasks
    #   ie. had I been asked to do task i how would I have done?
    outputs_1[i] = get_output(get_inputs(1), W)[1];
    outputs_2[i] = get_output(get_inputs(2), W)[1];

    modify_W!(x,y,initial_contingency[task_sequence[i]],W,realistic_feedback,change_reward_range);


    # if i % 2 == 0
    #   outputs_single[round(Int,i/2)] = get_output(get_inputs(single_sequence[i]), W_single)[1];
    #   if i < switch_point
    #     modify_W!(get_inputs(single_sequence[i]),outputs_single[round(Int,i/2)],initial_contingency,W_single);
    #   else
    #     modify_W!(get_inputs(single_sequence[i]),outputs_single[round(Int,i/2)],second_contingencies[single_sequence[i]],W_single);
    #   end
    # end

    # if i == switch_point
    #   print("Switching contingencies\n");
    # end

    # if i < switch_point
    #   modify_W!(x,y,initial_contingency,W);
    # else
    #   modify_W!(x,y,second_contingencies[task_sequence[i]],W);
    # end

    @show W task_sequence[i]
  end

  figure()
  plot(linspace(1,no_trials,no_trials), outputs, "b", linewidth=3);
  plot(linspace(1,no_trials,no_trials), outputs_1, "r", label="Task 1");
  plot(linspace(1,no_trials,no_trials), outputs_2, "g", label="Task 2");

  # plot(linspace(1,no_trials,no_trials/2), outputs_single, "k", label="Only learning Task 1, every second step")

  title("Contingencies $initial_contingency. Two phase changes.")
  # title("Matrix critic, slowly separating representations")
  ylabel("abstract reward/performance unit")
  xlabel("trial number")
  savefig("slowly_separating_representations.pdf")
end

# end # slowly_separating_representations module
