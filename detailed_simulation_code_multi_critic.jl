########## Main simulation functions #############


# putting noise updates in a function (which must be called!)
#  rather than in the post() function, for debugging reasons
function update_noise()
  global ksi = rand(Normal(0,output_noise), no_post_neurons)
end


function initialise_pre_population()
  # I think that a and b only get set once for this paper!
  #   they are receptive fields rather than 'noise'
  global a = rand(Normal(input_baseline,input_baseline_variance), no_pre_neurons);
  beta = 0.375; # easier problem
  pop1 = rand(Normal(0,beta), ((int)(no_pre_neurons/2)));
  beta = 0.25; # harder problem
  pop2 = rand(Normal(0,beta), (int)(no_pre_neurons/2));
  global b = [pop1; pop2]
  if(no_pre_neurons % 2 == 1)
    print("An odd number of presynaptic neurons was selected, this is going to lead to trouble.\n")
  end
end


function initialise_weight_matrix()
  #set initial weights
  global w = rand(Uniform(0,1), (no_pre_neurons, no_post_neurons));
  w[:,1] += -initial_weight_bias*b;
  w[:,2] += initial_weight_bias*b;
  #global w0 = deepcopy(w);
end


function initialise()
  srand(random_seed);

  initialise_pre_population();
  update_noise();
  initialise_weight_matrix();

  global average_delta_reward = 0.0;
  global average_choice = 0.0;
  global n = 0 :: Int; # use this to monitor trial ID per block
  # changing to multi-critic model
  global n_critic = int(zeros(no_task_critics, no_choices_per_task_critics)); # use this to monitor trial ID per critic per block
  global average_reward = zeros(no_task_critics, no_choices_per_task_critics);

  global instance_correct = 0;
  global instance_incorrect = 0;

  global proportion_1_correct = 0.0;
  global proportion_2_correct = 0.0; 

  global exp_results = Array(RovingExperiment, 0);

  print("RND seeded $random_seed\n")
end


__init__ = initialise();


# pre-synaptic firing rate upon presentation of pattern x
function pre(x::Float64, is_problem_1::Bool)
  # Problem/task 1 is presented to first half of pre neurons, second problem is presented
  #  only to second half of pre neurons
  # All pre neurons are linearly selective to their chosen input vector via: a+x.b
  if (is_problem_1)
    #print("DEBUG: p1 true\n")
    #return [(a[1:no_pre_neurons/2] + b[1:no_pre_neurons/2] * x); a[no_pre_neurons/2 + 1 : end]]
    return [(a[1:no_pre_neurons/2] + b[1:no_pre_neurons/2] * x); zeros(int(no_pre_neurons/2))]
  else # problem_2
    #print("DEBUG: p1 false\n")
    #return [a[1:no_pre_neurons/2]; a[no_pre_neurons/2 + 1 : end] + b[no_pre_neurons/2 + 1: end] * x]
    return [zeros(int(no_pre_neurons/2)); a[no_pre_neurons/2 + 1 : end] + b[no_pre_neurons/2 + 1: end] * x]
  end
end


# winner takes all
function wta(left::Float64, right::Float64, debug_on::Bool = false)
  # debugging code to highlight negative post firing rates
  if(debug_on)
    if(verbosity > 0)
      if (left < 0)
        print("Flag left -ve\n")
      end
      if (right < 0)
        print("Flag right -ve\n")
      end
    end
  end

	if (left > right)
    if(verbosity > 0)
      if(debug_on)
		    print("Left!\n")
      end
    end
		right = 0
	else
    if(verbosity > 0)
      if(debug_on)
		    print("Right!\n")
      end
		end
    left = 0
	end
	return [left right]
end


# post-synaptic firing rate upon presentation of pattern x
#  no longer generating a new noise value (ksi) on each call,
#  this must be done externally to allow for repeatibility during debug
function post(x::Float64, is_problem_1::Bool, debug_on::Bool=false)
	local_pre = pre(x, is_problem_1)
  noise_free_left = sum(local_pre.*w[:,1]);
  noise_free_right = sum(local_pre.*w[:,2]);
	left = noise_free_left + ksi[1]
	right = noise_free_right+ ksi[2]

  # calculated probability of getting this result given de-noised results and error size
  trial_probability_left = 0.5 + erf((noise_free_left - noise_free_right) / (output_noise / 2.0)) * 0.5;

  if(debug_on)
    if(verbosity > 0)
      print("n: $n, x: $x, left: $left, right: $right,\n noise_free_left: $noise_free_left, noise_free_right: $noise_free_right, trial_probability_left: $trial_probability_left ")
    end
  end
	return wta(left,right, debug_on)
end


# this is the only function which actually knows if things went right or wrong
# instance_correct = 0;
# instance_incorrect = 0;
function reward(x, is_problem_1::Bool)
	local_post = post(x, is_problem_1, true)

  # I've had some trouble with the logic here due to wta() accepting negative inputs
	if ((x > 0) && (local_post[2] != 0))#right
    if(verbosity > 1)
      global instance_correct += 1;
      print("Greater than zero (x: $x)\n") 
    end
		return 1
	elseif (x < 0 && local_post[1] != 0)#left
    if(verbosity > 1)
      instance_correct += 1;
      print("Less than zero (x: $x)\n")
    end
		return 1
	else
    if(verbosity > 1)
      global instance_incorrect += 1;
      print("WRONG\n")
    end
		return -1
	end
end


# we use a mix of global and local variables here to cut down on function parameters
#=average_reward = 0. :: Float64;
average_delta_reward = 0. :: Float64;
n = 0 :: Int;
#Rn = 0. :: Float64;
function running_av_reward(R)
  global n += 1;

  tau_r = running_av_window_length;
	tau = min(tau_r, n)

	Rn = ( (tau - 1) * average_reward + R ) / tau

  global average_delta_reward = ((n-1) * average_delta_reward + (R - average_reward)) / n; # an attempt at a running average for (R-Rn), the weight update reward
  global average_reward = Rn;

	return Rn;
end=#


# individual critics for running rewards
# no_task_critics = 2
# no_choices_per_task_critics = 2
# initialise
#  n = 0
#  average_reward = 0
function multi_critic_running_av_reward(R::Int, taskID::Int, choiceID::Int)
  global n_critic;
  global average_reward;

  n_critic[taskID,choiceID] += 1;

  tau_r = running_av_window_length;
  tau = min(tau_r, n_critic[taskID, choiceID]);

  Rn = ( (tau - 1) * average_reward[taskID, choiceID] + R ) / tau;

  # update average_reward monitor
  average_reward[taskID, choiceID] = Rn;

  return Rn;
end


# average_choice = 0. :: Float64;
function update_weights(x, is_problem_1::Bool, trial_dat::Trial)
  if(verbosity > 3)
    global instance_reward;
    global instance_average_reward;
  end
  global n += 1;

  # don't forget to update noise externally to this function on separate iterations
  local_pre = pre(x, is_problem_1);
  local_post = post(x, is_problem_1);
  local_reward = reward(x, is_problem_1) :: Int; # it is important that noise is not updated between calls to post() and reward()
  if(verbosity > 3)
    instance_reward[n] = local_reward;
  end

  # Save some data for later examination
  trial_dat.task_type = (is_problem_1 ? 1 : 2);
  trial_dat.correct_answer = (x > 0 ? 1 : -1);
  trial_dat.chosen_answer = (local_post[1] < local_post[2] ? 1 : -1) # note sign reversal, to maintain greater than relationship
  trial_dat.got_it_right = (local_reward > 0 ? true : false);

  # monitor average choice per block here
  #   using n independent of critic, for now
  local_choice = (local_post[1] > 0 ? 1 : 2);
  global average_choice = ( (n-1) * average_choice + local_choice ) / (n);

  #running_av_reward(local_reward); # Nicolas is doing this before dw update, so first timestep is less unstable...
  # TODO: need to improve critic response axis and task type bin logic here
  local_critic_response_bin = 1::Int;
  if(no_choices_per_task_critics > 1)
    local_correct_answer = (x > 0 ? 2 : 1);
    local_critic_response_bin = local_correct_answer;
  end
  local_critic_task_bin = 1::Int;
  if(no_task_critics > 1)
    local_critic_task_bin = trial_dat.task_type;
  end
  multi_critic_running_av_reward(local_reward, local_critic_task_bin, local_critic_response_bin);

  # decide which form of average reward we're using here:
  if( use_multi_critic)
    #   multi critic:
    #     currently expanding logic to use number of critics declared in params
    local_average_reward = average_reward[local_critic_task_bin, local_critic_response_bin];
  elseif (use_single_global_critic)
    #   single global critic:
    local_sum_reward = 0.;
    for i=1:no_task_critics
      for j = 1:no_choices_per_task_critics
        local_sum_reward += average_reward[i, j];
      end
    end
    local_average_reward = ( local_sum_reward / (no_task_critics * no_choices_per_task_critics) );
  else
    # Rmax (no running average):
    local_average_reward = 0.;
  end

  # the weight update matrix
  dw = zeros(no_pre_neurons, no_post_neurons);
  dw[:,1] = learning_rate * local_pre[:] * local_post[1] * (local_reward - local_average_reward);
  dw[:,2] = learning_rate * local_pre[:] * local_post[2] * (local_reward - local_average_reward);
  # Save some data for later examination
  trial_dat.reward_received = (local_reward - local_average_reward);
  trial_dat.w = deepcopy(w);
  trial_dat.dw = deepcopy(dw);


  if (verbosity > 3)
    instance_average_reward[n] = local_average_reward;
    reward_signal = (local_reward - local_average_reward)
    print("local_reward-average_reward: $local_reward - $average_reward = $reward_signal\n")
    #TODO: can I find a better measure of dw here?
    l1_norm_dw = sum(abs(dw));
    el_sum_dw = sum(dw);
    print("l1_norm_dw: $l1_norm_dw, element sum dw: $el_sum_dw\n")
    l1_norm_w = sum(abs(w))
    el_sum_w = sum(w)
    print("l1 norm w: $l1_norm_w, element sum w: $el_sum_w\n")
    left_sum_w = sum(w[:,1]);
    right_sum_w = sum(w[:,2]);
    print("before weight change, sum w left: $left_sum_w, sum w right: $right_sum_w\n")
  end
  
  # the weight update
  global w += dw;
  if (verbosity > 3)
    left_sum_w = sum(w[:,1]);
    right_sum_w = sum(w[:,2]);
    print("after weight change, sum w left: $left_sum_w, sum w right: $right_sum_w\n")
  end
  
  trial_dat.mag_dw = sum(abs(dw));
  
  # hard bound weights at +/- 10
  w[w .> weights_upper_bound] = weights_upper_bound;
  w[w .< weights_lower_bound] = weights_lower_bound;

  return (local_reward+1); # make it 0 or 2, rather than +/-1
end

