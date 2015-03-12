########## Main simulation functions #############

# putting noise updates in a function (which must be called!)
#  rather than in the post() function, for debugging reasons
function update_noise()
  global ksi = rand(Normal(0,output_noise), no_post_neurons);
end


function initialise_pre_population(tuning_type::linear_tc)
  # a and b only get set once for this paper
  #   they are receptive fields rather than 'noise'
  global a = rand(Normal(input_baseline, input_baseline_variance), (no_pre_neurons_per_task, no_input_tasks));
  global b = zeros(no_pre_neurons_per_task, no_input_tasks);

  for i = 1:no_input_tasks
    b[:,i] = rand(Normal(0, task_tuning_slope_variance[i]), no_pre_neurons_per_task);
  end
end

function initialise_pre_population(tuning_type::gaussian_tc)
  # Multiple dispatch should use this function rather than the linear
  #   receptive fields function if a boolean is passed as a parameter
  global a;
  # Note: there is no b generated by this function!

  #CONSIDER: should probably extend mu beyond (-1,1) for complete coverage of interval
  #CONSIDER: should I reduce height by no_tuning_curves_per_input_neuron?
  a = Array(gaussian_tc_type, (no_pre_neurons_per_task, no_input_tasks) );
  for i = 1:no_input_tasks;
    #figure()
    for j=1:no_pre_neurons_per_task;
      tuning_mu = rand(Uniform(gaussian_tuning_mu_lower_bound,gaussian_tuning_mu_upper_bound), no_tuning_curves_per_input_neuron);
      if (fix_tuning_gaussian_width)
        tuning_sigma = ones(no_tuning_curves_per_input_neuron);
        tuning_sigma *= gaussian_tuning_sigma;
      else
        tuning_sigma = rand(Normal(gaussian_tuning_sigma, gaussian_tuning_sigma_width), no_tuning_curves_per_input_neuron);
      end
      tuning_height = rand(Normal(gaussian_tuning_height,gaussian_tuning_height_variance), no_tuning_curves_per_input_neuron);
      if ( normalise_height_of_multiple_gaussian_inputs )
        tuning_height = tuning_height ./ no_tuning_curves_per_input_neuron;
      end
      a[j,i] = gaussian_tc_type(no_tuning_curves_per_input_neuron, tuning_mu, tuning_sigma, tuning_height);
  
      #scatter(tuning_mu, tuning_height, c="r");
      #scatter(tuning_mu, tuning_sigma, c="b");
    end
  end
end


function plot_gaussian_tuning_single_input(neuron_id::Int=1, task_id::Int=1)
  #figure();
  x = linspace(-1,1,101);
  y = zeros(101);
  for i = 1:101
    #local_pre = pre(x[i], task_id, gaussian_tc() );
    #y[i] = sum(local_pre);
    for j = 1:a[neuron_id, task_id].no_curves
      f(x) = a[neuron_id, task_id].height[j] .* exp( -(x - a[neuron_id,task_id].mu[j]).^2 ./ (2 * ( a[neuron_id, task_id].sigma[j] .^2 ) ) );
      y[i] += f(x[i]);
    end
  end

  plot(x,y);
end

function plot_gaussian_tuning_multi_inputs(task_id::Int=1, begin_id::Int=1, end_id::Int=no_pre_neurons_per_task)
  figure()
  for i = begin_id:end_id
    plot_gaussian_tuning_single_input(i, task_id);
  end
  xlim([-1,1])
  ylim([0,3])
  title("Input layer tuning curves, direction $task_id")
  xlabel("Input range")
  ylabel("Neuronal firing rate")
end


function initialise_weight_matrix(tuning_type::gaussian_tc)
  # Remember: always call this after a and b have already been initialised!
  #set initial weights
  global w = rand(Uniform(0,1), (no_pre_neurons_per_task, no_post_neurons, no_input_tasks));
  #w = ones(no_pre_neurons_per_task, no_post_neurons, no_input_tasks);
  
  centre_of_mass = zeros(no_pre_neurons_per_task, no_input_tasks);
  #CONSIDER: adding height to centre of mass calculation (sum(mu.*h)/no_curves)
  for i = 1:no_input_tasks
    for j = 1:no_pre_neurons_per_task
      centre_of_mass[j, i] = sum(a[j,i].mu) ./ a[j,i].no_curves ;
    end
    w[:,1,i] += -gaussian_weight_bias.*centre_of_mass[:,i];
    w[:,2,i] += gaussian_weight_bias.*centre_of_mass[:,i];
  end

  # hard bound weights at +/- 10
  w[w .> weights_upper_bound] = weights_upper_bound;
  w[w .< weights_lower_bound] = weights_lower_bound;
end

function initialise_weight_matrix(tuning_type::linear_tc)
  # Remember: always call this after a and b have already been initialised!
  #set initial weights
  global w = rand(Uniform(0,1), (no_pre_neurons_per_task, no_post_neurons, no_input_tasks));
  for i = 1:no_input_tasks
    w[:,1,i] += -initial_weight_bias.*b[:,i];
    w[:,2,i] += initial_weight_bias.*b[:,i];
  end
end


function initialise()
  srand(random_seed);

  if(use_gaussian_tuning_function)
    # use gaussian basis functions
    tuning_type = gaussian_tc();
  elseif(use_linear_tuning_function)
    # use linear tuning functions
    tuning_type = linear_tc();
  else
    print("ERROR: you need to define a tuning function\n");
    error(1);
  end

  initialise_pre_population(tuning_type);
  update_noise();
  initialise_weight_matrix(tuning_type);

  global average_delta_reward = 0.0;
  global average_choice = 0.0;
  #global n = 0 :: Int; # use this to monitor trial ID per block (very important: this is a block level counter!)
  global n_within_block = 0 :: Int; # use this to monitor trial ID per block (very important: this is a block level counter!)
  global n_task_within_block = zeros(Int, no_input_tasks) :: Array{Int,1};
  # changing to multi-critic model
  #   critic can be per block or over entire learning history
  global n_critic = zeros(Int, no_task_critics, no_choices_per_task_critics); # use this to monitor trial ID per critic
  global average_reward = zeros(no_task_critics, no_choices_per_task_critics); # running average, stored values represent end of a block value
  global average_block_reward = 0.0;
  global instance_correct = 0;
  global instance_incorrect = 0;

  global proportion_1_correct = 0.0;
  global proportion_2_correct = 0.0; 

  global exp_results = Array(RovingExperiment, 0);

  global enable_weight_updates = true::Bool;

  print("RND seeded $random_seed\n")
end


__init__ = initialise();


# pre-synaptic firing rate upon presentation of pattern x
function pre(x::Float64, task_id::Int, tuning_type::linear_tc)
  local_pre = zeros(no_pre_neurons_per_task, no_input_tasks);
  local_pre[:,task_id] = [(a[:,task_id] + b[:,task_id] .* x)];
  return local_pre;
end

# pre-synaptic firing rate upon presentation of pattern x
#   using gaussian based tuning curves
#   current version still maintains task specific populations
function pre(x::Float64, task_id::Int, tuning_type::gaussian_tc)
  local_pre = zeros(no_pre_neurons_per_task, no_input_tasks);

  for neuron_id = 1:no_pre_neurons_per_task
    for j = 1:a[neuron_id, task_id].no_curves
      f(x) = a[neuron_id, task_id].height[j] .* exp( -(x - a[neuron_id,task_id].mu[j]).^2 ./ (2 * ( a[neuron_id, task_id].sigma[j] .^2 ) ) );
      local_pre[neuron_id, task_id] += f(x);
    end
  end

  return local_pre;
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
# Note: local_post returns a tuple where one value is 0. All comparisons to find the non zero value should use absolute comparison.
function post(x::Float64, task_id::Int, tuning_type::TuningSelector, debug_on::Bool=false)
	local_pre = pre(x, task_id, tuning_type)
  
  noise_free_left = sum(local_pre[:,task_id] .* w[:,1,task_id]);
  noise_free_right = sum(local_pre[:,task_id] .* w[:,2,task_id]);
	
  left = noise_free_left + ksi[1]
	right = noise_free_right+ ksi[2]

  # calculated probability of getting this result given de-noised results and error size
  #   TODO: finish this code
  trial_probability_left = 0.5 + erf((noise_free_left - noise_free_right) / (output_noise / 2.0)) * 0.5;

  if(debug_on)
    if(verbosity > 0)
      print("n_within_block: $n_within_block, x: $x, left: $left, right: $right,\n noise_free_left: $noise_free_left, noise_free_right: $noise_free_right, trial_probability_left: $trial_probability_left ")
    end
  end
	return wta(left,right, debug_on)
end


function post_hoc_calculate_thresholds(tuning_type::TuningSelector, subjects::Array{Subject,2}, split_output::Bool=false)
  # globals required for correct processing of pre()
  global a,b,w;
  global no_pre_neurons_per_task;
  global no_input_tasks;

  # The second dimension of subjects is basically a separate experimental
  #   protocol. Call this variable no_experimental_tasks_dimension
  #   to make it stand out
  (local_no_subjects, local_no_experimental_tasks_dimension) = size(subjects);
  no_points = 30;
  x = linspace(0,1,no_points);

  for j = 1:local_no_experimental_tasks_dimension
    for i = 1:local_no_subjects
      a = deepcopy(subjects[i,j].a);
      if( isa(tuning_type, linear_tc) )
        b = deepcopy(subjects[i,j].b);
      end

      # Calculate pre() for an entire linspace of inputs for this subject
      #   This is the heavy part of the processing which I wanted to reduce
      (no_pre_neurons_per_task, no_input_tasks) = size(a);
      local_pre_pos = zeros(no_pre_neurons_per_task, no_input_tasks, no_points);
      local_pre_neg = zeros(no_pre_neurons_per_task, no_input_tasks, no_points);
      for task_id = 1:no_input_tasks
        for m = 1:no_points
          # Assuming that pre returns zero entries for all non task specific entries
          local_pre_pos[:,task_id,m] = pre(x[m], task_id, tuning_type)[task_id];
          local_pre_neg[:,task_id,m] = pre(-x[m], task_id, tuning_type)[task_id];
        end
      end
      
      # Calculate threshold for this subject using his trial by trial
      #   weight matrix and the associated task_id
      local_no_blocks_per_experiment = length(subjects[i,j].blocks);
      local_no_trials_per_block = length(subjects[i,j].blocks[1].trial);
      for k = 1:local_no_blocks_per_experiment
        for l = 1:local_no_trials_per_block
          # finally we get to processing a single threshold calculation
          task_id = subjects[i,j].blocks[k].trial[l].task_type;
          w = deepcopy(subjects[i,j].blocks[k].trial[l].w);
          
          x = linspace(0,1,no_points); # re-initialise due to sorting in loop below
          error_rate = zeros(no_points);
          split_error = zeros(no_points,2)

          # Loop over the xi in x, current implementation not really amenable
          #   to non-loop slicing
          for m = 1:no_points
            # calculate noise free post for xi
            local_noise_free_post_pos_left = sum(local_pre_pos[:,task_id,m].*w[:,1,task_id]);
            local_noise_free_post_pos_right = sum(local_pre_pos[:,task_id,m].*w[:,2,task_id]);

            # calculate noise free post for -xi
            local_noise_free_post_neg_left = sum(local_pre_neg[:,task_id,m].*w[:,1,task_id]);
            local_noise_free_post_neg_right = sum(local_pre_neg[:,task_id,m].*w[:,2,task_id]);
            if(verbosity > 2)
              print("DEBUG: $local_noise_free_post_pos_left, $local_noise_free_post_pos_right, ")
              print("$local_noise_free_post_neg_left, $local_noise_free_post_neg_right, ")
            end

            # probability, for a positive input (i) that we choose left
            p_pos_left = 0.5 + 0.5 * erf( (local_noise_free_post_pos_left - local_noise_free_post_pos_right) / (output_noise / 2.0) );
            p_pos_right = (1. - p_pos_left);
    
            if(verbosity > 2)
              print("p_pos_left: $p_pos_left, p_pos_right: $p_pos_right ")
            end

            # probability, for a negative input (-i) that we choose left
            p_neg_left = 0.5 + 0.5 * erf( (local_noise_free_post_neg_left - local_noise_free_post_neg_right) / (output_noise / 2.0) );
            p_neg_right = (1. - p_neg_left);
            if(verbosity > 2)
              print("p_neg_left: $p_neg_left, p_neg_right: $p_neg_right,")
            end

            # probability of choosing the wrong side
            error_rate[m] = ( p_pos_left + p_neg_right ) / 2.0;
            split_error[m,1] = p_pos_left;
            split_error[m,2] = p_neg_right; # prob for a negative input we falsely choose right of zero

            if(verbosity > 1)
              print(" m: $m, error_rate: ", error_rate[m],", error_left: ", split_error[m,1], ", error_right: ", split_error[m,2], "\n")
            end
          end # loop over m:no_points

          # do a sort to enforce increasing error rates
          A = [error_rate x];
          A = sortrows(A, by=x->(x[1],-x[2]), rev=false)

          # linear interpolator from target error rate back to value on input space producing this error rate
          z = InterpIrregular(A[:,1], A[:,2], 1, InterpLinear); # could also use BCnan as non-interp value

          if(!split_output)
            if(verbosity > 1)
              print("n_within_block: $n_within_block, z: ", z[detection_threshold], "\n");
            end
            subjects[i,j].blocks[k].trial[l].error_threshold = z[detection_threshold];
            #return z[detection_threshold];
          else # not currently in use
            Al = [split_error[:,1] x];
            Al = sortrows(Al, by=x->(x[1],-x[2]), rev=false); #sort by ascending error, and descending distance from 0
            zl = InterpIrregular(Al[:,1],Al[:,2], 1, InterpLinear);
    
            Ar = [split_error[end:-1:1,2] x[end:-1:1]];
            Ar = sortrows(Ar, by=x->(x[1],-x[2]), rev=false);
            zr = InterpIrregular(Ar[:,1],Ar[:,2], 1, InterpLinear);

            print("Al: ",Al,"\n")
            print("Ar: ",Ar,"\n")
            if(verbosity > 1)
              print("n_within_block: $n_within_block, z: ", z[detection_threshold], ", zl: ", zl[detection_threshold], ", zr: ", zr[detection_threshold],"\n");
            end

            #return [zl[detection_threshold] zr[detection_threshold]];
          end

        end # loop over trials per block
      end # loop over blocks per experiment

    end # loop over subjects in an experiment class
  end # loop over experiments classes for a group of subjects
  print("Post-hoc calculation of thresholds completed.\n\nNote: subject currently in memory has changed\n");
end

function detect_threshold(tuning_type::TuningSelector, task_id::Int=1, split_output::Bool=false)
  # find the detection threshold with current weight matrix and current subject
  no_points = 30;
  error_rate = zeros(no_points);
  split_error = zeros(no_points,2)
  x = linspace(0,1,no_points); # linspace of x values for detection threshold
  i = 1;
  for xi in x
    # calculate pre for +/- xi
    local_pre_pos = pre(xi, task_id, tuning_type);
    local_pre_neg = pre(-xi, task_id, tuning_type);

    #print("DEBUG: $local_pre_pos, $local_pre_neg ")

    # calculate noise free post for xi
    local_noise_free_post_pos_left = sum(local_pre_pos[:,task_id].*w[:,1,task_id]);
    local_noise_free_post_pos_right = sum(local_pre_pos[:,task_id].*w[:,2,task_id]);

    # calculate noise free post for -xi
    local_noise_free_post_neg_left = sum(local_pre_neg[:,task_id].*w[:,1,task_id]);
    local_noise_free_post_neg_right = sum(local_pre_neg[:,task_id].*w[:,2,task_id]);
    if(verbosity > 2)
      print("DEBUG: $local_noise_free_post_pos_left, $local_noise_free_post_pos_right, ")
      print("$local_noise_free_post_neg_left, $local_noise_free_post_neg_right, ")
    end

    # probability, for a positive input (i) that we choose left
    p_pos_left = 0.5 + 0.5 * erf( (local_noise_free_post_pos_left - local_noise_free_post_pos_right) / (output_noise / 2.0) );
    p_pos_right = (1. - p_pos_left);
    
    if(verbosity > 2)
      print("p_pos_left: $p_pos_left, p_pos_right: $p_pos_right ")
    end

    # probability, for a negative input (-i) that we choose left
    p_neg_left = 0.5 + 0.5 * erf( (local_noise_free_post_neg_left - local_noise_free_post_neg_right) / (output_noise / 2.0) );
    p_neg_right = (1. - p_neg_left);
    if(verbosity > 2)
      print("p_neg_left: $p_neg_left, p_neg_right: $p_neg_right,")
    end

    # probability of choosing the wrong side
    error_rate[i] = ( p_pos_left + p_neg_right ) / 2.0;
    split_error[i,1] = p_pos_left;
    split_error[i,2] = p_neg_right; # prob for a negative input we falsely choose right of zero

    if(verbosity > 1)
      print(" i: $i, xi: $xi, error_rate: ", error_rate[i],", error_left: ", split_error[i,1], ", error_right: ", split_error[i,2], "\n")
    end
    i += 1;
  end

  # do a sort to enforce increasing error rates
  A = [error_rate x];
  A = sortrows(A, by=x->(x[1],-x[2]), rev=false)

  # linear interpolator from target error rate back to value on input space producing this error rate
  z = InterpIrregular(A[:,1], A[:,2], 1, InterpLinear); # could also use BCnan as non-interp value

  if(!split_output)
    if(verbosity > 1)
      print("n_within_block: $n_within_block, z: ", z[detection_threshold], "\n");
    end
    return z[detection_threshold];
  else
    Al = [split_error[:,1] x];
    Al = sortrows(Al, by=x->(x[1],-x[2]), rev=false); #sort by ascending error, and descending distance from 0
    zl = InterpIrregular(Al[:,1],Al[:,2], 1, InterpLinear);
    
    Ar = [split_error[end:-1:1,2] x[end:-1:1]];
    Ar = sortrows(Ar, by=x->(x[1],-x[2]), rev=false);
    zr = InterpIrregular(Ar[:,1],Ar[:,2], 1, InterpLinear);

    print("Al: ",Al,"\n")
    print("Ar: ",Ar,"\n")
    if(verbosity > 1)
      print("n_within_block: $n_within_block, z: ", z[detection_threshold], ", zl: ", zl[detection_threshold], ", zr: ", zr[detection_threshold],"\n");
    end

    return [zl[detection_threshold] zr[detection_threshold]];
  end
end


# this is the only function which actually knows if things went right or wrong
# instance_correct = 0;
# instance_incorrect = 0;
function reward(x::Float64, task_id::Int, tuning_type::TuningSelector)
	local_post = post(x, task_id, tuning_type, true)

  # I've had some trouble with the logic here due to wta() accepting negative inputs
	if ((x > 0) && (abs(local_post[2]) > 0))#right
    if(verbosity > 1)
      global instance_correct += 1;
      print("Greater than zero (x: $x)\n") 
    end
		return (1);
	elseif ((x <= 0) && (abs(local_post[1]) > 0))#left
    if(verbosity > 1)
      instance_correct += 1;
      print("Less than zero (x: $x)\n")
    end
		return (1);
	else
    if(verbosity > 1)
      global instance_incorrect += 1;
      print("WRONG\n")
    end
		return (-1);
	end
end


# individual critics for running rewards
# no_task_critics = 2
# no_choices_per_task_critics = 2
# initialise
#  n = 0
#  average_reward = 0
function multi_critic_running_av_reward(R::Int, task_critic_id::Int, choice_critic_id::Int)
  global n_critic;
  global average_reward;
  
  tau_r = running_av_window_length;

  if (n_critic[task_critic_id, choice_critic_id] < tau_r)
    n_critic[task_critic_id, choice_critic_id] += 1;
  end
  
  tau = min(tau_r, n_critic[task_critic_id, choice_critic_id]);

  Rn = ( (tau - 1) * average_reward[task_critic_id, choice_critic_id] + R ) / tau;

  # update average_reward monitor
  average_reward[task_critic_id, choice_critic_id] = Rn;

  return Rn;
end


# average_choice = 0. :: Float64;
function update_weights(x::Float64, task_id::Int, tuning_type::TuningSelector, trial_dat::Trial)
  if(verbosity > 3)
    global instance_reward;
    global instance_average_reward;
  end
  global n_within_block += 1;
  global n_task_within_block;
  n_task_within_block[task_id] += 1;

  # don't forget to update noise externally to this function on separate iterations
  local_pre = pre(x, task_id, tuning_type);
  # Note: local_post returns a tuple where one value is 0. All comparisons to find the non zero value should use absolute comparison.
  local_post = post(x, task_id, tuning_type);
  local_reward = reward(x, task_id, tuning_type) :: Int; # it is important that noise is not updated between calls to post() and reward()
  if(perform_detection_threshold)
    local_threshold = detect_threshold(tuning_type, task_id);
    trial_dat.error_threshold = local_threshold;
  end
  if(verbosity > 3)
    instance_reward[n] = local_reward;
  end

  # Save some data for later examination
  trial_dat.task_type = task_id; #(is_problem_1 ? 1 : 2);
  trial_dat.correct_answer = x #(x > 0 ? 1 : -1);
  trial_dat.chosen_answer = ((abs(local_post[1]) > abs(local_post[2])) ? -1 : 1) # note sign reversal, to maintain greater than relationship
  trial_dat.got_it_right = ((local_reward > 0) ? true : false);

  # monitor average choice per block here
  #   using n independent of critic, for now
  local_choice = (abs(local_post[1]) > 0 ? 1 : 2);
  global average_choice = ( (n_within_block-1) * average_choice + local_choice ) / (n_within_block);
  global average_block_reward = ( ( n_within_block - 1) * average_block_reward + local_reward ) / (n_within_block);
  global average_task_reward;
  average_task_reward[task_id] = ( (n_task_within_block[task_id] - 1) * average_task_reward[task_id] + local_reward) / (n_task_within_block[task_id]);

  #running_av_reward(local_reward); # Nicolas is doing this before dw update, so first timestep is less unstable...
  # TODO: need to improve critic response axis and task type bin logic here
  # Binning along input-output/selection choice axis
  local_critic_response_bin = 1::Int;
  if(no_choices_per_task_critics > 1) #current logic has max 2 critics on this axis of choice
    local_correct_answer = (x > 0 ? 2 : 1);
    local_critic_response_bin = local_correct_answer;
  end
  # Binning along task type axis
  local_critic_task_bin = 1::Int;
  if(no_task_critics > 1)
    local_critic_task_bin = trial_dat.task_type;
  end
  multi_critic_running_av_reward(local_reward, local_critic_task_bin, local_critic_response_bin);

  # decide which form of average reward we're using here:
  if( use_multi_critic )
    #   multi critic:
    #     currently expanding logic to use number of critics declared in params
    local_average_reward = average_reward[local_critic_task_bin, local_critic_response_bin];
  elseif (use_single_global_critic)
    #   single global critic:
    local_sum_reward = 0.;
    local_sum_critics = 0;
    for i=1:no_task_critics
      for j = 1:no_choices_per_task_critics
        local_sum_reward += average_reward[i, j] * n_critic[i,j];
        local_sum_critics += n_critic[i,j];
      end
    end
    local_average_reward = ( local_sum_reward / local_sum_critics );
  else
    # Rmax (no running average):
    #TODO: actually this is not Rmax, that would also require (post-\bar{post}) in the dw formula
    local_average_reward = 0.;
  end

  # the weight update matrix
  dw = zeros(no_pre_neurons_per_task, no_post_neurons, no_input_tasks);
  dw[:,1,task_id] = learning_rate * local_pre[:,task_id] * local_post[1] * (local_reward - local_average_reward);
  dw[:,2,task_id] = learning_rate * local_pre[:,task_id] * local_post[2] * (local_reward - local_average_reward);
  # Save some data for later examination
  trial_dat.reward_received = (local_reward - local_average_reward);
  trial_dat.w = deepcopy(w);
  trial_dat.dw = deepcopy(dw);


  if (verbosity > 3)
    instance_average_reward[n_within_block] = local_average_reward;
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
  if(enable_weight_updates)
    global w += dw;
  end
  if (verbosity > 3) 
    #for now at least these sums are across all tasks, could make them task specific
    left_sum_w = sum(w[:,1,:]);
    right_sum_w = sum(w[:,2,:]);
    print("after weight change, sum w left: $left_sum_w, sum w right: $right_sum_w\n")
  end
  
  trial_dat.mag_dw = sum(abs(dw));
  
  # hard bound weights at +/- 10
  w[w .> weights_upper_bound] = weights_upper_bound;
  w[w .< weights_lower_bound] = weights_lower_bound;

  return (local_reward+1); # make it 0 or 2, rather than +/-1
end

