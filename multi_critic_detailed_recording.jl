######## External requirements ###########
using Distributions
using PyPlot
using Grid

########## Parameters #############

include("parameters_critic_simulations.jl");


######### Data Storage ##############

# RovingExperiment, Subject, Block and Trial defined externally
include("high_dim_array2.jl");


########## Main simulation functions #############

#include("detailed_simulation_code_herzog12.jl")
include("detailed_simulation_code_multi_critic.jl");

function reload_source()
  include("multi_critic_detailed_recording.jl");
end


########################
# 10 subjects are examined
# 14 blocks are performed per subject
# 1 block is 80 trials
# first experiment: only doing task/problem 1 learning
# second experiment: switching between task difficulties is interleaved (randomly or in alternation??)


# these are the trial inputs
function generate_test_sequence(seq_length::Int64)
  # linspace sequence:
  #x = linspace(-1,1,seq_length);

  # points are uniform randomly distributed:
  if(use_cts_random_inputs)
    x = rand(Uniform(problem_left_bound,problem_right_bound), seq_length);
  end

  # alternating +/-1 sequence
  if(use_binary_alternating_inputs)
    x = zeros(seq_length,1);
    for(i=1:seq_length)
      x[i] = -(-1)^i;
    end
  end

  # randomly alternating +/-1 sequence
  if(use_binary_random_inputs)
    x = zeros(seq_length,1);
    choice = rand(Uniform(0,1), seq_length);
    for (i = 1:seq_length)
      x[i] = (choice[i] > 0.5? -1.0 : 1.0)
    end
  end

  return x;
end


# this chooses whether a trial is of type 1 or type 2
function generate_task_sequence(seq_length::Int64)
  # alternating false/true sequence
  # x = Array(Bool, seq_length);
  # for(i = 1:seq_length)
    # x[i] = (i%2==0)
  # end

  # random arrangement of 1/2
  x = Array(Int, seq_length);
  choice = rand(Uniform(0,1), seq_length);
  for (i = 1:seq_length)
    x[i] = (choice[i] < 0.5 ? 1 : 2)
  end

  return x;
end


function perform_learning_block_single_problem(task_id::Int, tuning_type::TuningSelector, block_dat::Block)
  # generate 80 trial values for x
  # loop through x: update_noise, update_weights

  if(verbosity > 2)
    #DEBUG reward:
    # these variables are used as a back-communication channel from reward() and update_weights()
    # in order to provide debugging output at the end of this function
    global instance_correct = 0;
    global instance_incorrect = 0;

    global instance_reward = zeros(no_trials_in_block,1)
    global instance_average_reward = zeros(no_trials_in_block,1)
    global instance_choice = zeros(no_trials_in_block, 1)
    global instance_correct_direction = zeros(no_trials_in_block, 1)
    # end DEBUG
  end

  x = generate_test_sequence(no_trials_in_block);
  monitor_reward = 0;
  global average_reward;
  global n_critic;
  global average_block_reward = 0.0
  global average_task_reward;
  average_task_reward = zeros(no_input_tasks);
  if(reset_average_reward_on_each_block)
    for i = 1:no_task_critics
      for j = 1:no_choices_per_task_critics
        average_reward[i,j] = 0.;
        n_critic[i,j] = 0;
      end
    end
  end
  global average_delta_reward = 0;
  global average_choice = 0.0;
  global n_within_block = 0;
  global n_task_within_block = zeros(Int, no_input_tasks);
  local_average_threshold = 0.0;
  local_average_task_threshold = zeros(no_input_tasks);
  #for(xi in x)
  for(i = 1:no_trials_in_block)
    update_noise()
    monitor_reward += (update_weights(x[i], task_id, tuning_type, block_dat.trial[i]) / 2);
    if(perform_detection_threshold)
      local_average_threshold += block_dat.trial[i].error_threshold;
      local_average_task_threshold[task_id] += block_dat.trial[i].error_threshold;
    end
    if(verbosity > 0)
      print("\n")
    end
  end
  proportion_correct = monitor_reward / no_trials_in_block;
  if(perform_detection_threshold)
    local_average_threshold /= no_trials_in_block;
    local_average_task_threshold[task_id] = local_average_task_threshold[task_id] ./ no_trials_in_block;
  end
  #global wfinal = deepcopy(w)

  if(verbosity > 2)
    # Note this changes how the final proportion_correct is calculated!
    print("END of Learning Block, proportion correct: $proportion_correct, task_id: $task_id\n")
    proportion_correct = instance_correct / (instance_correct + instance_incorrect);
    print("DEBUG: instance_correct: $instance_correct, instance_incorrect: $instance_incorrect, new proportion correct: $proportion_correct\n")
  end

  block_dat.proportion_correct = proportion_correct;
  block_dat.proportion_task_correct[task_id] = proportion_correct;
  block_dat.average_choice = average_choice;
  
  block_dat.average_reward = average_block_reward;
  block_dat.average_task_reward = average_task_reward;

  block_dat.average_threshold = local_average_threshold;
  block_dat.average_task_threshold = local_average_task_threshold;

  return proportion_correct;
end


function perform_learning_block_trial_switching(tuning_type::TuningSelector, block_dat::Block)
  # generate 80 trial values for x
  # loop through x: update_noise, update_weights

  if(verbosity > 2)
    #DEBUG reward:
    # these variables are used as a back-communication channel from reward() and update_weights()
    # in order to provide debugging output at the end of this function
    global instance_correct = 0;
    global instance_incorrect = 0;

    global instance_reward = zeros(no_trials_in_block,1)
    global instance_average_reward = zeros(no_trials_in_block,1)
    global instance_choice = zeros(no_trials_in_block, 1)
    global instance_correct_direction = zeros(no_trials_in_block, 1)
    # end DEBUG
  end

  x = generate_test_sequence(no_trials_in_block);
  task = generate_task_sequence(no_trials_in_block);

  proportion_task_correct = zeros(no_input_tasks);
  task_count = zeros(no_input_tasks);


  monitor_reward = 0;
  global average_reward;
  global n_critic;
  if(reset_average_reward_on_each_block)
    for i = 1:no_task_critics
      for j = 1:no_choices_per_task_critics
        average_reward[i,j] = 0.;
        n_critic[i,j] = 0;
      end
    end
  end
  global average_delta_reward = 0;
  global average_choice = 0.0;
  global n_within_block = 0;
  global n_task_within_block = zeros(Int, no_input_tasks);
  global average_task_reward;
  global average_block_reward = 0.0;
  average_task_reward = zeros(no_input_tasks);
  local_average_task_choice = zeros(no_input_tasks);
  local_average_threshold = 0.0;
  local_average_task_threshold = zeros(no_input_tasks);
  for(i = 1:no_trials_in_block)
    update_noise()
    local_reward = (update_weights(x[i], task[i], tuning_type, block_dat.trial[i]) / 2);
    monitor_reward += local_reward;
    task_count[task[i]] += 1;
    proportion_task_correct[task[i]] += local_reward; # local_reward = {0,1}
    local_average_task_choice[task[i]] += block_dat.trial[i].chosen_answer;
    if(perform_detection_threshold)
      local_average_threshold += block_dat.trial[i].error_threshold;
      local_average_task_threshold[task[i]] += block_dat.trial[i].error_threshold;
    end
    if(verbosity > 0)
      print("\n")
    end
  end
  proportion_correct = monitor_reward / no_trials_in_block;
  proportion_task_correct = proportion_task_correct ./ task_count;
  local_average_task_choice = local_average_task_choice ./ task_count;
  if(perform_detection_threshold)
    local_average_threshold /= no_trials_in_block;
    local_average_task_threshold = local_average_task_threshold ./ task_count;
  end

  #global wfinal = deepcopy(w)

  if(verbosity > 2)
    # Note this changes how the final proportion_correct is calculated!
    print("END of Learning Block, proportion correct: $proportion_correct, alternating task set.\nProportion task correct: $proportion_task_correct.\n")
    print("DEBUG: task_1_count: $task_1_count, task_2_count: $task_2_count.\n")
    proportion_correct = instance_correct / (instance_correct + instance_incorrect);
    print("DEBUG: instance_correct: $instance_correct, instance_incorrect: $instance_incorrect, new proportion correct: $proportion_correct\n")
  end

  block_dat.proportion_correct = proportion_correct;
  block_dat.proportion_task_correct = proportion_task_correct;
  block_dat.average_choice = average_choice;
  block_dat.average_task_choice = local_average_task_choice;

  block_dat.average_reward = average_block_reward;
  block_dat.average_task_reward = average_task_reward;

  block_dat.average_threshold = local_average_threshold;
  block_dat.average_task_threshold = local_average_task_threshold;

  return proportion_correct;
end


function perform_single_subject_experiment(task_id::Int, tuning_type::TuningSelector, subjects_dat::Array{Subject,2}, subject_id::Int64=1)
  global enable_weight_updates :: Bool;
  global average_reward;
  global n_critic;

  global a = deepcopy(subjects_dat[subject_id, task_id].a);
  if( isa(tuning_type, linear_tc) )
    global b = deepcopy(subjects_dat[subject_id, task_id].b);
  end

  initialise_weight_matrix(tuning_type) # must be called after a and b are setup
  subjects_dat[subject_id, task_id].w_initial = deepcopy(w);

  if(disable_learning_on_first_block)
    enable_weight_updates = false :: Bool;
  end

  # these guys need to be reset here in case they're never
  #   initialised in perform_single_block...()
  #   slightly risky if not reset as they may contain values from
  #   previous runs!
  for i = 1:no_task_critics
    for j = 1:no_choices_per_task_critics
      average_reward[i,j] = 0.;
      n_critic[i,j] = 0;
    end
  end
  
  for (i = 1:no_blocks_in_experiment)
    #=if(i == no_blocks_in_experiment && subject_id == 9)
      local_old_verbosity = verbosity;
      global verbosity = 10;
    end=#
    if(verbosity > -1)
      print("------------------ Block number $i --------------------\n")
    end
    perform_learning_block_single_problem(task_id, tuning_type, subjects_dat[subject_id, task_id].blocks[i])
    if (save_reward_from_running_average)
      # Remember, average_reward is a running average not a block average
      local_average_reward = 0.;
      local_sum_critics = 0;
      for k = 1:no_task_critics
        for j = 1:no_choices_per_task_critics
          local_average_reward += average_reward[k,j] * n_critic[k,j];
          local_sum_critics += n_critic[k,j];
        end
      end
      subjects_dat[subject_id, task_id].blocks[i].average_reward = ( local_average_reward / local_sum_critics );
    end
    if(verbosity > -1)
      print("Block $i completed. Task_id: $task_id.\n") 
    end
    #=if(i == no_blocks_in_experiment && subject_id == 9)
      verbosity = local_old_verbosity;
    end=#
    enable_weight_updates = true;
  end

  subjects_dat[subject_id, task_id].w_final = deepcopy(w);
  
  return 0;
end


function perform_single_subject_experiment_trial_switching(tuning_type::TuningSelector, subjects::Array{Subject,2}, subject_id::Int64=1)
  global enable_weight_updates::Bool;
  global average_reward;
  global n_critic;

  roving_experiment_id = 1::Int;

  global a = deepcopy(subjects[subject_id, roving_experiment_id].a);
  if( isa(tuning_type, linear_tc) )
    global b = deepcopy(subjects[subject_id, roving_experiment_id].b);
  end

  initialise_weight_matrix(tuning_type) # must be called after a and b are setup
  subjects[subject_id, roving_experiment_id].w_initial = deepcopy(w);

  if(disable_learning_on_first_block)
    enable_weight_updates = false :: Bool;
  end

  # these guys need to be reset here in case they're never
  #   initialised in perform_single_block...()
  #   slightly risky if not reset as they may contain values from
  #   previous runs!
  for i = 1:no_task_critics
    for j = 1:no_choices_per_task_critics
      average_reward[i,j] = 0.;
      n_critic[i,j] = 0;
    end
  end
  
  if(double_no_of_trials_in_alternating_experiment)
    global no_trials_in_block = int(no_trials_in_block * 2);
  end

  for (i = 1:no_blocks_in_experiment)
    if(verbosity > -1)
      print("-------------------------------------------\n")
    end
    
    perform_learning_block_trial_switching(tuning_type, subjects[subject_id, roving_experiment_id].blocks[i])
    
    if (save_reward_from_running_average)
      # Remember, average_reward is a running average, not a block average.
      local_average_reward = 0.;
      local_sum_critics = 0;
      for k = 1:no_task_critics
        for j = 1:no_choices_per_task_critics
          local_average_reward += average_reward[k,j] * n_critic[k,j];
          local_sum_critics += n_critic[k,j];
        end
      end
      subjects[subject_id, roving_experiment_id].blocks[i].average_reward = ( local_average_reward / local_sum_critics );
    end

    if(verbosity > -1)
      print("Block $i completed. Alternating tasks.\n") 
    end
    enable_weight_updates = true;
  end
  
  if(double_no_of_trials_in_alternating_experiment)
    no_trials_in_block = int(no_trials_in_block / 2);
  end
  
  subjects[subject_id, roving_experiment_id].w_final = deepcopy(w);
  
  return 0;
end


function perform_multi_subject_experiment(task_id::Int, tuning_type::TuningSelector, subjects::Array{Subject,2}, no_subjects::Int64=no_subjects)
  #global subject = Array(Subject, no_subjects);

  for(i = 1:no_subjects)
    if(verbosity > -1)
      print("-----------Subject number $i------------\n")
    end
    perform_single_subject_experiment(task_id, tuning_type, subjects, i)
  end

  if(verbosity > -1)
    print("No subjects completed: $no_subjects\n")
  end
end


function perform_multi_subject_experiment_trial_switching(tuning_type::TuningSelector, subjects::Array{Subject,2}, no_subjects::Int64=no_subjects)
  #global subject = Array(Subject, no_subjects);

  for(i = 1:no_subjects)
    if(verbosity > -1)
      print("-----------Subject number $i------------\n")
    end
    perform_single_subject_experiment_trial_switching(tuning_type, subjects, i)
  end
end



############# Output #####################
function compare_three_trial_types_with_multiple_subjects()
  # figure()
  # xlim((0,no_blocks_in_experiment))
  # ylim((0,1))
  # xlabel("Block number")
  # ylabel("Proportion correct")
  # title("For x in ($problem_left_bound, $problem_right_bound), proportion correct. Comparing three task types.")

  #latest_experiment_results = Multi_subject_experiment_results(zeros(no_blocks_in_experiment), zeros(no_blocks_in_experiment), zeros(no_blocks_in_experiment), zeros(no_blocks_in_experiment), zeros(no_blocks_in_experiment));

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

  latest_experiment_results = initialise_empty_roving_experiment(tuning_type, no_subjects, no_blocks_in_experiment, no_trials_in_block);

  if(use_ab_persistence)
    for i = 1:no_subjects
      initialise_pre_population(tuning_type);
      for j = 1:no_input_tasks
        latest_experiment_results.subjects_task[i,j].a = deepcopy(a);
        if( isa(tuning_type, linear_tc) )
          latest_experiment_results.subjects_task[i,j].b = deepcopy(b);
        end
      end
      latest_experiment_results.subjects_roving_task[i,1].a = deepcopy(a);
      if( isa(tuning_type, linear_tc) )
        latest_experiment_results.subjects_roving_task[i,1].b = deepcopy(b);
      end
      initialise_pre_population(tuning_type);
      initialise_pre_population(tuning_type);
    end
  else # experiment to have identical RND sequences
    for i = 1:no_subjects
      initialise_pre_population(tuning_type); 
      latest_experiment_results.subjects_task[i,1].a = deepcopy(a);
      if( isa(tuning_type, linear_tc) )
        latest_experiment_results.subjects_task[i,1].b = deepcopy(b);
      end
      initialise_pre_population(tuning_type); 
      latest_experiment_results.subjects_task[i,2].a = deepcopy(a);
      if( isa(tuning_type, linear_tc) )
        latest_experiment_results.subjects_task[i,2].b = deepcopy(b);
      end
      initialise_pre_population(tuning_type); 
      latest_experiment_results.subjects_roving_task[i,1].a = deepcopy(a);
      if( isa(tuning_type, linear_tc) )
        latest_experiment_results.subjects_roving_task[i,1].b = deepcopy(b);
      end
    end
  end

  print("-----Experiment: task 1------\n")
  task_id = 1::Int;
  perform_multi_subject_experiment(task_id, tuning_type, latest_experiment_results.subjects_task);
  mean_correct = zeros(no_blocks_in_experiment);
  range_correct = zeros(no_blocks_in_experiment);
  err_correct = zeros(no_blocks_in_experiment);
  for i = 1:no_blocks_in_experiment
    local_prop = zeros(no_subjects);
    for j = 1:no_subjects
      local_prop[j] = latest_experiment_results.subjects_task[j,task_id].blocks[i].proportion_correct;
    end
    if(use_plot_mean)
      # mean calculation
      mean_correct[i] = mean(local_prop);
    else
      # median calculation
      mean_correct[i] = median(local_prop);
    end
    # other deviation and range statistics
    err_correct[i] = std(local_prop);
    #err_correct[i] /= sqrt(no_subjects); # standard error correction to sample standard deviation
    range_correct[i] = (maximum(local_prop) - minimum(local_prop)) / 2.0; 
  end
  # plot(mean_correct, "r", linewidth=2, label="Task 1")
  latest_experiment_results.task_correct[:,task_id] = mean_correct;
  latest_experiment_results.task_error[:,task_id] = err_correct; 
  latest_experiment_results.task_range[:,task_id] = range_correct;

  print("-----Experiment: task 2------\n")
  task_id = 2::Int;
  perform_multi_subject_experiment(task_id, tuning_type, latest_experiment_results.subjects_task);
  mean_correct = zeros(no_blocks_in_experiment);
  range_correct = zeros(no_blocks_in_experiment);
  err_correct = zeros(no_blocks_in_experiment);
  for i = 1:no_blocks_in_experiment
    local_prop = zeros(no_subjects);
    for j = 1:no_subjects
      local_prop[j] = latest_experiment_results.subjects_task[j,task_id].blocks[i].proportion_correct;
    end
    if(use_plot_mean)
      # mean calculation
      mean_correct[i] = mean(local_prop);
    else
      # median calculation
      mean_correct[i] = median(local_prop);
    end
    # other deviation and range statistics
    err_correct[i] = std(local_prop);
    #err_correct[i] /= sqrt(no_subjects); # standard error correction to sample standard deviation
    range_correct[i] = (maximum(local_prop) - minimum(local_prop)) / 2.0;
  end
  # plot(mean_correct, "g", linewidth=2, label="Task 2")
  latest_experiment_results.task_correct[:,task_id] = mean_correct;
  latest_experiment_results.task_error[:,task_id] = err_correct;
  latest_experiment_results.task_range[:,task_id] = range_correct;

  print("-----Experiment: roving task------\n")
  roving_experiment_id = 1 :: Int;
  # there's no point expanding the following to generic multiple roving pop experiments until I have such an experiment
  perform_multi_subject_experiment_trial_switching(tuning_type, latest_experiment_results.subjects_roving_task);
  mean_correct = zeros(no_blocks_in_experiment);
  mean_task_1_correct = zeros(no_blocks_in_experiment);
  mean_task_2_correct = zeros(no_blocks_in_experiment);
  err_correct = zeros(no_blocks_in_experiment);
  range_correct = zeros(no_blocks_in_experiment);
  for i = 1:no_blocks_in_experiment
    # can increase dimensionality of the following when I want to expand task space
    local_prop = zeros(no_subjects);
    local_prop_1 = zeros(no_subjects);
    local_prop_2 = zeros(no_subjects);
    for j = 1:no_subjects
      # save the proportions so that mean or median can be called
      local_prop[j] = latest_experiment_results.subjects_roving_task[j, roving_experiment_id].blocks[i].proportion_correct;
      local_prop_1[j] = latest_experiment_results.subjects_roving_task[j, roving_experiment_id].blocks[i].proportion_task_correct[1];
      local_prop_2[j] = latest_experiment_results.subjects_roving_task[j, roving_experiment_id].blocks[i].proportion_task_correct[2];
    end
    if(use_plot_mean)
      # mean calculation
      mean_correct[i] = mean(local_prop)
      mean_task_1_correct[i] = mean(local_prop_1)
      mean_task_2_correct[i] = mean(local_prop_2)
    else
      # median calculation
      mean_correct[i] = median(local_prop);
      mean_task_1_correct[i] = median(local_prop_1);
      mean_task_2_correct[i] = median(local_prop_2);
    end

    # other deviation and range statistics
    err_correct[i] = std(local_prop);
    #err_correct[i] /= sqrt(no_subjects); # standard error correction to sample standard deviation
    range_correct[i] = (maximum(local_prop) - minimum(local_prop)) / 2.0;
  end

  # plot(mean_correct, "b", linewidth=3, label="RDM alternating tasks")
  # plot(mean_task_1_correct, "k", linewidth=3, label="Task 1, from alternating tasks")
  # plot(mean_task_2_correct, "k", linewidth=3, label="Task 2, from alternating tasks")

  latest_experiment_results.roving_correct[:,roving_experiment_id] = mean_correct;
  latest_experiment_results.roving_task_correct[:,1,roving_experiment_id] = mean_task_1_correct;
  latest_experiment_results.roving_task_correct[:,2,roving_experiment_id] = mean_task_2_correct;
  latest_experiment_results.roving_error[:,roving_experiment_id] = err_correct;
  latest_experiment_results.roving_range[:,roving_experiment_id] = range_correct;

  print("Plotting...\n")
  # legend(loc=4)
  #figure()
  #plot_multi_subject_experiment(latest_experiment_results);
  #restore next line
  plot_multi_subject_experiment_as_subplots(latest_experiment_results);

  if(perform_post_hoc_detection_threshold)
    print("Calculating error detection thresholds...\n");
    post_hoc_calculate_thresholds(tuning_type, latest_experiment_results.subjects_task);
    post_hoc_calculate_thresholds(tuning_type, latest_experiment_results.subjects_roving_task);
  end

  global exp_results;
  resize!(exp_results, length(exp_results)+1);
  exp_results[length(exp_results)] = latest_experiment_results;
  print("End\n");
end


function plot_multi_subject_experiment(latest_experiment_results::RovingExperiment)
  #figure()
  xlim((0.5,no_blocks_in_experiment+0.5))
  ylim((0,1))
  xlabel("Block number")
  ylabel("Proportion correct")
  title("For x in ($problem_left_bound, $problem_right_bound), proportion correct. Comparing three task types.")

  if(plotting_scatter_plot_on)
    for i = 1:no_blocks_in_experiment
      for j = 1:no_subjects
        scatter(i, latest_experiment_results.subjects_task[j,1].blocks[i].proportion_correct, marker="o", c="r")
        scatter(i+0.1, latest_experiment_results.subjects_task[j,2].blocks[i].proportion_correct, marker="o", c="g")
        scatter(i-0.1, latest_experiment_results.subjects_roving_task[j,1].blocks[i].proportion_correct, marker="o", c="b")
      end
    end
  end

  block_id = linspace(1,no_blocks_in_experiment, no_blocks_in_experiment);

  if(plotting_error_bars_on)
    
    errorbar(block_id, latest_experiment_results.task_correct[:,1], latest_experiment_results.task_range[:,1], ecolor="r", color="r", linewidth=2)
    errorbar(block_id+0.1, latest_experiment_results.task_correct[:,2], latest_experiment_results.task_range[:,2], ecolor="g", color="g", linewidth=2)
    errorbar(block_id-0.1, latest_experiment_results.roving_correct[:,1], latest_experiment_results.roving_range[:,1], ecolor="b", color="b", linewidth=2)

    errorbar(block_id, latest_experiment_results.task_correct[:,1], latest_experiment_results.task_error[:,1], ecolor="k", color="r", linewidth=2)
    errorbar(block_id+0.1, latest_experiment_results.task_correct[:,2], latest_experiment_results.task_error[:,2], ecolor="k", color="g", linewidth=2)
    errorbar(block_id-0.1, latest_experiment_results.roving_correct[:,1], latest_experiment_results.roving_error[:,1], ecolor="k", color="b", linewidth=2)
  end

  if(plotting_individual_subjects_on)
    for j = 1:no_subjects
      local_prop_1_correct = zeros(no_blocks_in_experiment);
      local_prop_2_correct = zeros(no_blocks_in_experiment);
      local_prop_roving_correct = zeros(no_blocks_in_experiment);
      for i = 1:no_blocks_in_experiment
        local_prop_1_correct[i] = latest_experiment_results.subjects_task[j,1].blocks[i].proportion_correct;
        local_prop_2_correct[i] = latest_experiment_results.subjects_task[j,2].blocks[i].proportion_correct;
        local_prop_roving_correct[i] = latest_experiment_results.subjects_roving_task[j,1].blocks[i].proportion_correct;
      end
      plot(block_id, local_prop_1_correct, "r")
      plot(block_id+0.1, local_prop_2_correct, "g")
      plot(block_id-0.1, local_prop_roving_correct, "b")
    end
  end

  plot(block_id, latest_experiment_results.task_correct[:,1], "r", linewidth=2, label="Task 1")
  plot(block_id+0.1, latest_experiment_results.task_correct[:,2], "g", linewidth=2, label="Task 2")
  plot(block_id-0.1, latest_experiment_results.roving_correct[:,1], "b", linewidth=3, label="Roving tasks")
  plot(block_id-0.1, latest_experiment_results.roving_task_correct[:,1,1], "k", linewidth=3, label="Task 1, from roving tasks")
  plot(block_id-0.1, latest_experiment_results.roving_task_correct[:,2,1], "k", linewidth=3, label="Task 2, from roving tasks")
  
  legend(loc=4)
end


function plot_multi_subject_experiment_as_subplots(latest_experiment_results::RovingExperiment)
  figure(figsize=(12,12))
    if (use_multi_critic)
    suptitle("For x in ($problem_left_bound, $problem_right_bound), proportion correct. Comparing three task types. Multicritic: $use_multi_critic, no_task_critics: $no_task_critics, no_choices_per_task_critics: $no_choices_per_task_critics ")
  elseif (use_single_global_critic)
    suptitle("For x in ($problem_left_bound, $problem_right_bound), proportion correct. Comparing three task types. Single critic")
  else
    suptitle("For x in ($problem_left_bound, $problem_right_bound), proportion correct. Comparing three task types. No critic")
  end
  subplot(221);
  xlim((0,no_blocks_in_experiment))
  ylim((0,1))
  xlabel("Block number")
  ylabel("Proportion correct")

  ## Plot all in one pane
  plot_multi_subject_experiment(latest_experiment_results);
  title("");

  block_id = linspace(1,no_blocks_in_experiment, no_blocks_in_experiment);


  ## Task 1 subplot
  subplot(222)
  xlim((0,no_blocks_in_experiment))
  ylim((0,1))
  xlabel("Block number")
  ylabel("Proportion correct")

  if(plotting_scatter_plot_on)
    for i = 1:no_blocks_in_experiment
      for j = 1:no_subjects
        scatter(i, latest_experiment_results.subjects_task[j,1].blocks[i].proportion_correct, marker="o", c="r")
      end
    end
  end

  if(plotting_error_bars_on)
    errorbar(block_id, latest_experiment_results.task_correct[:,1], latest_experiment_results.task_range[:,1], ecolor="r", color="r", linewidth=2)
    errorbar(block_id, latest_experiment_results.task_correct[:,1], latest_experiment_results.task_error[:,1], ecolor="k", color="r", linewidth=2)
  end

  if(plotting_individual_subjects_on)
    for j = 1:no_subjects
      local_prop_1_correct = zeros(no_blocks_in_experiment);
      for i = 1:no_blocks_in_experiment
        local_prop_1_correct[i] = latest_experiment_results.subjects_task[j,1].blocks[i].proportion_correct;
      end
      plot(block_id, local_prop_1_correct, "r")
    end
  end

  plot(block_id, latest_experiment_results.task_correct[:,1], "r", linewidth=2, label="Task 1")
  legend(loc=4)


  ## Task 2 subplot
  subplot(223)
  xlim((0,no_blocks_in_experiment))
  ylim((0,1))
  xlabel("Block number")
  ylabel("Proportion correct")

  if(plotting_scatter_plot_on)
    for i = 1:no_blocks_in_experiment
      for j = 1:no_subjects
        scatter(i+0.1, latest_experiment_results.subjects_task[j,2].blocks[i].proportion_correct, marker="o", c="g")
      end
    end
  end

  if(plotting_error_bars_on)
    errorbar(block_id+0.1, latest_experiment_results.task_correct[:,2], latest_experiment_results.task_range[:,2], ecolor="g", color="g", linewidth=2)
    errorbar(block_id+0.1, latest_experiment_results.task_correct[:,2], latest_experiment_results.task_error[:,2], ecolor="k", color="g", linewidth=2)
  end

  if(plotting_individual_subjects_on)
    for j = 1:no_subjects
      local_prop_2_correct = zeros(no_blocks_in_experiment);
      for i = 1:no_blocks_in_experiment
        local_prop_2_correct[i] = latest_experiment_results.subjects_task[j,2].blocks[i].proportion_correct;
      end
      plot(block_id, local_prop_2_correct, "g")
    end
  end

  plot(block_id+0.1, latest_experiment_results.task_correct[:,2], "g", linewidth=2, label="Task 2")
  legend(loc=4)


  ## Roving subplot
  subplot(224)
  xlim((0,no_blocks_in_experiment))
  ylim((0,1))
  xlabel("Block number")
  ylabel("Proportion correct")

  if(plotting_scatter_plot_on)
    for i = 1:no_blocks_in_experiment
      for j = 1:no_subjects
        scatter(i-0.1, latest_experiment_results.subjects_roving_task[j,1].blocks[i].proportion_correct, marker="o", c="b")
      end
    end
  end

  if(plotting_error_bars_on)
    errorbar(block_id-0.1, latest_experiment_results.roving_correct[:,1], latest_experiment_results.roving_range[:,1], ecolor="b", color="b", linewidth=2)
    errorbar(block_id-0.1, latest_experiment_results.roving_correct[:,1], latest_experiment_results.roving_error[:,1], ecolor="k", color="b", linewidth=2)
  end

  if(plotting_individual_subjects_on)
    for j = 1:no_subjects
      local_prop_roving_correct = zeros(no_blocks_in_experiment);
      for i = 1:no_blocks_in_experiment
        local_prop_roving_correct[i] = latest_experiment_results.subjects_roving_task[j,1].blocks[i].proportion_correct;
      end
      plot(block_id-0.1, local_prop_roving_correct, "b")
    end
  end

  plot(block_id-0.1, latest_experiment_results.roving_correct[:,1], "b", linewidth=3, label="Roving tasks")
  plot(block_id-0.1, latest_experiment_results.roving_task_correct[:,1,1], "k", linewidth=3, label="Task 1, from roving tasks")
  plot(block_id-0.1, latest_experiment_results.roving_task_correct[:,2,1], "k", linewidth=3, label="Task 2, from roving tasks")
  
  
  legend(loc=4)
end


function plot_multi_subject_experiment_reward_as_subplots(latest_experiment_results::RovingExperiment)
  figure(figsize=(12,8))
  if (use_multi_critic)
    suptitle("For x in ($problem_left_bound, $problem_right_bound). Comparing three task types. Multicritic: $use_multi_critic, no_task_critics: $no_task_critics, no_choices_per_task_critics: $no_choices_per_task_critics ")
  elseif (use_single_global_critic)
    suptitle("For x in ($problem_left_bound, $problem_right_bound). Comparing three task types. Single critic")
  else
    suptitle("For x in ($problem_left_bound, $problem_right_bound). Comparing three task types. No critic")
  end
  subplot(221);
  xlim((0,no_blocks_in_experiment))
  ylim((-1,1))
  xlabel("Block number")
  ylabel("Average reward")

  ## Plot all in one pane
  #plot_multi_subject_experiment(latest_experiment_results);
  title("");

  block_id = linspace(1,no_blocks_in_experiment, no_blocks_in_experiment);


  ## Task 1 subplot
  subplot(311)
  xlim((0,no_blocks_in_experiment))
  ylim((-1,1))
  xlabel("Block number")
  ylabel("Average reward")

  if(plotting_scatter_plot_on)
    for i = 1:no_blocks_in_experiment
      for j = 1:no_subjects
        scatter(i, latest_experiment_results.subjects_task[j,1].blocks[i].average_reward, marker="o", c="r")
      end
    end
  end

  if(plotting_individual_subjects_on)
    for j = 1:no_subjects
      local_1_reward = zeros(no_blocks_in_experiment);
      for i = 1:no_blocks_in_experiment
        local_1_reward[i] = latest_experiment_results.subjects_task[j,1].blocks[i].average_reward;
      end
      plot(block_id, local_1_reward, "r")
    end
  end

  #legend(loc=4)


  ## Task 2 subplot
  subplot(312)
  xlim((0,no_blocks_in_experiment))
  ylim((-1,1))
  xlabel("Block number")
  ylabel("Average reward")

  if(plotting_scatter_plot_on)
    for i = 1:no_blocks_in_experiment
      for j = 1:no_subjects
        scatter(i+0.1, latest_experiment_results.subjects_task[j,2].blocks[i].average_reward, marker="o", c="g")
      end
    end
  end

  if(plotting_individual_subjects_on)
    for j = 1:no_subjects
      local_1_reward = zeros(no_blocks_in_experiment);
      for i = 1:no_blocks_in_experiment
        local_1_reward[i] = latest_experiment_results.subjects_task[j,2].blocks[i].average_reward;
      end
      plot(block_id, local_1_reward, "g")
    end
  end

  #legend(loc=4)


  ## Roving subplot
  subplot(313)
  xlim((0,no_blocks_in_experiment))
  ylim((-1,1))
  xlabel("Block number")
  ylabel("Average reward")

  if(plotting_scatter_plot_on)
    for i = 1:no_blocks_in_experiment
      for j = 1:no_subjects
        scatter(i-0.1, latest_experiment_results.subjects_roving_task[j,1].blocks[i].average_reward, marker="o", c="b")
      end
    end
  end

  if(plotting_individual_subjects_on)
    for j = 1:no_subjects
      local_roving_reward = zeros(no_blocks_in_experiment);
      for i = 1:no_blocks_in_experiment
        local_roving_reward[i] = latest_experiment_results.subjects_roving_task[j,1].blocks[i].average_reward;
      end
      plot(block_id-0.1, local_roving_reward, "b")
    end
  end

  #legend(loc=4)
end


function plot_multi_subject_experiment_choice_as_subplots(latest_experiment_results::RovingExperiment)
  figure(figsize=(12,8))
    if (use_multi_critic)
    suptitle("For x in ($problem_left_bound, $problem_right_bound). Comparing three task types. Multicritic: $use_multi_critic, no_task_critics: $no_task_critics, no_choices_per_task_critics: $no_choices_per_task_critics ")
  elseif (use_single_global_critic)
    suptitle("For x in ($problem_left_bound, $problem_right_bound). Comparing three task types. Single critic")
  else
    suptitle("For x in ($problem_left_bound, $problem_right_bound). Comparing three task types. No critic")
  end
#=  subplot(221);
  xlim((0,no_blocks_in_experiment))
  ylim((-1,1))
  xlabel("Block number")
  ylabel("Average reward")=#

  ## Plot all in one pane
  #plot_multi_subject_experiment(latest_experiment_results);
  title("");

  block_id = linspace(1,no_blocks_in_experiment, no_blocks_in_experiment);


  ## Task 1 subplot
  subplot(311)
  xlim((0,no_blocks_in_experiment))
  ylim((1,2))
  xlabel("Block number")
  ylabel("Average choice")

  if(plotting_scatter_plot_on)
    for i = 1:no_blocks_in_experiment
      for j = 1:no_subjects
        scatter(i, latest_experiment_results.subjects_task[j,1].blocks[i].average_choice, marker="o", c="r")
      end
    end
  end

  if(plotting_individual_subjects_on)
    for j = 1:no_subjects
      local_1_choice = zeros(no_blocks_in_experiment);
      for i = 1:no_blocks_in_experiment
        local_1_choice[i] = latest_experiment_results.subjects_task[j,1].blocks[i].average_choice;
      end
      plot(block_id, local_1_choice, "r")
    end
  end

  #legend(loc=4)


  ## Task 2 subplot
  subplot(312)
  xlim((0,no_blocks_in_experiment))
  ylim((1,2))
  xlabel("Block number")
  ylabel("Average choice")

  if(plotting_scatter_plot_on)
    for i = 1:no_blocks_in_experiment
      for j = 1:no_subjects
        scatter(i+0.1, latest_experiment_results.subjects_task[j,2].blocks[i].average_choice, marker="o", c="g")
      end
    end
  end

  if(plotting_individual_subjects_on)
    for j = 1:no_subjects
      local_1_choice = zeros(no_blocks_in_experiment);
      for i = 1:no_blocks_in_experiment
        local_1_choice[i] = latest_experiment_results.subjects_task[j,2].blocks[i].average_choice;
      end
      plot(block_id, local_1_choice, "g")
    end
  end

  #legend(loc=4)


  ## Roving subplot
  subplot(313)
  xlim((0,no_blocks_in_experiment))
  ylim((1,2))
  xlabel("Block number")
  ylabel("Average choice")

  if(plotting_scatter_plot_on)
    for i = 1:no_blocks_in_experiment
      for j = 1:no_subjects
        scatter(i-0.1, latest_experiment_results.subjects_roving_task[j,1].blocks[i].average_choice, marker="o", c="b")
      end
    end
  end

  if(plotting_individual_subjects_on)
    for j = 1:no_subjects
      local_roving_choice = zeros(no_blocks_in_experiment);
      for i = 1:no_blocks_in_experiment
        local_roving_choice[i] = latest_experiment_results.subjects_roving_task[j,1].blocks[i].average_choice;
      end
      plot(block_id-0.1, local_roving_choice, "b")
    end
  end

  #legend(loc=4)
end


function plot_multiplot(results::Array{RovingExperiment})
  if (length(results)>0)
    figure(figsize=(12,12))
    title("For x in ($problem_left_bound, $problem_right_bound), proportion correct. Comparing three task types.")

    i = 1::Int;
    plot_base = 22;

    for latest_experiment_results in results
      subplot(string(plot_base, i));
      xlim((0,no_blocks_in_experiment))
      ylim((0,1))
      xlabel("Block number")
      ylabel("Proportion correct")
      plot(latest_experiment_results.task_correct[:,1], "r", linewidth=2, label="Task 1")
      plot(latest_experiment_results.task_correct[:,2], "g", linewidth=2, label="Task 2")
      plot(latest_experiment_results.roving_correct[:,1], "b", linewidth=3, label="Roving tasks")
      plot(latest_experiment_results.roving_task_correct[:,1,1], "k", linewidth=3, label="Task 1, from roving tasks")
      plot(latest_experiment_results.roving_task_correct[:,2,1], "k", linewidth=3, label="Task 2, from roving tasks")
      i+=1;
    end
    
    legend(loc=4)
    suptitle("Multiplot")
  else
    print("Sorry, your results array was empty!\n");
  end
end


function print_single_block_performance(block::Block)
  print("Task|Chosen|Correct|Right|Success Signal|ErrorThreshold\n")
  local_av_task = 0;
  local_av_choice = 0.;
  local_av_x = 0.;
  local_av_correct = 0;
  local_av_reward = 0.;
  local_av_err_threshold = 0.;
  local_av_n = length(block.trial);
  for i = 1:local_av_n
    print("",block.trial[i].task_type," | ")
    print("",block.trial[i].chosen_answer," | ")
    print("",block.trial[i].correct_answer," | ")
    print("",block.trial[i].got_it_right," | ")
    print("",block.trial[i].reward_received," | ")
    print("",block.trial[i].error_threshold," \n");
    local_av_task += block.trial[i].task_type;
    local_av_choice += block.trial[i].chosen_answer;
    local_av_x += block.trial[i].correct_answer;
    (block.trial[i].got_it_right ? local_av_correct += 1 : 0 )
    local_av_reward += block.trial[i].reward_received
    local_av_err_threshold += block.trial[i].error_threshold;
  end
  local_av_task /= local_av_n;
  local_av_choice /= local_av_n;
  local_av_x /= local_av_n;
  local_av_correct /= local_av_n;
  local_av_reward /= local_av_n;
  local_av_err_threshold /= local_av_n;
  print("--------------------------\n")
  print("",local_av_task," | ")
  print("",local_av_choice," | ")
  print("",local_av_x," | ")
  print("",local_av_correct," | ")
  print("",local_av_reward," | ")
  print("",local_av_err_threshold," \n ")
end

# I guess that the following should become my plot of threshold not reward received...
function plot_single_block_threshold_performance(block::Block)
  #figure()
  no_trials_in_block = length(block.trial); # may not be global value due to double length roving sims
  local_error_threshold = zeros(no_trials_in_block);
  x = linspace(1, no_trials_in_block, no_trials_in_block);
  for i = 1:no_trials_in_block
    local_error_threshold[i] = block.trial[i].error_threshold;
    #print("", x[i], " ", local_reward_received[i], "\n")
  end
  #print("", size(local_reward_received), " ", size(x),"\n")
  plot(x, local_error_threshold, linewidth=2)
  return no_trials_in_block;
end

function plot_multi_block_threshold_performance(subject::Subject, begin_id::Int=1, end_id::Int=no_blocks_in_experiment)
  figure()
  max_no_trials_in_block = 0::Int;
  for i = begin_id:end_id
    no_trials = plot_single_block_threshold_performance(subject.blocks[i]);
    if (no_trials > max_no_trials_in_block)
      max_no_trials_in_block = no_trials;
    end
  end
  xlabel("Trial number")
  ylabel("Error threshold")
  axis([0,max_no_trials_in_block,0,1])
end


function plot_single_block_reward_received(block::Block)
  #figure()
  no_trials_in_block = length(block.trial); # may not be global value due to double length roving sims
  local_reward_received = zeros(no_trials_in_block);
  x = linspace(1, no_trials_in_block, no_trials_in_block);
  for i = 1:no_trials_in_block
    local_reward_received[i] = block.trial[i].reward_received;
    #print("", x[i], " ", local_reward_received[i], "\n")
  end
  #print("", size(local_reward_received), " ", size(x),"\n")
  plot(x, local_reward_received, linewidth=2)
  return no_trials_in_block;
end

function plot_multi_block_reward_recived(subject::Subject, begin_id::Int=1, end_id::Int=no_blocks_in_experiment)
  figure()
  max_no_trials_in_block = 0::Int;
  for i = begin_id:end_id
    no_trials = plot_single_block_reward_received(subject.blocks[i])
    if (no_trials > max_no_trials_in_block)
      max_no_trials_in_block = no_trials;
    end
  end
  xlabel("Trial number")
  ylabel("Reward received")
  axis([0,max_no_trials_in_block,-2,2])
end


function plot_single_block_mag_dw(block::Block)
  #figure()
  no_trials_in_block = length(block.trial); # may not be global value due to double length roving sims
  local_mag_dw = zeros(no_trials_in_block);
  x = linspace(1, no_trials_in_block, no_trials_in_block);
  for i = 1:no_trials_in_block
    local_mag_dw[i] = block.trial[i].mag_dw;
    #print("", x[i], " ", local_reward_received[i], "\n")
  end
  #print("", size(local_reward_received), " ", size(x),"\n")
  plot(x, local_mag_dw, linewidth=2)
  return no_trials_in_block;
end

function plot_multi_block_mag_dw(subject::Subject, begin_id::Int=1, end_id::Int=no_blocks_in_experiment)
  figure()
  max_no_trials_in_block = 0::Int;
  for i = begin_id:end_id
    no_trials = plot_single_block_mag_dw(subject.blocks[i]);
    if (no_trials > max_no_trials_in_block)
      max_no_trials_in_block = no_trials;
    end
  end
  xlabel("Trial number")
  ylabel("Magnitude dw")
  #axis([0,max_no_trials_in_block,-2,2])
end


function plot_single_subject_proportion_correct(subject::Subject)
  #figure()
  local_av_reward = zeros(no_blocks_in_experiment);
  local_av_task_reward = zeros(no_blocks_in_experiment, no_input_tasks);
  x = linspace(1, no_blocks_in_experiment, no_blocks_in_experiment);
  for i = 1:no_blocks_in_experiment
    local_av_reward[i] = subject.blocks[i].proportion_correct; #average_reward;
    local_av_task_reward[i,:] = subject.blocks[i].proportion_task_correct; 
    #print("", x[i], " ", local_reward_received[i], "\n")
  end
  #print("", size(local_reward_received), " ", size(x),"\n")
  plot(x, local_av_task_reward[:,1], linewidth=2, c="k")
  plot(x, local_av_task_reward[:,2], linewidth=2, c="g")
  plot(x, local_av_reward, linewidth=2, c="r")
end

function plot_multi_subject_proportion_correct(subjects::Array{Subject,2}, task_id::Int=1, begin_id::Int=1, end_id::Int=no_subjects)
  figure()
  for i = begin_id:end_id
    plot_single_subject_proportion_correct(subjects[i,task_id])
  end
  xlabel("Block number")
  ylabel("Proportion correct")
  axis([0,no_blocks_in_experiment,0,1])
end


function plot_single_subject_average_threshold(subject::Subject)
  #figure()
  local_av_threshold = zeros(no_blocks_in_experiment);
  local_av_task_threshold = zeros(no_blocks_in_experiment, no_input_tasks);
  x = linspace(1, no_blocks_in_experiment, no_blocks_in_experiment);
  for i = 1:no_blocks_in_experiment
    local_av_threshold[i] = subject.blocks[i].average_threshold; #average_reward;
    local_av_task_threshold[i,:] = subject.blocks[i].average_task_threshold; 
    #print("", x[i], " ", local_reward_received[i], "\n")
  end
  #print("", size(local_reward_received), " ", size(x),"\n")
  plot(x, local_av_task_threshold[:,1], linewidth=2, c="r")
  plot(x, local_av_task_threshold[:,2], linewidth=2, c="g")
  plot(x, local_av_threshold, linewidth=2, c="k")
end

function plot_multi_subject_average_threshold(subjects::Array{Subject,2}, task_id::Int=1, begin_id::Int=1, end_id::Int=no_subjects)
  figure()
  for i = begin_id:end_id
    plot_single_subject_average_threshold(subjects[i,task_id])
  end
  xlabel("Block number")
  ylabel("Average error threshold (x | error = 0.25)")
  axis([0,no_blocks_in_experiment,0,1])
end


function plot_single_subject_average_reward(subject::Subject)
  #figure()
  local_av_reward = zeros(no_blocks_in_experiment);
  local_av_task_reward = zeros(no_blocks_in_experiment, no_input_tasks);
  x = linspace(1, no_blocks_in_experiment, no_blocks_in_experiment);
  for i = 1:no_blocks_in_experiment
    local_av_reward[i] = subject.blocks[i].average_reward;
    local_av_task_reward[i,:] = subject.blocks[i].average_task_reward; 
    #print("", x[i], " ", local_reward_received[i], "\n")
  end
  #print("", size(local_reward_received), " ", size(x),"\n")
  plot(x, local_av_task_reward[:,1], linewidth=2, c="r")
  plot(x, local_av_task_reward[:,2], linewidth=2, c="g")
  plot(x, local_av_reward, linewidth=2, c="k")
end

function plot_multi_subject_average_reward(subjects::Array{Subject,2}, task_id::Int=1, begin_id::Int=1, end_id::Int=no_subjects)
  figure()
  for i = begin_id:end_id
    plot_single_subject_average_reward(subjects[i,task_id])
  end
  xlabel("Block number")
  ylabel("Average reward")
  axis([0,no_blocks_in_experiment,-1,1])
end


function plot_single_subject_average_choice(subject::Subject)
  #figure()
  local_av_choice = zeros(no_blocks_in_experiment);
  local_av_task_choice= zeros(no_blocks_in_experiment, no_input_tasks);
  x = linspace(1, no_blocks_in_experiment, no_blocks_in_experiment);
  for i = 1:no_blocks_in_experiment
    local_av_choice[i] = (subject.blocks[i].average_choice - 1.5) * 2;
    local_av_task_choice[i,:] = (subject.blocks[i].average_task_choice );
    #print("", x[i], " ", local_reward_received[i], "\n")
  end
  #print("", size(local_reward_received), " ", size(x),"\n")
  plot(x, local_av_task_choice[:,1], linewidth=2, c="r")
  plot(x, local_av_task_choice[:,2], linewidth=2, c="g")
  plot(x, local_av_choice, linewidth=2, c="k")
end

function plot_multi_subject_average_choice(subjects::Array{Subject,2}, task_id::Int=1, begin_id::Int=1, end_id::Int=no_subjects)
  figure()
  for i = begin_id:end_id
    plot_single_subject_average_choice(subjects[i,task_id])
  end
  xlabel("Block number")
  ylabel("Average choice")
  axis([0,no_blocks_in_experiment,-1,1])
end


function plot_single_subject_final_weight_vs_bias(subject::Array{Subject,2}, task_ids::Array{Int,1}=[1])
  # subject array should contain subject i with each of his task instances
  #figure()
  for task_id in task_ids
    if (task_id == 1)
      scatter(subject[1,task_id].b[:,task_id], subject[1,task_id].w_final[:,1,task_id], marker="o", c="g", label="left, easy")
      scatter(subject[1,task_id].b[:,task_id], subject[1,task_id].w_final[:,2,task_id], marker="o", c="y", label="right, easy")
    elseif (task_id == 2)
      scatter(subject[1,task_id].b[:,task_id], subject[1,task_id].w_final[:,1,task_id], marker="o", c="r", label="left, hard")
      scatter(subject[1,task_id].b[:,task_id], subject[1,task_id].w_final[:,2,task_id], marker="o", c="k", label="right, hard")
    else
      scatter(subject[1,task_id].b[:,task_id], subject[1,task_id].w_final[:,1,task_id], marker="o", c="c", label="left, other")
      scatter(subject[1,task_id].b[:,task_id], subject[1,task_id].w_final[:,2,task_id], marker="o", c="m", label="right, other")
    end
  end
  xlim([-1.2,1.2])
  ylim([-12,12])
  #legend()
end

function plot_multi_subject_final_weight_vs_bias(subjects::Array{Subject,2}, task_ids::Array{Int,1}=[1], begin_id::Int=1, end_id::Int=no_subjects)
  figure(figsize=(6,20))
  for i = begin_id:end_id
    subplot(10,1,i)
    plot_single_subject_final_weight_vs_bias(subjects[i,:],task_ids);
  end
  legend()
end


function plot_single_subject_initial_weight_vs_bias(subject::Array{Subject,2}, task_ids::Array{Int,1}=[1])
  #figure()
  for task_id in task_ids
    if (task_id == 1)
      scatter(subject[1,task_id].b[:,task_id], subject[1,task_id].w_initial[:,1,task_id], marker="o", c="g", label="left, easy")
      scatter(subject[1,task_id].b[:,task_id], subject[1,task_id].w_initial[:,2,task_id], marker="o", c="y", label="right, easy")
    elseif (task_id == 2)
      scatter(subject[1,task_id].b[:,task_id], subject[1,task_id].w_initial[:,1,task_id], marker="o", c="r", label="left, hard")
      scatter(subject[1,task_id].b[:,task_id], subject[1,task_id].w_initial[:,2,task_id], marker="o", c="k", label="right, hard")
    else
      scatter(subject[1,task_id].b[:,task_id], subject[1,task_id].w_initial[:,1,task_id], marker="o", c="c", label="left, other")
      scatter(subject[1,task_id].b[:,task_id], subject[1,task_id].w_initial[:,2,task_id], marker="o", c="m", label="right, other")
    end
  end
  xlim([-1.2,1.2])
  ylim([-12,12])
  #legend()
end

function plot_multi_subject_initial_weight_vs_bias(subjects::Array{Subject,2}, task_ids::Array{Int,1}=[1], begin_id::Int=1, end_id::Int=no_subjects)
  figure(figsize=(6,20))
  for i = begin_id:end_id
    subplot(10,1,i)
    plot_single_subject_initial_weight_vs_bias(subjects[i,:], task_ids);
  end
  legend()
end


function plot_subjects_initial_weight_distributions(subjects::Array{Subject,2}, task_id::Int=1)
  (no_subjects, no_tasks) = size(subjects);

  inter_subject_gap = 0.1;
  lr_gap = (no_subjects+2) * inter_subject_gap;
  figure()
  x1 = ones(no_pre_neurons_per_task);
  x2 = ones(no_pre_neurons_per_task) * lr_gap;
  
  for i = 1:no_subjects
    restore_subject(subjects[i,task_id]);
    #=scatter(x1+( (i-1) * inter_subject_gap), w[:,1,1], c="b")
    scatter(x2+( (i-1) * inter_subject_gap), w[:,2,1], c="g")=#
    scatter( (i * x1) , w[:,1,1], c="b")
    scatter( (i * x1) + 0.5, w[:,2,1], c="g")
  end
end


function who_doesnt_learn(subjects::Array{Subject,2}, task_id::Int=1, threshold::Float64=10.0, begin_id::Int=1, end_id::Int=no_subjects)
  print("Detection threshold: $threshold\n")
  for i = begin_id:end_id
    if ( subjects[i,task_id].blocks[end].proportion_correct < threshold )
      print("Subject $i proportion correct:", subjects[i,task_id].blocks[end].proportion_correct,"\n")
    end
  end
end


function will_subject_learn(subjects::Array{Subject,2}, task_id::Int=1, begin_id::Int=1, end_id::Int=no_subjects)
  heuristic_threshold = 1e-3;
  weight_sum_threshold = 11.0;

  if(use_gaussian_tuning_function)
    tuning_type = gaussian_tc();
  elseif(use_linear_tuning_function)
    tuning_type = linear_tc();
  else
    print("Undefined tuning function type!\n")
    error(1);
  end

  print("Heuristic for who will learn based on inital weights and tuning curves, error heuristic threshold $heuristic_threshold, weight sum threshold $weight_sum_threshold:\n") 
  for i = begin_id:end_id
    global a = deepcopy(subjects[i,task_id].a);
    if( isa(tuning_type, linear_tc) )
      global b = deepcopy(subjects[i,task_id].b);
    end
    global w = deepcopy(subjects[i,task_id].w_initial);

    pre_pos_1 = pre(1.0, task_id, tuning_type);
    pre_neg_1 = pre(-1.0, task_id, tuning_type);

    # calculate noise free post for +1
    noise_free_post_pos_left = sum(pre_pos_1[:,task_id].*w[:,1,task_id]);
    noise_free_post_pos_right = sum(pre_pos_1[:,task_id].*w[:,2,task_id]);

    # calculate noise free post for -1
    noise_free_post_neg_left = sum(pre_neg_1[:,task_id].*w[:,1,task_id]);
    noise_free_post_neg_right = sum(pre_neg_1[:,task_id].*w[:,2,task_id]);

    p_pos_left = 0.5 + 0.5 * erf( (noise_free_post_pos_left - noise_free_post_pos_right) / (output_noise / 2.0) );
    p_neg_left = 0.5 + 0.5 * erf( (noise_free_post_neg_left - noise_free_post_neg_right) / (output_noise / 2.0) );
    p_neg_right = (1. - p_neg_left);

    if (verbosity > 1)
      print("Subject $i, p_pos_left: $p_pos_left, p_neg_right: $p_neg_right, sum wts left: ", sum(w[:,1,task_id]), ", sum wts right: ", sum(w[:,2,task_id]),"\n")
    end

    if ( ( ( abs(p_pos_left - 0) < heuristic_threshold ) && ( abs(p_neg_right - 1) < heuristic_threshold ) ) || ( abs( sum(w[:,1,task_id]) - sum(w[:,2,task_id]) ) > weight_sum_threshold ) )
      print("Subject $i: Probably not going to learn, biased left for task $task_id. p_pos_left: $p_pos_left, p_neg_right: $p_neg_right, sum wts left: ", sum(w[:,1,task_id]), ", sum wts right: ", sum(w[:,2,task_id]),"\n")
    elseif ( ( abs(p_pos_left - 1) < heuristic_threshold ) && ( abs(p_neg_right - 0) < heuristic_threshold ) )
      print("Subject $i: Probably not going to learn, biased right for task $task_id. p_pos_left: $p_pos_left, p_neg_right: $p_neg_right, sum wts left: ", sum(w[:,1,task_id]), ", sum wts right: ", sum(w[:,2,task_id]),"\n")
    else
      print("Subject $i, no bias, can learn for task $task_id. sum wts left: ", sum(w[:,1,task_id]), ", sum wts right: ", sum(w[:,2,task_id]),"\n")
    end
  end
  print("\nWarning: function altered global variables a, b and w\n")
end


function restore_subject(subject::Subject, initial_weights::Bool=true)
  if(use_gaussian_tuning_function)
    tuning_type = gaussian_tc();
  elseif(use_linear_tuning_function)
    tuning_type = linear_tc();
  else
    print("Undefined tuning function type!\n")
    error(1);
  end

  global a = deepcopy(subject.a);
  if( isa(tuning_type, linear_tc) )
    global b = deepcopy(subject.b);
  end
  if (initial_weights)
    global w = deepcopy(subject.w_initial);
  else
    global w = deepcopy(subject.w_final);
  end
  print("Subject restored\n")
end

#####################################
print("------------NEW RUN--------------\n")
#perform_multi_subject_experiment(true)
#perform_single_subject_experiment(true)
#perform_learning_block_single_problem(false)


#plot_multi_subject_results(10, 14)
#plot_multi_subject_rewards(10, 14)
#plot_multi_subject_reward_deltas(10, 14)




