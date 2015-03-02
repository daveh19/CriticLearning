######## External requirements ###########
using Distributions
using PyPlot
using Grid

########## Parameters #############

include("parameters_critic_simulations.jl")


######### Data Storage ##############

# RovingExperiment, Subject, Block and Trial defined externally
include("high_dim_array2.jl");


########## Main simulation functions #############

#include("detailed_simulation_code_herzog12.jl")
include("detailed_simulation_code_multi_critic.jl")

function reload_source()
  include("multi_critic_detailed_recording.jl")
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

  # random arrangement of true/false
  x = Array(Bool, seq_length);
  choice = rand(Uniform(0,1), seq_length);
  for (i = 1:seq_length)
    x[i] = (choice[i] < 0.5 ? true : false)
  end

  return x;
end


function perform_learning_block_single_problem(is_problem_1::Bool, block_dat::Block)
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
  if(reset_average_reward_on_each_block)
    for i = 1: no_task_critics
      for j = 1:no_choices_per_task_critics
        average_reward[i,j] = 0.;
        n_critic[i,j] = 0;
      end
    end
  end
  global average_delta_reward = 0;
  global average_choice = 0.0;
  global n = 0;
  #for(xi in x)
  for(i = 1:no_trials_in_block)
    update_noise()
    monitor_reward += (update_weights(x[i], is_problem_1, block_dat.trial[i]) / 2);
    if(verbosity > 0)
      print("\n")
    end
  end
  proportion_correct = monitor_reward / no_trials_in_block;

  #global wfinal = deepcopy(w)

  if(verbosity > 2)
    # Note this changes how the final proportion_correct is calculated!
    print("END of Learning Block, proportion correct: $proportion_correct, is problem 1: $is_problem_1\n")
    proportion_correct = instance_correct / (instance_correct + instance_incorrect);
    print("DEBUG: instance_correct: $instance_correct, instance_incorrect: $instance_incorrect, new proportion correct: $proportion_correct\n")
  end

  return proportion_correct;
end


function perform_learning_block_trial_switching(block_dat::Block)
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

  global proportion_1_correct = 0;
  global proportion_2_correct = 0;
  task_1_count = 0;
  task_2_count = 0;

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
  global n = 0;
  for(i = 1:no_trials_in_block)
    update_noise()
    local_reward = (update_weights(x[i], task[i], block_dat.trial[i]) / 2);
    monitor_reward += local_reward;
    if (task[i])
      task_1_count += 1;
      if (local_reward == 1)
        proportion_1_correct += 1;
      end
    else 
      task_2_count += 1;
      if (local_reward == 1)
        proportion_2_correct += 1;
      end
    end
    if(verbosity > 0)
      print("\n")
    end
  end
  proportion_correct = monitor_reward / no_trials_in_block;

  proportion_1_correct = proportion_1_correct / task_1_count;
  proportion_2_correct = proportion_2_correct / task_2_count;

  #global wfinal = deepcopy(w)

  if(verbosity > 2)
    # Note this changes how the final proportion_correct is calculated!
    print("END of Learning Block, proportion correct: $proportion_correct, alternating task set.\nProportion 1 correct: $proportion_1_correct, proportion 2 correct: $proportion_2_correct.\n")
    print("DEBUG: task_1_count: $task_1_count, task_2_count: $task_2_count.\n")
    proportion_correct = instance_correct / (instance_correct + instance_incorrect);
    print("DEBUG: instance_correct: $instance_correct, instance_incorrect: $instance_incorrect, new proportion correct: $proportion_correct\n")
  end

  return proportion_correct;
end


function perform_single_subject_experiment(is_trial_1_task::Bool, subjects_dat::Array{Subject,1}, subject_id::Int64=1)
  #global subject
  #subject[subject_id] = Subject(zeros(no_blocks_in_experiment), zeros(no_blocks_in_experiment), zeros(no_blocks_in_experiment), zeros(no_blocks_in_experiment), zeros(no_blocks_in_experiment), zeros(no_blocks_in_experiment), zeros(no_blocks_in_experiment), zeros(no_pre_neurons,2),zeros(no_pre_neurons,2));
  initialise_weight_matrix()
  subjects_dat[subject_id].w_initial = deepcopy(w);

#  if (use_ab_persistence)
    global a = deepcopy(subjects_dat[subject_id].a);
    global b = deepcopy(subjects_dat[subject_id].b);
#=  else
    initialise_pre_population()
    subjects_dat[subject_id].a = deepcopy(a);
    subjects_dat[subject_id].b = deepcopy(b);
  end=#

  global average_reward;
  global n_critic;
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
    subjects_dat[subject_id].blocks[i].proportion_correct = perform_learning_block_single_problem(is_trial_1_task, subjects_dat[subject_id].blocks[i])
    local_average_reward = 0.;
    local_sum_critics = 0;
    for k = 1:no_task_critics
      for j = 1:no_choices_per_task_critics
        local_average_reward += average_reward[k,j] * n_critic[k,j];
        local_sum_critics += n_critic[k,j];
      end
    end
    #subjects[subject_id].blocks[i].average_reward = ( local_average_reward / (no_task_critics * no_choices_per_task_critics) );
    subjects_dat[subject_id].blocks[i].average_reward = ( local_average_reward / local_sum_critics );
    #subjects[subject_id].blocks[i].average_delta_reward = average_delta_reward;
    subjects_dat[subject_id].blocks[i].average_choice = average_choice;
    if(verbosity > -1)
      print("Block $i completed. Type 1 task: $is_trial_1_task.\n") 
    end
    #=if(i == no_blocks_in_experiment && subject_id == 9)
      verbosity = local_old_verbosity;
    end=#
  end
  subjects_dat[subject_id].w_final = deepcopy(w);
  return 0;
end


function perform_single_subject_experiment_trial_switching(subjects::Array{Subject,1}, subject_id::Int64=1)
  #global subject
  #subject[subject_id] = Subject(zeros(no_blocks_in_experiment), zeros(no_blocks_in_experiment), zeros(no_blocks_in_experiment), zeros(no_blocks_in_experiment), zeros(no_blocks_in_experiment), zeros(no_blocks_in_experiment), zeros(no_blocks_in_experiment), zeros(no_pre_neurons,2),zeros(no_pre_neurons,2));
  initialise_weight_matrix()
  subjects[subject_id].w_initial = deepcopy(w);

#  if (use_ab_persistence)
    global a = deepcopy(subjects[subject_id].a)
    global b = deepcopy(subjects[subject_id].b)
#=  else
    initialise_pre_population()
    subjects[subject_id].a = deepcopy(a);
    subjects[subject_id].b = deepcopy(b);
  end=#

  global average_reward;
  global n_critic;
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
    subjects[subject_id].blocks[i].proportion_correct = perform_learning_block_trial_switching(subjects[subject_id].blocks[i])
    local_average_reward = 0.;
    local_sum_critics = 0;
    for k = 1:no_task_critics
      for j = 1:no_choices_per_task_critics
        local_average_reward += average_reward[k,j] * n_critic[k,j];
        local_sum_critics += n_critic[k,j];
      end
    end
    #subjects[subject_id].blocks[i].average_reward = ( local_average_reward / (no_task_critics * no_choices_per_task_critics) );
    subjects[subject_id].blocks[i].average_reward = ( local_average_reward / local_sum_critics );
    #subjects[subject_id].blocks[i].average_delta_reward = average_delta_reward;
    subjects[subject_id].blocks[i].proportion_1_correct = proportion_1_correct;
    subjects[subject_id].blocks[i].proportion_2_correct = proportion_2_correct;
    subjects[subject_id].blocks[i].average_choice = average_choice;
    if(verbosity > -1)
      print("Block $i completed. Alternating tasks.\n") 
    end
  end
  if(double_no_of_trials_in_alternating_experiment)
    no_trials_in_block = int(no_trials_in_block / 2);
  end
  subjects[subject_id].w_final = deepcopy(w);
  return 0;
end


function perform_multi_subject_experiment(is_trial_1_task::Bool, subjects::Array{Subject,1}, no_subjects::Int64=no_subjects)
  #global subject = Array(Subject, no_subjects);

  for(i = 1:no_subjects)
    if(verbosity > -1)
      print("-----------Subject number $i------------\n")
    end
    perform_single_subject_experiment(is_trial_1_task, subjects, i)
  end

  if(verbosity > -1)
    print("No subjects completed: $no_subjects\n")
  end
end


function perform_multi_subject_experiment_trial_switching(subjects::Array{Subject,1}, no_subjects::Int64=no_subjects)
  #global subject = Array(Subject, no_subjects);

  for(i = 1:no_subjects)
    if(verbosity > -1)
      print("-----------Subject number $i------------\n")
    end
    perform_single_subject_experiment_trial_switching(subjects, i)
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

  latest_experiment_results = initialise_empty_roving_experiment(no_subjects, no_blocks_in_experiment, no_trials_in_block);

  if(use_ab_persistence)
    for i = 1:no_subjects
      initialise_pre_population();
      latest_experiment_results.subjects_task1[i].a = deepcopy(a);
      latest_experiment_results.subjects_task1[i].b = deepcopy(b);
      latest_experiment_results.subjects_task2[i].a = deepcopy(a);
      latest_experiment_results.subjects_task2[i].b = deepcopy(b);
      latest_experiment_results.subjects_roving_task[i].a = deepcopy(a);
      latest_experiment_results.subjects_roving_task[i].b = deepcopy(b);
      initialise_pre_population();
      initialise_pre_population();
    end
  else # experiment to have identical RND sequences
    for i = 1:no_subjects
      initialise_pre_population(); 
      latest_experiment_results.subjects_task1[i].a = deepcopy(a);
      latest_experiment_results.subjects_task1[i].b = deepcopy(b);
      initialise_pre_population(); 
      latest_experiment_results.subjects_task2[i].a = deepcopy(a);
      latest_experiment_results.subjects_task2[i].b = deepcopy(b);
      initialise_pre_population(); 
      latest_experiment_results.subjects_roving_task[i].a = deepcopy(a);
      latest_experiment_results.subjects_roving_task[i].b = deepcopy(b);
    end
  end

  print("-----Experiment: task 1------\n")
  perform_multi_subject_experiment(true, latest_experiment_results.subjects_task1);
  mean_correct = zeros(no_blocks_in_experiment);
  range_correct = zeros(no_blocks_in_experiment);
  err_correct = zeros(no_blocks_in_experiment);
  for i = 1:no_blocks_in_experiment
    local_prop = zeros(no_subjects);
    for j = 1:no_subjects
      #mean_correct[i] += latest_experiment_results.subjects_task1[j].blocks[i].proportion_correct;
      local_prop[j] = latest_experiment_results.subjects_task1[j].blocks[i].proportion_correct;
    end
    if(use_plot_mean)
      # mean calculation
      #mean_correct[i] /= no_subjects;
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
  latest_experiment_results.task1_correct = mean_correct;
  latest_experiment_results.task1_error = err_correct; #min_correct;
  latest_experiment_results.task1_range = range_correct;

  print("-----Experiment: task 2------\n")
  perform_multi_subject_experiment(false, latest_experiment_results.subjects_task2);
  mean_correct = zeros(no_blocks_in_experiment);
  range_correct = zeros(no_blocks_in_experiment);
  err_correct = zeros(no_blocks_in_experiment);
  for i = 1:no_blocks_in_experiment
    local_prop = zeros(no_subjects);
    for j = 1:no_subjects
      #mean_correct[i] += latest_experiment_results.subjects_task2[j].blocks[i].proportion_correct;
      local_prop[j] = latest_experiment_results.subjects_task2[j].blocks[i].proportion_correct;
    end
    if(use_plot_mean)
      # mean calculation
      #mean_correct[i] /= no_subjects;
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
  latest_experiment_results.task2_correct = mean_correct;
  latest_experiment_results.task2_error = err_correct;
  latest_experiment_results.task2_range = range_correct;

  print("-----Experiment: roving task------\n")
  perform_multi_subject_experiment_trial_switching(latest_experiment_results.subjects_roving_task);
  mean_correct = zeros(no_blocks_in_experiment);
  mean_task_1_correct = zeros(no_blocks_in_experiment);
  mean_task_2_correct = zeros(no_blocks_in_experiment);
  err_correct = zeros(no_blocks_in_experiment);
  range_correct = zeros(no_blocks_in_experiment);
  for i = 1:no_blocks_in_experiment
    local_prop = zeros(no_subjects);
    local_prop_1 = zeros(no_subjects);
    local_prop_2 = zeros(no_subjects);
    for j = 1:no_subjects
      #mean_correct[i] += latest_experiment_results.subjects_roving_task[j].blocks[i].proportion_correct;
      # mean calculation
      #=mean_task_1_correct[i] += latest_experiment_results.subjects_roving_task[j].blocks[i].proportion_1_correct;
      mean_task_2_correct[i] += latest_experiment_results.subjects_roving_task[j].blocks[i].proportion_2_correct;
      local_prop[j] = latest_experiment_results.subjects_roving_task[j].blocks[i].proportion_correct;=#
      # save the proportions so that mean or median can be called
      local_prop[j] = latest_experiment_results.subjects_roving_task[j].blocks[i].proportion_correct;
      local_prop_1[j] = latest_experiment_results.subjects_roving_task[j].blocks[i].proportion_1_correct;
      local_prop_2[j] = latest_experiment_results.subjects_roving_task[j].blocks[i].proportion_2_correct;
    end
    if(use_plot_mean)
      # mean calculation
      #=mean_task_1_correct[i] /= no_subjects;
      mean_task_2_correct[i] /= no_subjects;=#
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

  latest_experiment_results.roving_correct = mean_correct;
  latest_experiment_results.roving_task1_correct = mean_task_1_correct;
  latest_experiment_results.roving_task2_correct = mean_task_2_correct;
  latest_experiment_results.roving_error = err_correct;
  latest_experiment_results.roving_range = range_correct;

  print("Plotting...\n")
  # legend(loc=4)
  #figure()
  #plot_multi_subject_experiment(latest_experiment_results);
  plot_multi_subject_experiment_as_subplots(latest_experiment_results);

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
        scatter(i, latest_experiment_results.subjects_task1[j].blocks[i].proportion_correct, marker="o", c="r")
        scatter(i+0.1, latest_experiment_results.subjects_task2[j].blocks[i].proportion_correct, marker="o", c="g")
        scatter(i-0.1, latest_experiment_results.subjects_roving_task[j].blocks[i].proportion_correct, marker="o", c="b")
      end
    end
  end

  block_id = linspace(1,no_blocks_in_experiment, no_blocks_in_experiment);

  if(plotting_error_bars_on)
    
    errorbar(block_id, latest_experiment_results.task1_correct, latest_experiment_results.task1_range, ecolor="r", color="r", linewidth=2)
    errorbar(block_id+0.1, latest_experiment_results.task2_correct, latest_experiment_results.task2_range, ecolor="g", color="g", linewidth=2)
    errorbar(block_id-0.1, latest_experiment_results.roving_correct, latest_experiment_results.roving_range, ecolor="b", color="b", linewidth=2)

    errorbar(block_id, latest_experiment_results.task1_correct, latest_experiment_results.task1_error, ecolor="k", color="r", linewidth=2)
    errorbar(block_id+0.1, latest_experiment_results.task2_correct, latest_experiment_results.task2_error, ecolor="k", color="g", linewidth=2)
    errorbar(block_id-0.1, latest_experiment_results.roving_correct, latest_experiment_results.roving_error, ecolor="k", color="b", linewidth=2)
  end

  if(plotting_individual_subjects_on)
    for j = 1:no_subjects
      local_prop_1_correct = zeros(no_blocks_in_experiment);
      local_prop_2_correct = zeros(no_blocks_in_experiment);
      local_prop_roving_correct = zeros(no_blocks_in_experiment);
      for i = 1:no_blocks_in_experiment
        local_prop_1_correct[i] = latest_experiment_results.subjects_task1[j].blocks[i].proportion_correct;
        local_prop_2_correct[i] = latest_experiment_results.subjects_task2[j].blocks[i].proportion_correct;
        local_prop_roving_correct[i] = latest_experiment_results.subjects_roving_task[j].blocks[i].proportion_correct;
      end
      plot(block_id, local_prop_1_correct, "r")
      plot(block_id+0.1, local_prop_2_correct, "g")
      plot(block_id-0.1, local_prop_roving_correct, "b")
    end
  end

  plot(block_id, latest_experiment_results.task1_correct, "r", linewidth=2, label="Task 1")
  plot(block_id+0.1, latest_experiment_results.task2_correct, "g", linewidth=2, label="Task 2")
  plot(block_id-0.1, latest_experiment_results.roving_correct, "b", linewidth=3, label="Roving tasks")
  plot(block_id-0.1, latest_experiment_results.roving_task1_correct, "k", linewidth=3, label="Task 1, from roving tasks")
  plot(block_id-0.1, latest_experiment_results.roving_task2_correct, "k", linewidth=3, label="Task 2, from roving tasks")
  
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
        scatter(i, latest_experiment_results.subjects_task1[j].blocks[i].proportion_correct, marker="o", c="r")
      end
    end
  end

  if(plotting_error_bars_on)
    errorbar(block_id, latest_experiment_results.task1_correct, latest_experiment_results.task1_range, ecolor="r", color="r", linewidth=2)
    errorbar(block_id, latest_experiment_results.task1_correct, latest_experiment_results.task1_error, ecolor="k", color="r", linewidth=2)
  end

  if(plotting_individual_subjects_on)
    for j = 1:no_subjects
      local_prop_1_correct = zeros(no_blocks_in_experiment);
      for i = 1:no_blocks_in_experiment
        local_prop_1_correct[i] = latest_experiment_results.subjects_task1[j].blocks[i].proportion_correct;
      end
      plot(block_id, local_prop_1_correct, "r")
    end
  end

  plot(block_id, latest_experiment_results.task1_correct, "r", linewidth=2, label="Task 1")
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
        scatter(i+0.1, latest_experiment_results.subjects_task2[j].blocks[i].proportion_correct, marker="o", c="g")
      end
    end
  end

  if(plotting_error_bars_on)
    errorbar(block_id+0.1, latest_experiment_results.task2_correct, latest_experiment_results.task2_range, ecolor="g", color="g", linewidth=2)
    errorbar(block_id+0.1, latest_experiment_results.task2_correct, latest_experiment_results.task2_error, ecolor="k", color="g", linewidth=2)
  end

  if(plotting_individual_subjects_on)
    for j = 1:no_subjects
      local_prop_2_correct = zeros(no_blocks_in_experiment);
      for i = 1:no_blocks_in_experiment
        local_prop_2_correct[i] = latest_experiment_results.subjects_task2[j].blocks[i].proportion_correct;
      end
      plot(block_id, local_prop_2_correct, "g")
    end
  end

  plot(block_id+0.1, latest_experiment_results.task2_correct, "g", linewidth=2, label="Task 2")
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
        scatter(i-0.1, latest_experiment_results.subjects_roving_task[j].blocks[i].proportion_correct, marker="o", c="b")
      end
    end
  end

  if(plotting_error_bars_on)
    errorbar(block_id-0.1, latest_experiment_results.roving_correct, latest_experiment_results.roving_range, ecolor="b", color="b", linewidth=2)
    errorbar(block_id-0.1, latest_experiment_results.roving_correct, latest_experiment_results.roving_error, ecolor="k", color="b", linewidth=2)
  end

  if(plotting_individual_subjects_on)
    for j = 1:no_subjects
      local_prop_roving_correct = zeros(no_blocks_in_experiment);
      for i = 1:no_blocks_in_experiment
        local_prop_roving_correct[i] = latest_experiment_results.subjects_roving_task[j].blocks[i].proportion_correct;
      end
      plot(block_id-0.1, local_prop_roving_correct, "b")
    end
  end

  plot(block_id-0.1, latest_experiment_results.roving_correct, "b", linewidth=3, label="Roving tasks")
  plot(block_id-0.1, latest_experiment_results.roving_task1_correct, "k", linewidth=3, label="Task 1, from roving tasks")
  plot(block_id-0.1, latest_experiment_results.roving_task2_correct, "k", linewidth=3, label="Task 2, from roving tasks")
  
  
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
        scatter(i, latest_experiment_results.subjects_task1[j].blocks[i].average_reward, marker="o", c="r")
      end
    end
  end

  if(plotting_individual_subjects_on)
    for j = 1:no_subjects
      local_1_reward = zeros(no_blocks_in_experiment);
      for i = 1:no_blocks_in_experiment
        local_1_reward[i] = latest_experiment_results.subjects_task1[j].blocks[i].average_reward;
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
        scatter(i+0.1, latest_experiment_results.subjects_task2[j].blocks[i].average_reward, marker="o", c="g")
      end
    end
  end

  if(plotting_individual_subjects_on)
    for j = 1:no_subjects
      local_1_reward = zeros(no_blocks_in_experiment);
      for i = 1:no_blocks_in_experiment
        local_1_reward[i] = latest_experiment_results.subjects_task2[j].blocks[i].average_reward;
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
        scatter(i-0.1, latest_experiment_results.subjects_roving_task[j].blocks[i].average_reward, marker="o", c="b")
      end
    end
  end

  if(plotting_individual_subjects_on)
    for j = 1:no_subjects
      local_roving_reward = zeros(no_blocks_in_experiment);
      for i = 1:no_blocks_in_experiment
        local_roving_reward[i] = latest_experiment_results.subjects_roving_task[j].blocks[i].average_reward;
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
        scatter(i, latest_experiment_results.subjects_task1[j].blocks[i].average_choice, marker="o", c="r")
      end
    end
  end

  if(plotting_individual_subjects_on)
    for j = 1:no_subjects
      local_1_choice = zeros(no_blocks_in_experiment);
      for i = 1:no_blocks_in_experiment
        local_1_choice[i] = latest_experiment_results.subjects_task1[j].blocks[i].average_choice;
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
        scatter(i+0.1, latest_experiment_results.subjects_task2[j].blocks[i].average_choice, marker="o", c="g")
      end
    end
  end

  if(plotting_individual_subjects_on)
    for j = 1:no_subjects
      local_1_choice = zeros(no_blocks_in_experiment);
      for i = 1:no_blocks_in_experiment
        local_1_choice[i] = latest_experiment_results.subjects_task2[j].blocks[i].average_choice;
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
        scatter(i-0.1, latest_experiment_results.subjects_roving_task[j].blocks[i].average_choice, marker="o", c="b")
      end
    end
  end

  if(plotting_individual_subjects_on)
    for j = 1:no_subjects
      local_roving_choice = zeros(no_blocks_in_experiment);
      for i = 1:no_blocks_in_experiment
        local_roving_choice[i] = latest_experiment_results.subjects_roving_task[j].blocks[i].average_choice;
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
      plot(latest_experiment_results.task1_correct, "r", linewidth=2, label="Task 1")
      plot(latest_experiment_results.task2_correct, "g", linewidth=2, label="Task 2")
      plot(latest_experiment_results.roving_correct, "b", linewidth=3, label="Roving tasks")
      plot(latest_experiment_results.roving_task1_correct, "k", linewidth=3, label="Task 1, from roving tasks")
      plot(latest_experiment_results.roving_task2_correct, "k", linewidth=3, label="Task 2, from roving tasks")
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
  for i = 1:length(block.trial)
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
  local_av_n = length(block.trial);
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

function plot_single_block_performance(block::Block)
  #figure()
  local_reward_received = zeros(no_trials_in_block);
  x = linspace(1, no_trials_in_block, no_trials_in_block);
  for i = 1:no_trials_in_block
    local_reward_received[i] = block.trial[i].reward_received;
    #print("", x[i], " ", local_reward_received[i], "\n")
  end
  #print("", size(local_reward_received), " ", size(x),"\n")
  plot(x, local_reward_received, linewidth=2)
end

function plot_multi_block_performance(subject::Subject, begin_id::Int=1, end_id::Int=no_blocks_in_experiment)
  figure()
  for i = begin_id:end_id
    plot_single_block_performance(subject.blocks[i])
  end
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
end

function plot_multi_block_reward_recived(subject::Subject, begin_id::Int=1, end_id::Int=no_blocks_in_experiment)
  figure()
  for i = begin_id:end_id
    plot_single_block_reward_received(subject.blocks[i])
  end
  xlabel("Trial number")
  ylabel("Reward received")
  axis([0,no_trials_in_block,-2,2])
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
end

function plot_multi_block_mag_dw(subject::Subject, begin_id::Int=1, end_id::Int=no_blocks_in_experiment)
  figure()
  for i = begin_id:end_id
    plot_single_block_mag_dw(subject.blocks[i])
  end
  xlabel("Trial number")
  ylabel("Magnitude dw")
end


function plot_single_subject_average_reward(subject::Subject)
  #figure()
  local_av_reward = zeros(no_blocks_in_experiment);
  x = linspace(1, no_blocks_in_experiment, no_blocks_in_experiment);
  for i = 1:no_blocks_in_experiment
    local_av_reward[i] = subject.blocks[i].average_reward;
    #print("", x[i], " ", local_reward_received[i], "\n")
  end
  #print("", size(local_reward_received), " ", size(x),"\n")
  plot(x, local_av_reward, linewidth=2)
end

function plot_multi_subject_average_reward(subjects::Array{Subject,1}, begin_id::Int=1, end_id::Int=no_subjects)
  figure()
  for i = begin_id:end_id
    plot_single_subject_average_reward(subjects[i])
  end
end


function plot_single_subject_average_choice(subject::Subject)
  #figure()
  local_av_choice = zeros(no_blocks_in_experiment);
  x = linspace(1, no_blocks_in_experiment, no_blocks_in_experiment);
  for i = 1:no_blocks_in_experiment
    local_av_choice[i] = (subject.blocks[i].average_choice);# - 1) * 2;
    #print("", x[i], " ", local_reward_received[i], "\n")
  end
  #print("", size(local_reward_received), " ", size(x),"\n")
  plot(x, local_av_choice, linewidth=2)
end

function plot_multi_subject_average_choice(subjects::Array{Subject,1}, begin_id::Int=1, end_id::Int=no_subjects)
  figure()
  for i = begin_id:end_id
    plot_single_subject_average_choice(subjects[i])
  end
end


function plot_single_subject_final_weight_vs_bias(subject::Subject, is_task_1::Bool=true, is_task_2::Bool=false)
  #figure()
  easy_size = length(subject.b) / 2 :: Int;
  if (is_task_1)
    scatter(subject.b[1:easy_size], subject.w_final[1:easy_size,1], marker="o", c="g", label="left, easy")
    scatter(subject.b[1:easy_size], subject.w_final[1:easy_size,2], marker="o", c="y", label="right, easy")
  end
  if (is_task_2)
    scatter(subject.b[easy_size+1:end], subject.w_final[easy_size+1:end,1], marker="o", c="r", label="left, hard")
    scatter(subject.b[easy_size+1:end], subject.w_final[easy_size+1:end,2], marker="o", c="k", label="right, hard")
  end
  xlim([-1.2,1.2])
  ylim([-12,12])
  #legend()
end

function plot_multi_subject_final_weight_vs_bias(subjects::Array{Subject,1}, begin_id::Int=1, end_id::Int=no_subjects)
  figure(figsize=(4,20))
  for i = begin_id:end_id
    subplot(10,1,i)
    plot_single_subject_final_weight_vs_bias(subjects[i]);
  end
  legend()
end


function plot_single_subject_initial_weight_vs_bias(subject::Subject, is_task_1::Bool=true, is_task_2::Bool=false)
  #figure()
  easy_size = length(subject.b) / 2 :: Int;
  if (is_task_1)
    scatter(subject.b[1:easy_size], subject.w_initial[1:easy_size,1], marker="o", c="g", label="left, easy")
    scatter(subject.b[1:easy_size], subject.w_initial[1:easy_size,2], marker="o", c="y", label="right, easy")
  end
  if (is_task_2)
    scatter(subject.b[easy_size+1:end], subject.w_initial[easy_size+1:end,1], marker="o", c="r", label="left, hard")
    scatter(subject.b[easy_size+1:end], subject.w_initial[easy_size+1:end,2], marker="o", c="k", label="right, hard")
  end
  xlim([-1.2,1.2])
  ylim([-12,12])
  #legend()
end

function plot_multi_subject_initial_weight_vs_bias(subjects::Array{Subject,1}, begin_id::Int=1, end_id::Int=no_subjects)
  figure(figsize=(4,20))
  for i = begin_id:end_id
    subplot(10,1,i)
    plot_single_subject_initial_weight_vs_bias(subjects[i]);
  end
  legend()
end



function who_doesnt_learn(subjects::Array{Subject,1}, threshold::Float64=10.0, begin_id::Int=1, end_id::Int=no_subjects)
  print("Detection threshold: $threshold\n")
  for i = begin_id:end_id
    if ( subjects[i].blocks[end].proportion_correct < threshold )
      print("Subject $i proportion correct:", subjects[i].blocks[end].proportion_correct,"\n")
    end
  end
end


function restore_subject(subject::Subject, initial_weights::Bool=true)
  global a = deepcopy(subject.a);
  global b = deepcopy(subject.b);
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




