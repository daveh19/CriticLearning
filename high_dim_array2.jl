
# Gaussian tuning function definition parameters
#    for a single presynaptic input neuron
type gaussian_tc_type
  no_curves :: Int;
  mu :: Array{Float64, 1};
  sigma :: Array{Float64, 1};
  height :: Array{Float64, 1};
end

######### Data Storage ##############

type Trial
  # data for a single subject in a single trial
  task_type :: Int;
  correct_answer :: Float64;
  chosen_answer :: Float64;
  got_it_right :: Bool;
  reward_received :: Float64;
  mag_dw :: Float64;
  error_threshold :: Float64;
  # Note: dimension of w is 3, first dimension are the elements corresponding
  #   a set of unified inputs, second dimension the output directions, third
  #   dimension the separate input tasks
  w :: Array{Float64, 3};
  dw :: Array{Float64, 3};
  decision_criterion_monitor :: Array{Float64,2};
end

function initialise_empty_trials(no_trials::Int)
  trial = Array{Trial}(no_trials);

  for i = 1:no_trials
    trial[i] = Trial( 0, 0., 0., false, 0., 0., 0., zeros(no_pre_neurons_per_task, no_post_neurons, no_input_tasks), zeros(no_pre_neurons_per_task, no_post_neurons, no_input_tasks), zeros(no_decision_monitors,1) );
  end

  return trial;
end


type Block
  # an array of trials for this block
  trial :: Array{Trial, 1};
  # summary statistics
  proportion_correct :: Float64;
  average_reward :: Float64; # this is the state of the running average at the end of the block
  average_choice :: Float64;
  average_threshold :: Float64;
  average_decision_criterion_monitor :: Array{Float64,2};

  proportion_task_correct :: Array{Float64, 1};
  average_task_reward :: Array{Float64, 1}; # this is a true per task average for the block
  average_task_choice :: Array{Float64, 1};
  average_task_threshold :: Array{Float64, 1};
  #proportion_1_correct :: Float64;
  #proportion_2_correct :: Float64;
  #average_delta_reward :: Float64;

  noise_free_positive_output :: Array{Float64, 2};
  probability_correct :: Array{Float64, 2};

  reward_prediction :: Array{Float64, 2};
end

function initialise_empty_block(no_blocks::Int, trials_per_block::Int, double_trials::Bool=false)
  block = Array{Block}(no_blocks);

  if (double_trials)
    trials_per_block *= 2;
  end

  for i = 1:no_blocks
    local_trial = initialise_empty_trials(trials_per_block);
    block[i] = Block( local_trial, 0., 0., 0., 0., zeros(no_decision_monitors,1), zeros(no_input_tasks), zeros(no_input_tasks), zeros(no_input_tasks), zeros(no_input_tasks), zeros(no_input_tasks, no_classifications_per_task), zeros(no_input_tasks, no_classifications_per_task), zeros(2,1) );
  end

  return block;
end


abstract type Subject end
type LinearInputsSubject <: Subject
  # an array of blocks for this subject
  blocks :: Array{Block, 1}
  # summary information for this subject
  # inherent receptive field, this is unique per subject and does not change
  a :: Array{Float64, 2}
  b :: Array{Float64, 2}
  # initial weights at beginning of experiment
  w_initial :: Array{Float64, 3}
  # final weights at end of experiment
  w_final :: Array{Float64, 3}
end
type GaussianInputsSubject <: Subject
  # an array of blocks for this subject
  blocks :: Array{Block, 1}
  # summary information for this subject
  # inherent receptive field, this is unique per subject and does not change
  a :: Array{gaussian_tc_type, 2}
  #b :: Array{Float64, 2}
  # initial weights at beginning of experiment
  w_initial :: Array{Float64, 3}
  # final weights at end of experiment
  w_final :: Array{Float64, 3}
end

# linear tuning function on inputs
function initialise_empty_subject(tuning_type::linear_tc, blocks_per_subject::Int, trials_per_block::Int, double_trials::Bool=false)
  blocks = initialise_empty_block(blocks_per_subject, trials_per_block, double_trials);
  subject = LinearInputsSubject( blocks, zeros(no_pre_neurons_per_task, no_input_tasks), zeros(no_pre_neurons_per_task, no_input_tasks), zeros(no_pre_neurons_per_task, no_post_neurons, no_input_tasks), zeros(no_pre_neurons_per_task, no_post_neurons, no_input_tasks) );

  return subject;
end
# gaussian tuning function on inputs
function initialise_empty_subject(tuning_type::gaussian_tc, blocks_per_subject::Int, trials_per_block::Int, double_trials::Bool=false)
  blocks = initialise_empty_block(blocks_per_subject, trials_per_block, double_trials);
  a = Array(gaussian_tc_type, (no_pre_neurons_per_task, no_input_tasks) );
  subject = GaussianInputsSubject( blocks, a, zeros(no_pre_neurons_per_task, no_post_neurons, no_input_tasks), zeros(no_pre_neurons_per_task, no_post_neurons, no_input_tasks) );

  return subject;
end


type RovingExperiment
  # Second dimension of the arrays is for per task versions of results
  #   we'll allow a second dimension also for roving tasks but there's
  #   only one task type in that category so far

  # an array of subjects who participate in experiment
  subjects_task :: Array{Subject, 2}
  subjects_roving_task :: Array{Subject, 2}

  # summary statistics of experiment
  task_correct :: Array{Float64,2}
  roving_correct :: Array{Float64,2}
  roving_task_correct :: Array{Float64,3} # 3 dimensions: individual trace, per task ID, per roving experiment

  task_error :: Array{Float64,2}
  roving_error :: Array{Float64,2}

  task_range :: Array{Float64,2}
  roving_range :: Array{Float64,2}
end

function initialise_empty_roving_experiment(tuning_type::TuningSelector, no_subjects::Int, blocks_per_subject::Int, trials_per_block::Int, no_roving_experiments::Int)
  #no_roving_tasks = 1::Int;

  subjects_task = Array{Subject}((no_subjects, no_input_tasks) );
  subjects_roving_task = Array{Subject}((no_subjects, no_roving_experiments) );

  task_correct = zeros(blocks_per_subject, no_input_tasks);
  roving_correct = zeros(blocks_per_subject, no_roving_experiments);
  roving_task_correct = zeros(blocks_per_subject, no_input_tasks, no_roving_experiments);

  task_error = zeros(blocks_per_subject, no_input_tasks);
  roving_error = zeros(blocks_per_subject, no_input_tasks);

  task_range = zeros(blocks_per_subject, no_input_tasks);
  roving_range = zeros(blocks_per_subject, no_input_tasks);

  for i = 1:no_subjects
    for j = 1:no_input_tasks
      subjects_task[i,j] = initialise_empty_subject(tuning_type, blocks_per_subject, trials_per_block);
    end
    for j = 1:no_roving_experiments
      subjects_roving_task[i,j] = initialise_empty_subject(tuning_type, blocks_per_subject, trials_per_block, double_no_of_trials_in_alternating_experiment);
    end
  end
  experiment = RovingExperiment(subjects_task, subjects_roving_task, task_correct, roving_correct, roving_task_correct, task_error, roving_error, task_range, roving_range );

  return experiment;
end
