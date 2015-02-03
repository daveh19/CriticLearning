
######### Data Storage ##############

type Trial
  # data for a single subject in a single trial
  task_type :: Int;
  correct_answer :: Float64;
  chosen_answer :: Float64;
  got_it_right :: Bool;
  reward_received :: Float64;
  w :: Array{Float64, 2};
  dw :: Array{Float64, 2};
  mag_dw :: Float64;
  error_probability :: Float64;
end

function initialise_empty_trials(no_trials)
  trial = Array(Trial, no_trials);

  for i = 1:no_trials
    trial[i] = Trial( 0, 0, 0, false, 0, zeros(no_pre_neurons,2), zeros(no_pre_neurons,2), 0, 0 );
  end

  return trial;
end


type Block
  # an array of trials for this block
  trial :: Array{Trial, 1};
  # summary statistics
  proportion_correct :: Float64;
  proportion_1_correct :: Float64;
  proportion_2_correct :: Float64;
  average_reward :: Float64;
  average_delta_reward :: Float64;
  average_choice :: Float64;
end

function initialise_empty_block(no_blocks, trials_per_block)
  block = Array(Block, no_blocks);

  for i = 1:no_blocks
    local_trial = initialise_empty_trials(trials_per_block);
    block[i] = Block( local_trial, 0, 0, 0, 0, 0, 0);
  end  

  return block;
end


type Subject
  # an array of blocks for this subject
  blocks :: Array{Block, 1}
  # summary information for this subject
  # inherent receptive field, this is unique per subject and does not change
  a :: Array{Float64, 1} 
  b :: Array{Float64, 1}
  # initial weights at beginning of experiment
  w_initial :: Array{Float64, 2}
  # final weights at end of experiment
  w_final :: Array{Float64, 2}
end

function initialise_empty_subject(blocks_per_subject, trials_per_block)
  blocks = initialise_empty_block(blocks_per_subject, trials_per_block);
  subject = Subject( blocks, zeros(no_pre_neurons), zeros(no_pre_neurons), zeros(no_pre_neurons,2), zeros(no_pre_neurons, 2));

  return subject;
end


type RovingExperiment
  # an array of subjects who participate in experiment
  subjects_task1 :: Array{Subject, 1}
  subjects_task2 :: Array{Subject, 1}
  subjects_roving_task :: Array{Subject, 1}

  # summary statistics of experiment
  task1_correct :: Array{Float64,1}
  task2_correct :: Array{Float64,1}
  roving_correct :: Array{Float64,1}
  roving_task1_correct :: Array{Float64,1}
  roving_task2_correct :: Array{Float64,1}

  task1_error :: Array{Float64,1}
  task2_error :: Array{Float64,1}
  roving_error :: Array{Float64,1}

  task1_range :: Array{Float64,1}
  task2_range :: Array{Float64,1}
  roving_range :: Array{Float64,1}
end

function initialise_empty_roving_experiment(no_subjects, blocks_per_subject, trials_per_block)
  subjects_task1 = Array(Subject, no_subjects);
  subjects_task2 = Array(Subject, no_subjects);
  subjects_roving_task = Array(Subject, no_subjects);

  task1_correct = zeros(blocks_per_subject);
  task2_correct = zeros(blocks_per_subject);
  roving_correct = zeros(blocks_per_subject);
  roving_task1_correct = zeros(blocks_per_subject);
  roving_task2_correct = zeros(blocks_per_subject);

  task1_error = zeros(blocks_per_subject);
  task2_error = zeros(blocks_per_subject);
  roving_error = zeros(blocks_per_subject);

  task1_min = zeros(blocks_per_subject);
  task2_min = zeros(blocks_per_subject);
  roving_min = zeros(blocks_per_subject);

  for i = 1:no_subjects
      subjects_task1[i] = initialise_empty_subject(blocks_per_subject, trials_per_block);
      subjects_task2[i] = initialise_empty_subject(blocks_per_subject, trials_per_block);
      subjects_roving_task[i] = initialise_empty_subject(blocks_per_subject, trials_per_block);
  end
  experiment = RovingExperiment(subjects_task1, subjects_task2, subjects_roving_task, task1_correct, task2_correct, roving_correct, roving_task1_correct, roving_task2_correct, task1_error, task2_error, roving_error, task1_min, task2_min, roving_min);

  return experiment;
end
