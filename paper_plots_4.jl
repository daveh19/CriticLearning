# do a simulation which has a simple critic which
#  sends no weight update signals when there are two
#  tasks, until enough blocks have passed, upon which
#  time two critics are activated
# for version 2.0 of manuscript (October 2017)

print("Zeroing results!!\n");
include("multi_critic_detailed_recording.jl");
using PyPlot
using LaTeXStrings;

@pyimport seaborn as sns
#sns.set(font_scale=1.5)
# sns.set_context("poster")
#sns.set_context("talk")
#sns.set(font_scale=3)
#exp_results = [];


no_blocks_in_experiment = 60 :: Int;
no_blocks_to_maintain_simple_variance_cut_off = 30 :: Int;
plotting_hack_to_have_separate_choices_in_roving_example = false :: Bool;
initialise();

tuning_type = linear_tc();
no_roving_experiments = 1::Int;
latest_experiment_results = initialise_empty_roving_experiment(tuning_type, no_subjects, no_blocks_in_experiment, no_trials_in_block, no_roving_experiments);

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


print("-----Experiment: roving task------\n")
roving_experiment_id = 1 :: Int;

# this is the long line to run
perform_multi_subject_experiment_trial_switching(tuning_type, latest_experiment_results.subjects_roving_task, no_subjects);

# summary statistics
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



latest_experiment_results.roving_correct[:,roving_experiment_id] = mean_correct;
latest_experiment_results.roving_task_correct[:,1,roving_experiment_id] = mean_task_1_correct;
latest_experiment_results.roving_task_correct[:,2,roving_experiment_id] = mean_task_2_correct;
latest_experiment_results.roving_error[:,roving_experiment_id] = err_correct;
latest_experiment_results.roving_range[:,roving_experiment_id] = range_correct;



# export results
global exp_results;
resize!(exp_results, length(exp_results)+1);
exp_results[length(exp_results)] = latest_experiment_results;


print("Plotting...\n")

function plot_figure_simple_2_critic(results_id=1::Int)
  # uses:
  #    compare_three_trial_types_with_multiple_subjects()
  #    paper_binary_inputs_parameters_critic_simulations.jl
  # shows:
  #

  plotting_separate_choices_on = false;

  latest_experiment_results = exp_results[results_id];

  figure(figsize=(10,12))

  block_id = linspace(1,no_blocks_in_experiment, no_blocks_in_experiment);

  xlim((0-0.1,no_blocks_in_experiment+0.1))
  ylim((0-0.02,1+0.02))
  xlabel("Block number")
  ylabel("Proportion correct")

  if(plotting_scatter_plot_on)
    for i = 1:no_blocks_in_experiment
      for j = 1:no_subjects
        #scatter(i-0., latest_experiment_results.subjects_roving_task[j,1].blocks[i].proportion_correct, marker="o", edgecolors="face", c="b", alpha=0.7)
        if(plotting_task_by_task_on)
          scatter(i-0., latest_experiment_results.subjects_roving_task[j,1].blocks[i].proportion_task_correct[1], marker="o", edgecolors="face", c="r", alpha=0.3)
          scatter(i-0., latest_experiment_results.subjects_roving_task[j,1].blocks[i].proportion_task_correct[2], marker="o", edgecolors="face", c="g", alpha=0.3)
        end
      end
    end
  end

  if(plotting_error_bars_on)
    errorbar(block_id-0., latest_experiment_results.roving_correct[:,1], latest_experiment_results.roving_range[:,1], ecolor="b", color="b", linewidth=2)
    errorbar(block_id-0., latest_experiment_results.roving_correct[:,1], latest_experiment_results.roving_error[:,1], ecolor="k", color="b", linewidth=2)
  end

  if(plotting_individual_subjects_on)
    for j = 1:no_subjects
      local_prop_roving_correct = zeros(no_blocks_in_experiment);
      local_prop_roving_task_1_correct = zeros(no_blocks_in_experiment);
      local_prop_roving_task_2_correct = zeros(no_blocks_in_experiment);
      for i = 1:no_blocks_in_experiment
        local_prop_roving_correct[i] = latest_experiment_results.subjects_roving_task[j,1].blocks[i].proportion_correct;
        local_prop_roving_task_1_correct[i] = latest_experiment_results.subjects_roving_task[j,1].blocks[i].proportion_task_correct[1];
        local_prop_roving_task_2_correct[i] = latest_experiment_results.subjects_roving_task[j,1].blocks[i].proportion_task_correct[2];
      end
      if(plotting_task_by_task_on)
        plot(block_id-0., local_prop_roving_task_1_correct, "r", alpha=0.1)
        plot(block_id-0., local_prop_roving_task_2_correct, "g", alpha=0.1)
      end
      #plot(block_id-0., local_prop_roving_correct, "b", alpha=0.1)
    end
  end

  #plot(block_id-0., latest_experiment_results.roving_correct[:,1], "b", linewidth=3, label="Block mean performance average")
  plot(block_id-0., latest_experiment_results.roving_task_correct[:,1,1], "r", linewidth=3, label="Performance on Bisection Task 1")
  plot(block_id-0., latest_experiment_results.roving_task_correct[:,2,1], "g", linewidth=3, label="Performance on Bisection Task 2")

  # legend(loc=4)

  #savefig("figure_1_2.pdf", transparent="True", bbox_inches="tight", pad_inches=0.1)
  savefig("figure_simple_learning_2_critic.pdf", bbox_inches="tight", pad_inches=0.1)
end

plot_figure_simple_2_critic()

print("End\n");
