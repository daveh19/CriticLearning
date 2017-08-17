# for preprint article August 2017
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

# Run the full set of simulations
# compare_three_trial_types_with_multiple_subjects()
biased_compare_three_trial_types_with_multiple_subjects()

function plot_figure_2(results_id=1::Int, exp_id=2::Int)
  # uses:
  #    compare_three_trial_types_with_multiple_subjects()
  #    paper_binary_inputs_parameters_critic_simulations.jl
  # shows:
  #

  plotting_separate_choices_on = true;

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
        # TODO: How do I get separate left-right performance out of roving task?
        if(plotting_separate_choices_on)
          # adding plotting of sub-task related results
          scatter(i+0., latest_experiment_results.subjects_roving_task[j,exp_id].blocks[i].proportion_task_correct[1], marker="o", c="c")
          scatter(i+0., latest_experiment_results.subjects_roving_task[j,exp_id].blocks[i].proportion_task_correct[2], marker="o", c="m")
        end
        # END paste
        #scatter(i-0., latest_experiment_results.subjects_roving_task[j,1].blocks[i].proportion_correct, marker="o", edgecolors="face", c="b", alpha=0.7)
        if(plotting_task_by_task_on)
          scatter(i-0., latest_experiment_results.subjects_roving_task[j,exp_id].blocks[i].proportion_task_correct[1], marker="o", edgecolors="face", c="r", alpha=0.3)
          scatter(i-0., latest_experiment_results.subjects_roving_task[j,exp_id].blocks[i].proportion_task_correct[2], marker="o", edgecolors="face", c="g", alpha=0.3)
        end
      end
    end
  end

  if(plotting_error_bars_on)
    errorbar(block_id-0., latest_experiment_results.roving_correct[:,exp_id], latest_experiment_results.roving_range[:,exp_id], ecolor="b", color="b", linewidth=2)
    errorbar(block_id-0., latest_experiment_results.roving_correct[:,exp_id], latest_experiment_results.roving_error[:,exp_id], ecolor="k", color="b", linewidth=2)
  end

  if(plotting_individual_subjects_on)
    for j = 1:no_subjects
      local_prop_roving_correct = zeros(no_blocks_in_experiment);
      local_prop_roving_task_1_correct = zeros(no_blocks_in_experiment);
      local_prop_roving_task_2_correct = zeros(no_blocks_in_experiment);
      for i = 1:no_blocks_in_experiment
        local_prop_roving_correct[i] = latest_experiment_results.subjects_roving_task[j,exp_id].blocks[i].proportion_correct;
        local_prop_roving_task_1_correct[i] = latest_experiment_results.subjects_roving_task[j,exp_id].blocks[i].proportion_task_correct[1];
        local_prop_roving_task_2_correct[i] = latest_experiment_results.subjects_roving_task[j,exp_id].blocks[i].proportion_task_correct[2];
      end
      if(plotting_task_by_task_on)
        plot(block_id-0., local_prop_roving_task_1_correct, "c")#, alpha=0.1)
        plot(block_id-0., local_prop_roving_task_2_correct, "m")#, alpha=0.1)
      end
      #plot(block_id-0., local_prop_roving_correct, "b", alpha=0.1)
    end
  end

  #plot(block_id-0., latest_experiment_results.roving_correct[:,1], "b", linewidth=3, label="Block mean performance average")
  # plot(block_id-0., latest_experiment_results.roving_task_correct[:,1,exp_id], "r", linewidth=3, label="Block mean performance on Bisection Task 1")
  plot(block_id-0., latest_experiment_results.roving_task_correct[:,2,exp_id], "g", linewidth=3, label="Block mean performance on Bisection Task 2")

  # legend(loc=4)

  #savefig("figure_1_2.pdf", transparent="True", bbox_inches="tight", pad_inches=0.1)
  savefig("figure_biased_sim_strong_bias.pdf", bbox_inches="tight", pad_inches=0.1)
end



function plot_figure_1(results_id=1::Int)
  # uses:
  #    compare_three_trial_types_with_multiple_subjects()
  #    paper_binary_inputs_parameters_critic_simulations.jl
  # shows:
  #

  plotting_separate_choices_on = true;

  latest_experiment_results = exp_results[results_id];

  figure(figsize=(10,12))

  block_id = linspace(1,no_blocks_in_experiment, no_blocks_in_experiment);

  ## Task 2 subplot
  xlim((0-0.1,no_blocks_in_experiment+0.1))
  ylim((0-0.02,1+0.02))
  xlabel("Block number")
  ylabel("Proportion correct")

  if(plotting_scatter_plot_on)
    for i = 1:no_blocks_in_experiment
      for j = 1:no_subjects
        scatter(i+0., latest_experiment_results.subjects_task[j,2].blocks[i].proportion_correct, marker="o", edgecolors="face", c="g", alpha=0.5)
        if(plotting_separate_choices_on)
          # adding plotting of sub-task related results
          scatter(i+0., latest_experiment_results.subjects_task[j,2].blocks[i].proportion_task_correct[1], marker="o", c="c")
          scatter(i+0., latest_experiment_results.subjects_task[j,2].blocks[i].proportion_task_correct[2], marker="o", c="m")
        end
      end
    end
  end

  if(plotting_error_bars_on)
    errorbar(block_id+0., latest_experiment_results.task_correct[:,2], latest_experiment_results.task_range[:,2], ecolor="g", color="g", linewidth=2)
    errorbar(block_id+0., latest_experiment_results.task_correct[:,2], latest_experiment_results.task_error[:,2], ecolor="k", color="g", linewidth=2)
  end

  if(plotting_individual_subjects_on)
    for j = 1:no_subjects
      local_prop_2_correct = zeros(no_blocks_in_experiment);
      # adding plotting of sub-task related results
      local_prop_sub_1_correct = zeros(no_blocks_in_experiment);
      local_prop_sub_2_correct = zeros(no_blocks_in_experiment);
      for i = 1:no_blocks_in_experiment
        local_prop_2_correct[i] = latest_experiment_results.subjects_task[j,2].blocks[i].proportion_correct;
        local_prop_sub_1_correct[i] = latest_experiment_results.subjects_task[j,2].blocks[i].proportion_task_correct[1];
        local_prop_sub_2_correct[i] = latest_experiment_results.subjects_task[j,2].blocks[i].proportion_task_correct[2];
      end
      plot(block_id, local_prop_2_correct, "g", alpha=0.1)
      if(plotting_separate_choices_on)
        # adding plotting of sub-task related results
        plot(block_id, local_prop_sub_1_correct, "c")
        plot(block_id, local_prop_sub_2_correct, "m")
      end
    end
  end

  plot(block_id+0., latest_experiment_results.task_correct[:,2], "g", linewidth=2, label="Subjects expect 50:50")
  # legend(loc=4)
  #savefig("figure_1_1.pdf", transparent="True", bbox_inches="tight", pad_inches=0.1)
  savefig("figure_biased_sim_no_bias.pdf", bbox_inches="tight", pad_inches=0.1)
end



plot_figure_1()
plot_figure_2()

# Update number of critics in plot_D_vector.jl,
#   make sure you're using all plotting options
#    then import it
# include("plot_D_vector.jl")

# Run scripted plotting of flow fields with overlaid trajectories
# include("script_check_S.jl")

# savefig("sim_trace_no_bias.pdf", bbox_inches="tight", pad_inches=0.1)

# change settings in nuber of critics
# change which data is loaded in script_check_S
# savefig("sim_trace_strong_bias.pdf", bbox_inches="tight", pad_inches=0.1)
