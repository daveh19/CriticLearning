print("Zeroing results!!\n");
include("multi_critic_detailed_recording.jl");
using PyPlot
using PyCall
using LaTeXStrings;

@pyimport seaborn as sns
#sns.set(font_scale=1.5)
sns.set_context("poster")

#exp_results = [];



function plot_figure_1_1(results_id=1::Int)
  # uses:
  #    compare_three_trial_types_with_multiple_subjects()
  #    paper_binary_inputs_parameters_critic_simulations.jl
  # shows:
  #    learning occurs on hard task when it is performed alone

  plotting_separate_choices_on = false;

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

  plot(block_id+0., latest_experiment_results.task_correct[:,2], "g", linewidth=2, label="Subjects learning difficult task")
  legend(loc=4)
  #savefig("figure_1_1.pdf", transparent="True", bbox_inches="tight", pad_inches=0.1)
  savefig("figure_1_1.pdf", bbox_inches="tight", pad_inches=0.1)
end


function plot_figure_1_2(results_id=1::Int)
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
        scatter(i-0., latest_experiment_results.subjects_roving_task[j,1].blocks[i].proportion_correct, marker="o", edgecolors="face", c="b", alpha=0.7)
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
      plot(block_id-0., local_prop_roving_correct, "b", alpha=0.1)
    end
  end

  plot(block_id-0., latest_experiment_results.roving_correct[:,1], "b", linewidth=3, label="Block mean performance average")
  plot(block_id-0., latest_experiment_results.roving_task_correct[:,1,1], "r", linewidth=3, label="Block mean performance on Bisection Task 1")
  plot(block_id-0., latest_experiment_results.roving_task_correct[:,2,1], "g", linewidth=3, label="Block mean performance on Bisection Task 2")

  legend(loc=4)

  #savefig("figure_1_2.pdf", transparent="True", bbox_inches="tight", pad_inches=0.1)
  savefig("figure_1_2.pdf", bbox_inches="tight", pad_inches=0.1)
end


function plot_figure_2_1(results_id=1::Int)
  # uses:
  #    compare_three_trial_types_with_multiple_subjects()
  #    paper_binary_inputs_parameters_critic_simulations.jl
  # shows:
  #

  plotting_separate_choices_on = false;

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
  legend(loc=4)
  #savefig("figure_1_1.pdf", transparent="True", bbox_inches="tight", pad_inches=0.1)
  savefig("figure_2_1.pdf", bbox_inches="tight", pad_inches=0.1)
end


function plot_figure_2_2(results_id=1::Int)
  # uses:
  #    compare_three_trial_types_with_multiple_subjects()
  #    paper_binary_inputs_parameters_critic_simulations.jl
  # shows:
  #

  plotting_separate_choices_on = false;

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

  plot(block_id+0., latest_experiment_results.task_correct[:,2], "g", linewidth=2, label="Subjects expect 75:25")
  legend(loc=4)
  #savefig("figure_1_1.pdf", transparent="True", bbox_inches="tight", pad_inches=0.1)
  savefig("figure_2_2.pdf", bbox_inches="tight", pad_inches=0.1)
end


function plot_figure_3_1(results_id=1::Int)
  # uses:
  #    biased_compare_three_trial_types_with_multiple_subjects()
  #    paper_binary_inputs_parameters_critic_simulations.jl
  # shows:
  #

  plotting_separate_choices_on = false;

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
        scatter(i+0., latest_experiment_results.subjects_roving_task[j,2].blocks[i].proportion_correct, marker="o", edgecolors="face", c="g", alpha=0.5)
        if(plotting_separate_choices_on)
          # adding plotting of sub-task related results
          scatter(i+0., latest_experiment_results.subjects_roving_task[j,2].blocks[i].proportion_task_correct[1], marker="o", c="c")
          scatter(i+0., latest_experiment_results.subjects_roving_task[j,2].blocks[i].proportion_task_correct[2], marker="o", c="m")
        end
      end
    end
  end

  if(plotting_error_bars_on)
    errorbar(block_id+0., latest_experiment_results.roving_correct[:,2], latest_experiment_results.task_range[:,2], ecolor="g", color="g", linewidth=2)
    errorbar(block_id+0., latest_experiment_results.roving_correct[:,2], latest_experiment_results.task_error[:,2], ecolor="k", color="g", linewidth=2)
  end

  if(plotting_individual_subjects_on)
    for j = 1:no_subjects
      local_prop_roving_correct = zeros(no_blocks_in_experiment);
      local_prop_roving_task_1_correct = zeros(no_blocks_in_experiment);
      local_prop_roving_task_2_correct = zeros(no_blocks_in_experiment);
      for i = 1:no_blocks_in_experiment
        local_prop_roving_correct[i] = latest_experiment_results.subjects_roving_task[j,2].blocks[i].proportion_correct;
        local_prop_roving_task_1_correct[i] = latest_experiment_results.subjects_roving_task[j,2].blocks[i].proportion_task_correct[1];
        local_prop_roving_task_2_correct[i] = latest_experiment_results.subjects_roving_task[j,2].blocks[i].proportion_task_correct[2];
      end
      if(plotting_separate_choices_on)#(plotting_task_by_task_on)
        plot(block_id-0., local_prop_roving_task_1_correct, "c", alpha=0.1)
        plot(block_id-0., local_prop_roving_task_2_correct, "m", alpha=0.1)
      end
      #delete here for simple mean plot
      plot(block_id-0., local_prop_roving_correct, "g", alpha=0.1)
    end
  end

  plot(block_id-0., latest_experiment_results.roving_correct[:,2], "g", linewidth=3, label="Manually biased critic")
  legend(loc=4)
  #savefig("figure_1_1.pdf", transparent="True", bbox_inches="tight", pad_inches=0.1)
  savefig("figure_3_1.pdf", bbox_inches="tight", pad_inches=0.1)
end


function plot_figure_3_2(results_id=1::Int)
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
        scatter(i-0., latest_experiment_results.subjects_roving_task[j,1].blocks[i].proportion_correct, marker="o", edgecolors="face", c="b", alpha=0.7)
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
      plot(block_id-0., local_prop_roving_correct, "b", alpha=0.1)
    end
  end

  plot(block_id-0., latest_experiment_results.roving_correct[:,1], "b", linewidth=3, label="Block mean performance average")
  plot(block_id-0., latest_experiment_results.roving_task_correct[:,1,1], "r", linewidth=3, label="Block mean performance on Bisection Task 1")
  plot(block_id-0., latest_experiment_results.roving_task_correct[:,2,1], "g", linewidth=3, label="Block mean performance on Bisection Task 2")

  legend(loc=4)

  #savefig("figure_1_2.pdf", transparent="True", bbox_inches="tight", pad_inches=0.1)
  savefig("figure_3_2.pdf", bbox_inches="tight", pad_inches=0.1)
end


function plot_figure_3_3(results_id=1::Int)
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
        scatter(i-0., latest_experiment_results.subjects_roving_task[j,1].blocks[i].proportion_correct, marker="o", edgecolors="face", c="b", alpha=0.7)
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
      plot(block_id-0., local_prop_roving_correct, "b", alpha=0.1)
    end
  end

  plot(block_id-0., latest_experiment_results.roving_correct[:,1], "b", linewidth=3, label="Block mean performance average")
  plot(block_id-0., latest_experiment_results.roving_task_correct[:,1,1], "r", linewidth=3, label="Block mean performance on Bisection Task 1")
  plot(block_id-0., latest_experiment_results.roving_task_correct[:,2,1], "g", linewidth=3, label="Block mean performance on Bisection Task 2")

  legend(loc=4)

  #savefig("figure_1_2.pdf", transparent="True", bbox_inches="tight", pad_inches=0.1)
  savefig("figure_3_3.pdf", bbox_inches="tight", pad_inches=0.1)
end


function plot_figure_3_4(results_id=1::Int)
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
        scatter(i-0., latest_experiment_results.subjects_roving_task[j,1].blocks[i].proportion_correct, marker="o", edgecolors="face", c="b", alpha=0.7)
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
      plot(block_id-0., local_prop_roving_correct, "b", alpha=0.1)
    end
  end

  plot(block_id-0., latest_experiment_results.roving_correct[:,1], "b", linewidth=3, label="Block mean performance on Roved tasks")
  plot(block_id-0., latest_experiment_results.roving_task_correct[:,1,1], "r", linewidth=3, label="Block mean performance on Bisection Task 1")
  plot(block_id-0., latest_experiment_results.roving_task_correct[:,2,1], "g", linewidth=3, label="Block mean performance on Bisection Task 2")

  legend(loc=4)

  #savefig("figure_1_2.pdf", transparent="True", bbox_inches="tight", pad_inches=0.1)
  savefig("figure_3_4.pdf", bbox_inches="tight", pad_inches=0.1)
end



function paper_plots_init()
  print("Running comparison simulation\n")
  compare_three_trial_types_with_multiple_subjects();
end

function paper_biased_plots_init()
  print("Running bias comparison simulation\n")
  biased_compare_three_trial_types_with_multiple_subjects();
end

#__init__ = paper_plots_init()
