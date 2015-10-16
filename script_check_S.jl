## load exprimental candidates and calculate their similarity matrices
# usage: Run a simulation of the roving experiment first.
#   then run this script
#   if S is commented out in plot_D_vector.jl then it will plot the
#   accurate flow fields for each simulation in turn

task_id = 2;
use_linear_out = false :: Bool;

function add_specific_trajectory_to_linear_p_plot(latest_experiment_results, sub_task_id, subject_id)
	include("parameters_critic_simulations.jl"); # don't change the paramters in between calls!

	for j = subject_id
		local_prop_sub_1_correct = zeros(no_blocks_in_experiment);
		local_prop_sub_2_correct = zeros(no_blocks_in_experiment);
		for i = 1:no_blocks_in_experiment
			#scatter(latest_experiment_results.subjects_task[j,sub_task_id].blocks[i].proportion_task_correct[1], latest_experiment_results.subjects_task[j,sub_task_id].blocks[i].proportion_task_correct[2], marker="o", c="c")
			if(use_plot_measured_proportion_correct)
				local_prop_sub_1_correct[i] = latest_experiment_results.subjects_task[j,sub_task_id].blocks[i].proportion_task_correct[1];
				local_prop_sub_2_correct[i] = latest_experiment_results.subjects_task[j,sub_task_id].blocks[i].proportion_task_correct[2];
			else
				local_prop_sub_1_correct[i] = latest_experiment_results.subjects_task[j,sub_task_id].blocks[i].probability_correct[sub_task_id,1];
				local_prop_sub_2_correct[i] = latest_experiment_results.subjects_task[j,sub_task_id].blocks[i].probability_correct[sub_task_id,2];
			end
		end
		plot(local_prop_sub_1_correct, local_prop_sub_2_correct, "r", zorder=1)
		#print("",local_prop_sub_1_correct, local_prop_sub_2_correct, "\n-----\n")
	end
	for j = 1:no_subjects
		for i = 1:no_blocks_in_experiment
			if(use_plot_measured_proportion_correct)
				# start point
				scatter(latest_experiment_results.subjects_task[j,sub_task_id].blocks[1].proportion_task_correct[1], latest_experiment_results.subjects_task[j,sub_task_id].blocks[1].proportion_task_correct[2], marker="s", c="r", s=40, zorder=2)
				# end point
				scatter(latest_experiment_results.subjects_task[j,sub_task_id].blocks[end].proportion_task_correct[1], latest_experiment_results.subjects_task[j,sub_task_id].blocks[end].proportion_task_correct[2], marker="D", c="g", s=60, zorder=3)
			else
				# start point
				scatter(latest_experiment_results.subjects_task[j,sub_task_id].blocks[1].probability_correct[sub_task_id,1], latest_experiment_results.subjects_task[j,sub_task_id].blocks[1].probability_correct[sub_task_id,2], marker="s", c="r", s=40, zorder=2)
				# end point
				scatter(latest_experiment_results.subjects_task[j,sub_task_id].blocks[end].probability_correct[sub_task_id,1], latest_experiment_results.subjects_task[j,sub_task_id].blocks[end].probability_correct[sub_task_id,2], marker="D", c="g", s=60, zorder=3)
			end
		end
	end

	axis([-0.005,1.005,-0.005,1.005]);
end


for i=1:no_subjects
  restore_subject(exp_results[1].subjects_task[i,task_id], false);
  pos = sum( pre(1.0,task_id,linear_tc()) .* pre(1.0, task_id, linear_tc()) );
  neg = sum( pre(-1.0,task_id,linear_tc()) .* pre(-1.0, task_id, linear_tc()) );
  pn = sum( pre(1.0,task_id,linear_tc()) .* pre(-1.0, task_id, linear_tc()) );

  S = [pos pn; pn neg];
  S /= S[1,1];
  #S[1,:] /= S[1,1];
  #S[2,:] /= S[2,2];

  print("$i: \n", S , "\n");
  if(use_linear_out)
    include("plot_D_vector.jl")
  else
    include("plot_Dbinary_vector.jl")
  end
  add_specific_trajectory_to_linear_p_plot(exp_results[1],task_id, i);
end
