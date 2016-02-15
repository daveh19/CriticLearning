##
# Functions for plotting trajectories on flow fields


function add_trajectories_to_linear_p_plot(latest_experiment_results, sub_task_id)
	include("parameters_critic_simulations.jl"); # don't change the paramters in between calls!

	for j = 1:no_subjects
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

	axis([-0.02,1.02,-0.02,1.02]);
end


function add_biased_trajectories_to_linear_p_plot(latest_experiment_results, sub_task_id)
	include("parameters_critic_simulations.jl"); # don't change the paramters in between calls!

	for j = 1:no_subjects
		local_prop_sub_1_correct = zeros(no_blocks_in_experiment);
		local_prop_sub_2_correct = zeros(no_blocks_in_experiment);
		for i = 1:no_blocks_in_experiment
			#scatter(latest_experiment_results.subjects_task[j,sub_task_id].blocks[i].proportion_task_correct[1], latest_experiment_results.subjects_task[j,sub_task_id].blocks[i].proportion_task_correct[2], marker="o", c="c")
			if(use_plot_measured_proportion_correct)
				local_prop_sub_1_correct[i] = latest_experiment_results.subjects_roving_task[j,sub_task_id].blocks[i].proportion_task_correct[1];
				local_prop_sub_2_correct[i] = latest_experiment_results.subjects_roving_task[j,sub_task_id].blocks[i].proportion_task_correct[2];
			else
				local_prop_sub_1_correct[i] = latest_experiment_results.subjects_roving_task[j,sub_task_id].blocks[i].probability_correct[sub_task_id,1];
				local_prop_sub_2_correct[i] = latest_experiment_results.subjects_roving_task[j,sub_task_id].blocks[i].probability_correct[sub_task_id,2];
			end
		end
		plot(local_prop_sub_1_correct, local_prop_sub_2_correct, "r", zorder=1)
		#print("",local_prop_sub_1_correct, local_prop_sub_2_correct, "\n-----\n")
	end
	for j = 1:no_subjects
		for i = 1:no_blocks_in_experiment
			if(use_plot_measured_proportion_correct)
				# start point
				scatter(latest_experiment_results.subjects_roving_task[j,sub_task_id].blocks[1].proportion_task_correct[1], latest_experiment_results.subjects_roving_task[j,sub_task_id].blocks[1].proportion_task_correct[2], marker="s", c="r", s=40, zorder=2)
				# end point
				scatter(latest_experiment_results.subjects_roving_task[j,sub_task_id].blocks[end].proportion_task_correct[1], latest_experiment_results.subjects_roving_task[j,sub_task_id].blocks[end].proportion_task_correct[2], marker="D", c="g", s=60, zorder=3)
			else
				# start point
				scatter(latest_experiment_results.subjects_roving_task[j,sub_task_id].blocks[1].probability_correct[sub_task_id,1], latest_experiment_results.subjects_roving_task[j,sub_task_id].blocks[1].probability_correct[sub_task_id,2], marker="s", c="r", s=40, zorder=2)
				# end point
				scatter(latest_experiment_results.subjects_roving_task[j,sub_task_id].blocks[end].probability_correct[sub_task_id,1], latest_experiment_results.subjects_roving_task[j,sub_task_id].blocks[end].probability_correct[sub_task_id,2], marker="D", c="g", s=60, zorder=3)
			end
		end
	end

	axis([-0.02,1.02,-0.02,1.02]);
end


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

    axis([-0.02,1.02,-0.02,1.02]);
end


function add_specific_roving_trajectory_to_linear_p_plot(latest_experiment_results, sub_task_id, subject_id)
    include("parameters_critic_simulations.jl"); # don't change the paramters in between calls!

    for j = subject_id
        local_prop_sub_1_correct = zeros(no_blocks_in_experiment);
        local_prop_sub_2_correct = zeros(no_blocks_in_experiment);
        for i = 1:no_blocks_in_experiment
            #scatter(latest_experiment_results.subjects_task[j,sub_task_id].blocks[i].proportion_task_correct[1], latest_experiment_results.subjects_task[j,sub_task_id].blocks[i].proportion_task_correct[2], marker="o", c="c")
            if(use_plot_measured_proportion_correct)
                local_prop_sub_1_correct[i] = latest_experiment_results.subjects_roving_task[j,sub_task_id].blocks[i].proportion_task_correct[1];
                local_prop_sub_2_correct[i] = latest_experiment_results.subjects_roving_task[j,sub_task_id].blocks[i].proportion_task_correct[2];
            else
                local_prop_sub_1_correct[i] = latest_experiment_results.subjects_roving_task[j,sub_task_id].blocks[i].probability_correct[sub_task_id,1];
                local_prop_sub_2_correct[i] = latest_experiment_results.subjects_roving_task[j,sub_task_id].blocks[i].probability_correct[sub_task_id,2];
            end
        end
        plot(local_prop_sub_1_correct, local_prop_sub_2_correct, "r", zorder=1)
        #print("",local_prop_sub_1_correct, local_prop_sub_2_correct, "\n-----\n")
    end
    for j = 1:no_subjects
        for i = 1:no_blocks_in_experiment
            if(use_plot_measured_proportion_correct)
                # start point
                scatter(latest_experiment_results.subjects_roving_task[j,sub_task_id].blocks[1].proportion_task_correct[1], latest_experiment_results.subjects_roving_task[j,sub_task_id].blocks[1].proportion_task_correct[2], marker="s", c="r", s=40, zorder=2)
                # end point
                scatter(latest_experiment_results.subjects_roving_task[j,sub_task_id].blocks[end].proportion_task_correct[1], latest_experiment_results.subjects_roving_task[j,sub_task_id].blocks[end].proportion_task_correct[2], marker="D", c="g", s=60, zorder=3)
            else
                # start point
                scatter(latest_experiment_results.subjects_roving_task[j,sub_task_id].blocks[1].probability_correct[sub_task_id,1], latest_experiment_results.subjects_roving_task[j,sub_task_id].blocks[1].probability_correct[sub_task_id,2], marker="s", c="r", s=40, zorder=2)
                # end point
                scatter(latest_experiment_results.subjects_roving_task[j,sub_task_id].blocks[end].probability_correct[sub_task_id,1], latest_experiment_results.subjects_roving_task[j,sub_task_id].blocks[end].probability_correct[sub_task_id,2], marker="D", c="g", s=60, zorder=3)
            end
        end
    end

    axis([-0.02,1.02,-0.02,1.02]);
end


#= function add_trajectories_to_linear_p_plot(latest_experiment_results, sub_task_id)
	include("parameters_critic_simulations.jl"); # don't change the paramters in between calls!

	for j = 1:no_subjects
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
end =#
