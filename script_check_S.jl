## load experimental candidates and calculate their similarity matrices
# usage: Run a simulation of the roving experiment first.
#   then run this script
#   if S is commented out in plot_D_vector.jl then it will plot the
#   accurate flow fields for each simulation in turn

task_id = 2;
exp_id = 1;
use_linear_out = true :: Bool;
use_roving_subjects = true :: Bool;

result_set_id = 2 :: Int;

# set critic_dimensions and
# comment out S and
# disable trajectories and
# choose measured vs perfect probabilites
#	 in the plotting file!

include("plotting_assist_functions.jl");

for i=1:no_subjects
	if (!use_roving_subjects)
  	restore_subject(exp_results[result_set_id].subjects_task[i, task_id], false);
	else
		print("Roving subject\n");
		task_id = exp_id;
		restore_subject(exp_results[result_set_id].subjects_roving_task[i, exp_id], false);
	end
  pos = sum( pre(1.0,task_id,linear_tc()) .* pre(1.0, task_id, linear_tc()) );
  neg = sum( pre(-1.0,task_id,linear_tc()) .* pre(-1.0, task_id, linear_tc()) );
  pn = sum( pre(1.0,task_id,linear_tc()) .* pre(-1.0, task_id, linear_tc()) );

	if(use_linear_out)
    include("plot_D_vector.jl");
		setup_plot_D_basic_variables();
	end

#=	if(use_roving_subjects)
		global critic_dimensions = 4;
		# perfect critic (overwritten if any of the following are active)
		global C = eye(critic_dimensions);
		# Probabilistic presentation of individual tasks critic
		global prob_task = ones(1,critic_dimensions);
		prob_task /= critic_dimensions;
		global A = eye(critic_dimensions) - C;

	end=#
	print("$pos $neg $pn\n")
	global a = neg;
  global S = [neg pn; pn pos];
	print("",S,"\n")
  S /= S[1,1];
	a = S[1,2];
  #S[1,:] /= S[1,1];
  #S[2,:] /= S[2,2];

  print("$i: \n", S , "\n");
  if(use_linear_out)
    #include("plot_D_vector.jl");
		#setup_plot_D_basic_variables();
		global use_overlay_p_Euler_trajectories = false;
		calculate_linear_model_flow_vectors()
		plot_linear_model_flow_vectors()
	else
    include("plot_Dbinary_vector.jl")
  end
	if (!use_roving_subjects)
  	add_specific_trajectory_to_linear_p_plot(exp_results[result_set_id],task_id, i);
	else
		add_specific_roving_trajectory_to_linear_p_plot(exp_results[result_set_id],exp_id, i);
	end
end
