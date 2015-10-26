## load experimental candidates and calculate their similarity matrices
# usage: Run a simulation of the roving experiment first.
#   then run this script
#   if S is commented out in plot_D_vector.jl then it will plot the
#   accurate flow fields for each simulation in turn

task_id = 2;
exp_id = 2;
use_linear_out = false :: Bool;
use_roving_subjects = false :: Bool;

# set critic_dimensions and
# comment out S and
# disable trajectories and
# choose measured vs perfect probabilites
#	 in the plotting file!

include("plotting_assist_functions.jl");

for i=1:no_subjects
	if (!use_roving_subjects)
  	restore_subject(exp_results[1].subjects_task[i, task_id], false);
	else
		print("Roving subject\n");
		task_id = exp_id;
		restore_subject(exp_results[1].subjects_roving_task[i, exp_id], false);
	end
  pos = sum( pre(1.0,task_id,linear_tc()) .* pre(1.0, task_id, linear_tc()) );
  neg = sum( pre(-1.0,task_id,linear_tc()) .* pre(-1.0, task_id, linear_tc()) );
  pn = sum( pre(1.0,task_id,linear_tc()) .* pre(-1.0, task_id, linear_tc()) );

	print("$pos $neg $pn\n")
	global a = pn;
  global S = [pos pn; pn neg];
	print("",S,"\n")
  S /= S[1,1];
  #S[1,:] /= S[1,1];
  #S[2,:] /= S[2,2];

  print("$i: \n", S , "\n");
  if(use_linear_out)
    include("plot_D_vector.jl")
  else
    include("plot_Dbinary_vector.jl")
  end
	if (!use_roving_subjects)
  	add_specific_trajectory_to_linear_p_plot(exp_results[1],task_id, i);
	else
		add_specific_roving_trajectory_to_linear_p_plot(exp_results[1],exp_id, i);
	end
end
