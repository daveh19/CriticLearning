using PyPlot;
using Distributions;
using LaTeXStrings;
using Debug;

### Useful functions
## There are a number of alternative ways to calculate pdf and cdf inverse
dist_pdf(x) = pdf(Normal(0,1), x);
dist_cdf(x) = cdf(Normal(0,1), x);
# Note: inv_cdf(x) != 1.0 / cdf(Normal(0,1), x); #Not 1/fn but inverse function!!
include("inverse_cdf.jl"); #contains invnorm(), consider switching to invphi()
invphi(p) = sqrt(2) * erfinv(2 * p - 1.0) # variance 1 inverse phi function
#invphi(p) = 2.0 * erfinv(2 * p - 1.0) # variance 2 (1+1) inverse phi function
#include("plotting_assist_functions.jl");
#include("p_space_outcome_integrator_linear.jl");
#include("d_pos_space_outcome_integrator_linear.jl");

function setup_plot_wta_D_basic_variables(local_a = 0.5, local_c = -1)
	## Plotting over D, D~ (+ve), and p optional
	global use_plot_over_D_pos = false :: Bool;
	global use_plot_over_D = false :: Bool;
	global use_plot_over_p = true :: Bool;
	# performance lines can be overlaid on plots in D-space to aid legibility
	global use_overlay_performance_on_D = true :: Bool;
	# simulation trajectories
	global use_add_trajectories_to_plot = false :: Bool;
	global sub_task_id_to_plot = 2 ::Int;
	global use_plot_measured_proportion_correct = false :: Bool;
	# forward Euler integrated trajectories
	global use_overlay_p_Euler_trajectories = false :: Bool;
	global use_overlay_D_pos_Euler_trajectories = false :: Bool;
	# separate components of flow field
	global use_include_learning_term_in_flow = true :: Bool;
	global use_include_internal_bias_term_in_flow = true :: Bool;
	global use_include_external_bias_term_in_flow = true :: Bool;

	## Space over which vector field is calculated / plotted
	global no_points = 17; #25; #30;
	#no_points = 10;
	#no_y_points = no_points - 1;
	# The no_y_points is to ensure that I plot the vector field in the right direction,
	#	 julia is column major but matplot lib is row major which causes confusion!
	#	Set no_y_points = no_points - 1; to check if an error is thrown, no error means
	#		that the array access is correct.
	global epsilon = 1e-7
	global no_y_points = no_points;
	global p = linspace(0+epsilon, 1-epsilon, no_points);
	global p_y = linspace(0+epsilon, 1-epsilon, no_y_points);
	global d_a = linspace(-3,3, no_points);
	global d_b = linspace(-3,3, no_points);

	global D_pos_scale = 100.0:: Float64;
	global D_scale = 100.0 :: Float64;
	global p_scale = 100.0:: Float64;

	#debug vars
	global Da = zeros(no_points);
	global Db = zeros(no_y_points);

	## Vector flow field variables
	global deriv_p_a = zeros(no_points, no_y_points);
	global deriv_p_b = zeros(no_points, no_y_points);
	global p_deriv_D_a = zeros(no_points, no_y_points);
	global p_deriv_D_b = zeros(no_points, no_y_points);
	global deriv_D_a = zeros(no_points, no_y_points);
	global deriv_D_b = zeros(no_points, no_y_points);
	global deriv_D_a_pos = zeros(no_points, no_y_points);
	global deriv_D_b_pos = zeros(no_points, no_y_points);


	# Confusion parameter
	global critic_dimensions = 2;
	# perfect critic (overwritten if any of the following are active)
	global C = eye(critic_dimensions)
	#=
	# equal mix critic
	c = 1 / critic_dimensions; # currently equal confusion mix of all true critics
	C = ones(critic_dimensions,critic_dimensions)
	C *= c
	A = eye(critic_dimensions) - C=#

	# Probabilistic presentation of individual tasks critic
	global internal_task_probability = ones(1,2) / 2; # relative probability of task 1 vs task 2
	global prob_task = ones(1,critic_dimensions); # only used in critic now!
	prob_task /= critic_dimensions;
	prob_task = [0.25 0.25 0.5]; #[0.5 0.5]; #[0.25 0.25 0.25 0.25]; #[0.25 0.25 0.5];
	#prob_task = [1, 0.001, 10, 10]; # manual tweaking
	#prob_task /= sum(prob_task); # normalise, so I can use arbitrary units
	# this influences Confustion matrix
	for k = 1:critic_dimensions
		C[k,:] = prob_task[1:critic_dimensions];
	end
	global A = eye(critic_dimensions) - C;
	#A = eye(critic_dimensions);

	if(local_c != -1)
		global A = eye(critic_dimensions) - local_c;
	end

	# Input representation similarity parameter
	global a = local_a; #0.5; #0.9;
	global S = [1 a; a 1]
	S /= S[1,1];

	# Output correlation with +ve D
	global O = [1; -1];

	# Noise and external bias
	global sigma = 1;
	global R_ext = 0.1; #0.95;

	#mu2 associated with less favourable output choice (be careful of this!)
	global mu2 = zeros(2);
	mu2[1] = 14; # 40Hz mu_2 for input task 1
	mu2[2] = 14; # 40Hz mu_2 for input task 1
end


function calculate_linear_wta_model_flow_vectors()
	for i = 1:no_points
		for j = 1:(no_y_points)
			#####
			#
			# Calculation of change of difference in outputs
			#
			# positive association in plotting of D with reward (use d_a as d_a^~)
			if (use_plot_over_D_pos)
				temp_a = 0;
				temp_b = 0;

				# Linear-WTA has mean output firing rates per task
				# mu2 associated with less favourable output choice (be careful of this!)
				# indexing of elements of mu refers to task (1 or 2, ie a or b)
				mu1 = zeros(2);
				mu1[1] = mu2[1] + d_a[i];
				mu1[2] = mu2[2] + d_b[j];

				if (use_include_learning_term_in_flow)
					# *2 for R^{true} = (2p-1)
					temp_a = [mu1[1]-mu2[1]] * cdf(Normal(0,sigma), (d_a[i])) + mu2[1] - ( (2 *cdf(Normal(0,sigma), (d_a[i])) - 1) * ( (mu1[1]-mu2[1]) * cdf(Normal(0,sigma), (d_a[i])) - mu2[1] ) ) :: Float64;
					temp_b = [mu1[2]-mu2[2]] * cdf(Normal(0,sigma), (d_b[j])) + mu2[2] - ( (2 *cdf(Normal(0,sigma), (d_b[j])) - 1) * ( (mu1[2]-mu2[2]) * cdf(Normal(0,sigma), (d_b[j])) - mu2[2] ) ) :: Float64;

					#print("$temp_a $temp_b\n");
					# for some reason the WTA mathematical equation is becoming an array in julia
					temp_a = temp_a[1];
					temp_b = temp_b[1];
					#print("$temp_a $temp_b\n");
				end

				if (use_include_internal_bias_term_in_flow)
					# equations for R^{true} = (2p-1)
					temp_a += ( (mu1[1] + mu2[1]) * cdf(Normal(0,sigma), (d_a[i])) - mu2[1] ) * A[1,1] * (2 * cdf(Normal(0,sigma), (d_a[i])) - 1);
					temp_a += ( (mu1[1] + mu2[1]) * cdf(Normal(0,sigma), (d_a[i])) - mu2[1] ) * A[1,2] * (2 * cdf(Normal(0,sigma), (d_b[j])) - 1);

					temp_b += ( (mu1[2] + mu2[2]) * cdf(Normal(0,sigma), (d_b[j])) - mu2[2] ) * A[2,1] * (2 * cdf(Normal(0,sigma), (d_a[i])) - 1);
					temp_b += ( (mu1[2] + mu2[2]) * cdf(Normal(0,sigma), (d_b[j])) - mu2[2] ) * A[2,2] * (2 * cdf(Normal(0,sigma), (d_b[j])) - 1);
				end

				if (use_include_external_bias_term_in_flow)
					# Bias from other tasks
					if(critic_dimensions > 2)
						for(k = 3:critic_dimensions)
							temp_a += ( (mu1[1] + mu2[1]) * cdf(Normal(0,sigma), (d_a[i])) - mu2[1] ) * (A[1,k] * R_ext);
							temp_b += ( (mu1[2] + mu2[2]) * cdf(Normal(0,sigma), (d_b[j])) - mu2[2] ) * (A[1,k] * R_ext);
						end
					end
				end

				# Multiply by probability of occurence of each task
				temp_a *= internal_task_probability[1]; #prob_task[1];
				temp_b *= internal_task_probability[2]; #prob_task[2];

				# putting it all together
				deriv_D_a_pos[i,j] = ( O[1] * S[1,1] * temp_a + O[2] * S[1,2] * temp_b );
				deriv_D_b_pos[i,j] = ( O[1] * S[2,1] * temp_a + O[2] * S[2,2] * temp_b );

				# multiply again by output encoding to give +ve D for success representation
				deriv_D_a_pos[i,j] *= O[1];
				deriv_D_b_pos[i,j] *= O[2];
			end

			######
			#
			# no correction for -ve association in plotting of D with reward (use d_a as d_a)
			if (use_plot_over_D)
				temp_a = 0;
				temp_b = 0;

				# Linear-WTA has mean output firing rates per task
				# mu2 associated with less favourable output choice (be careful of this!)
				# indexing of elements of mu refers to task (1 or 2, ie a or b)
				mu1 = zeros(2);
				mu1[1] = mu2[1] + d_a[i]*O[1];
				mu1[2] = mu2[2] + d_b[j]*O[2];

				if (use_include_learning_term_in_flow)
					# *2 for R^{true} = (2p-1)
					temp_a = [mu1[1]-mu2[1]] * cdf(Normal(0,sigma), (d_a[i]*O[1])) + mu2[1] - ( (2 *cdf(Normal(0,sigma), (d_a[i]*O[1])) - 1) * ( (mu1[1]-mu2[1]) * cdf(Normal(0,sigma), (d_a[i]*O[1])) - mu2[1] ) ) :: Float64;
					temp_b = [mu1[2]-mu2[2]] * cdf(Normal(0,sigma), (d_b[j]*O[2])) + mu2[2] - ( (2 *cdf(Normal(0,sigma), (d_b[j]*O[2])) - 1) * ( (mu1[2]-mu2[2]) * cdf(Normal(0,sigma), (d_b[j]*O[2])) - mu2[2] ) ) :: Float64;

					#print("$temp_a $temp_b\n");
					# for some reason the WTA mathematical equation is becoming an array in julia
					temp_a = temp_a[1];
					temp_b = temp_b[1];
					#print("$temp_a $temp_b\n");
				end

				if (use_include_internal_bias_term_in_flow)
					# equations for R^{true} = (2p-1)
					temp_a += ( (mu1[1] + mu2[1]) * cdf(Normal(0,sigma), (d_a[i]*O[1])) - mu2[1] ) * A[1,1] * (2 * cdf(Normal(0,sigma), (d_a[i]*O[1])) - 1);
					temp_a += ( (mu1[1] + mu2[1]) * cdf(Normal(0,sigma), (d_a[i]*O[1])) - mu2[1] ) * A[1,2] * (2 * cdf(Normal(0,sigma), (d_b[j]*O[2])) - 1);

					temp_b += ( (mu1[2] + mu2[2]) * cdf(Normal(0,sigma), (d_b[j]*O[2])) - mu2[2] ) * A[2,1] * (2 * cdf(Normal(0,sigma), (d_a[i]*O[1])) - 1);
					temp_b += ( (mu1[2] + mu2[2]) * cdf(Normal(0,sigma), (d_b[j]*O[2])) - mu2[2] ) * A[2,2] * (2 * cdf(Normal(0,sigma), (d_b[j]*O[2])) - 1);
				end

				if (use_include_external_bias_term_in_flow)
					# Bias from other tasks
					if(critic_dimensions > 2)
						for(k = 3:critic_dimensions)
							temp_a += ( (mu1[1] + mu2[1]) * cdf(Normal(0,sigma), (d_a[i]*O[1])) - mu2[1] ) * (A[1,k] * R_ext);
							temp_b += ( (mu1[2] + mu2[2]) * cdf(Normal(0,sigma), (d_b[j]*O[2])) - mu2[2] ) * (A[1,k] * R_ext);
						end
					end
				end

				# Multiply by probability of occurence of each task
				temp_a *= internal_task_probability[1]; #prob_task[1];
				temp_b *= internal_task_probability[2]; #prob_task[2];

				#debug_str = string(typeof(temp_a))
				#print("$debug_str\n");

				# putting it all together
				deriv_D_a[i,j] = ( O[1] * S[1,1] * temp_a + O[2] * S[1,2] * temp_b );
				deriv_D_b[i,j] = ( O[1] * S[2,1] * temp_a + O[2] * S[2,2] * temp_b );
			end


			#####
			#
			# Calculation of change of probability of outcome
			#
			if (use_plot_over_p)
				Da[i] = invphi(p[i]);
				Db[j] = invphi(p_y[j]);

				p_temp_a = 0;
				p_temp_b = 0;

				# Linear-WTA has mean output firing rates per task
				# mu2 associated with less favourable output choice (be careful of this!)
				# indexing of elements of mu refers to task (1 or 2, ie a or b)
				mu1 = zeros(2);
				mu1[1] = mu2[1] + Da[i];
				mu1[2] = mu2[2] + Db[j];

				if (use_include_learning_term_in_flow)
					p_temp_a = [mu1[1]-mu2[1]] * cdf(Normal(0,sigma), (Da[i])) + mu2[1] - ( (2 *cdf(Normal(0,sigma), (Da[i])) - 1) * ( (mu1[1]-mu2[1]) * cdf(Normal(0,sigma), (Da[i])) - mu2[1] ) ) :: Float64;
					p_temp_b = [mu1[2]-mu2[2]] * cdf(Normal(0,sigma), (Db[j])) + mu2[2] - ( (2 *cdf(Normal(0,sigma), (Db[j])) - 1) * ( (mu1[2]-mu2[2]) * cdf(Normal(0,sigma), (Db[j])) - mu2[2] ) ) :: Float64;

					#print("$temp_a $temp_b\n");
					# for some reason the WTA mathematical equation is becoming an array in julia
					p_temp_a = p_temp_a[1];
					p_temp_b = p_temp_b[1];
					#print("$temp_a $temp_b\n");
				end

				if (use_include_internal_bias_term_in_flow)
					# equations for R^{true} = (2p-1)
					p_temp_a += ( (mu1[1] + mu2[1]) * cdf(Normal(0,sigma), (Da[i])) - mu2[1] ) * A[1,1] * (2 * cdf(Normal(0,sigma), (Da[i])) - 1);
					p_temp_a += ( (mu1[1] + mu2[1]) * cdf(Normal(0,sigma), (Da[i])) - mu2[1] ) * A[1,2] * (2 * cdf(Normal(0,sigma), (Db[j])) - 1);

					p_temp_b += ( (mu1[2] + mu2[2]) * cdf(Normal(0,sigma), (Db[j])) - mu2[2] ) * A[2,1] * (2 * cdf(Normal(0,sigma), (Da[i])) - 1);
					p_temp_b += ( (mu1[2] + mu2[2]) * cdf(Normal(0,sigma), (Db[j])) - mu2[2] ) * A[2,2] * (2 * cdf(Normal(0,sigma), (Db[j])) - 1);
				end

				if (use_include_external_bias_term_in_flow)
					# Bias from other tasks
					if(critic_dimensions > 2)
						for(k = 3:critic_dimensions)
							p_temp_a += ( (mu1[1] + mu2[1]) * cdf(Normal(0,sigma), (Da[i])) - mu2[1] ) * (A[1,k] * R_ext);
							p_temp_b += ( (mu1[2] + mu2[2]) * cdf(Normal(0,sigma), (Db[j])) - mu2[2] ) * (A[1,k] * R_ext);

							#p_temp_a += Da[i] * (A[1,k] * R_ext);
							#p_temp_b += Db[j] * (A[2,k] * R_ext);
						end
					end
				end

				# Multiply by probability of occurence of each task
				p_temp_a *= internal_task_probability[1]; #prob_task[1];
				p_temp_b *= internal_task_probability[2]; #prob_task[2];

				# putting it all together
				p_deriv_D_a[i,j] = (O[1] * S[1,1] * p_temp_a + O[2] * S[1,2] * p_temp_b);
				p_deriv_D_b[i,j] = (O[1] * S[2,1] * p_temp_a + O[2] * S[2,2] * p_temp_b);

				# we need to transform derivatives to D_pos space
				p_deriv_D_a[i,j] *= O[1];
				p_deriv_D_b[i,j] *= O[2];

				# and we scale everything by the pdf of the underlying probability
				deriv_p_a[i,j] = pdf(Normal(0,sigma), Da[i]) * p_deriv_D_a[i,j];
				deriv_p_b[i,j] = pdf(Normal(0,sigma), Db[j]) * p_deriv_D_b[i,j];
			end
		end
	end
end # end function calculate_linear_model_flow_vectors()

function plot_linear_wta_model_flow_vectors()
	## Plotting
	print("Plotting...\n")
	filename_change = "unbounded_post"
	filename_change = "binary_new"
	#filename_change = "rescaled_new"
	file_name_change = "blah"
	filename_base = string("vector_field_", filename_change);
	filename_quiver = string("quiver_",filename_base,".pdf")
	filename_stream = string("stream_",filename_base,".pdf")

	#figure();
	#quiver(p,p,deriv_p_a', deriv_p_b');
	#quiver(p,p_y,deriv_p_a', deriv_p_b');
	#savefig(filename_quiver);

	#figure();
	#streamplot(p,p,deriv_p_a',deriv_p_b');
	#streamplot(p,p_y,deriv_p_a',deriv_p_b')
	#savefig(filename_stream);

	if (use_plot_over_D_pos)
		## Difference in positive outputs view
		figure();
		#streamplot(d_a,d_b,deriv_D_a',deriv_D_b');
		quiver(d_a, d_b, deriv_D_a_pos', deriv_D_b_pos', units="width", scale=D_pos_scale);
		xtxt = latexstring("D_1^+");
		ytxt = latexstring("D_2^+");
		xlabel(xtxt)
		ylabel(ytxt) # L"D_2"
		title("Similarity s=$a");
		if (critic_dimensions > 2)
			titletxt = latexstring();
			title("Similarity s=$a, R_ext = $R_ext, no external processes = $(critic_dimensions-2)");
		end

		if ( use_add_trajectories_to_plot )
			scalar_for_d_pos = 10.0 / sigma; # since sigma in simulation is 10 times sigma here
			scalar_for_d_pos = 350.0;
			for j = 1:no_subjects
				local_prop_sub_1_correct = zeros(no_blocks_in_experiment);
				local_prop_sub_2_correct = zeros(no_blocks_in_experiment);
				for i = 1:no_blocks_in_experiment
					#scatter(latest_experiment_results.subjects_task[j,sub_task_id].blocks[i].proportion_task_correct[1], latest_experiment_results.subjects_task[j,sub_task_id].blocks[i].proportion_task_correct[2], marker="o", c="c")
					local_prop_sub_1_correct[i] = (exp_results[1].subjects_task[j,sub_task_id_to_plot].blocks[i].noise_free_positive_output[sub_task_id_to_plot,1]) / scalar_for_d_pos;
					local_prop_sub_2_correct[i] = (exp_results[1].subjects_task[j,sub_task_id_to_plot].blocks[i].noise_free_positive_output[sub_task_id_to_plot,2]) / scalar_for_d_pos;
				end
				plot(local_prop_sub_1_correct, local_prop_sub_2_correct, "r", zorder=1)
				#print("",local_prop_sub_1_correct, local_prop_sub_2_correct, "\n-----\n")
			end
			for j = 1:no_subjects
				for i = 1:no_blocks_in_experiment
					# start point
					scatter(exp_results[1].subjects_task[j,sub_task_id_to_plot].blocks[1].noise_free_positive_output[sub_task_id_to_plot,1] / scalar_for_d_pos, exp_results[1].subjects_task[j,sub_task_id_to_plot].blocks[1].noise_free_positive_output[sub_task_id_to_plot,2] / scalar_for_d_pos, marker="s", c="r", s=40, zorder=2)
					# end point
					scatter(exp_results[1].subjects_task[j,sub_task_id_to_plot].blocks[end].noise_free_positive_output[sub_task_id_to_plot,1] /scalar_for_d_pos, exp_results[1].subjects_task[j,sub_task_id_to_plot].blocks[end].noise_free_positive_output[sub_task_id_to_plot,2] / scalar_for_d_pos, marker="D", c="g", s=60, zorder=3)
				end
			end
		end

		if ( use_overlay_performance_on_D )
			overlay_level_80 = invphi(0.8);
			overlay_level_90 = invphi(0.9);
			overlay_level_95 = invphi(0.95);
			overlay_level_99 = invphi(0.99);

			performance_overlay = ones(no_points);

			plot(performance_overlay * overlay_level_80, d_a, linewidth=2, c="c", zorder=0);
			#plot(performance_overlay * overlay_level_90, d_a, linewidth=2, c="m", zorder=0);
			plot(performance_overlay * overlay_level_95, d_a, linewidth=2, c="y", zorder=0);
			plot(performance_overlay * overlay_level_99, d_a, linewidth=2, c="g", zorder=0);

			plot(-performance_overlay * overlay_level_80, d_a, linewidth=2, c="c", zorder=0);
			#plot(-performance_overlay * overlay_level_90, d_a, linewidth=2, c="m", zorder=0);
			plot(-performance_overlay * overlay_level_95, d_a, linewidth=2, c="y", zorder=0);
			plot(-performance_overlay * overlay_level_99, d_a, linewidth=2, c="g", zorder=0);

			plot(d_a, performance_overlay * overlay_level_80, linewidth=2, c="c", zorder=0);
			#plot(d_a, performance_overlay * overlay_level_90, linewidth=2, c="m", zorder=0);
			plot(d_a, performance_overlay * overlay_level_95, linewidth=2, c="y", zorder=0);
			plot(d_a, performance_overlay * overlay_level_99, linewidth=2, c="g", zorder=0);

			plot(d_a, -performance_overlay * overlay_level_80, linewidth=2, c="c", zorder=0);
			#plot(d_a, -performance_overlay * overlay_level_90, linewidth=2, c="m", zorder=0);
			plot(d_a, -performance_overlay * overlay_level_95, linewidth=2, c="y", zorder=0);
			plot(d_a, -performance_overlay * overlay_level_99, linewidth=2, c="g", zorder=0);

			#plot(d_a, Db_null);
			## x=0 and y=0 lines for visual inspection
			origin = zeros(no_points);
			#origin_space = linspace(-100,100,no_points);
			plot(origin, d_a, linewidth=1, c="0.75", zorder=-1);
			plot(d_b, origin, linewidth=1, c="0.75", zorder=-1);
		end

		if (use_overlay_D_pos_Euler_trajectories)
			D_pos_trajectories = calculate_D_pos_trajectories();
			plot_D_pos_space_trajectories(D_pos_trajectories)
			#report_end_point_results(p_trajectories)
			axis([-5,5,-5,5])
		end
	end


	if (use_plot_over_D)
		## Difference in outputs view
		figure();
		#streamplot(d_a,d_b,deriv_D_a',deriv_D_b');
		quiver(d_a,d_b,deriv_D_a',deriv_D_b', units="width", scale=D_scale);
		xtxt = latexstring("D_1");
		ytxt = latexstring("D_2");
		xlabel(xtxt)
		ylabel(ytxt) # L"D_2"
		title("Similarity s=$a");
		if (critic_dimensions > 2)
			titletxt = latexstring();
			title("Similarity s=$a, R_ext = $R_ext, no external processes = $(critic_dimensions-2)");
		end

		if ( use_overlay_performance_on_D )
			overlay_level_80 = invphi(0.8);
			overlay_level_90 = invphi(0.9);
			overlay_level_95 = invphi(0.95);
			overlay_level_99 = invphi(0.99);

			performance_overlay = ones(no_points);

			plot(performance_overlay * overlay_level_80, d_a, linewidth=2, c="c", zorder=0);
			#plot(performance_overlay * overlay_level_90, d_a, linewidth=2, c="m", zorder=0);
			plot(performance_overlay * overlay_level_95, d_a, linewidth=2, c="y", zorder=0);
			plot(performance_overlay * overlay_level_99, d_a, linewidth=2, c="g", zorder=0);

			plot(-performance_overlay * overlay_level_80, d_a, linewidth=2, c="c", zorder=0);
			#plot(-performance_overlay * overlay_level_90, d_a, linewidth=2, c="m", zorder=0);
			plot(-performance_overlay * overlay_level_95, d_a, linewidth=2, c="y", zorder=0);
			plot(-performance_overlay * overlay_level_99, d_a, linewidth=2, c="g", zorder=0);

			plot(d_a, performance_overlay * overlay_level_80, linewidth=2, c="c", zorder=0);
			#plot(d_a, performance_overlay * overlay_level_90, linewidth=2, c="m", zorder=0);
			plot(d_a, performance_overlay * overlay_level_95, linewidth=2, c="y", zorder=0);
			plot(d_a, performance_overlay * overlay_level_99, linewidth=2, c="g", zorder=0);

			plot(d_a, -performance_overlay * overlay_level_80, linewidth=2, c="c", zorder=0);
			#plot(d_a, -performance_overlay * overlay_level_90, linewidth=2, c="m", zorder=0);
			plot(d_a, -performance_overlay * overlay_level_95, linewidth=2, c="y", zorder=0);
			plot(d_a, -performance_overlay * overlay_level_99, linewidth=2, c="g", zorder=0);

			#plot(d_a, Db_null);
			## x=0 and y=0 lines for visual inspection
			origin = zeros(no_points);
			#origin_space = linspace(-100,100,no_points);
			plot(origin, d_a, linewidth=1, c="0.75", zorder=-1);
			plot(d_b, origin, linewidth=1, c="0.75", zorder=-1);
		end
	end


	if (use_plot_over_p)
		## probabilistic view
		figure(figsize=(5,5));
		##streamplot(d_a,d_b,deriv_D_a',deriv_D_b');
		quiver(p,p_y,deriv_p_a',deriv_p_b', units="width", scale=p_scale);
		xtxt = latexstring("p_1");
		ytxt = latexstring("p_2");
		xlabel(xtxt)
		ylabel(ytxt) # L"D_2"
		aa = abs(a);
		aa = a;
		title("Similarity s=$aa");
		if (critic_dimensions > 2)
			titletxt = latexstring();
			title("Similarity s=$aa, R_ext = $R_ext, no external processes = $(critic_dimensions-2)");
		end

		if (use_overlay_p_Euler_trajectories)
			p_trajectories = calculate_p_trajectories();
			plot_p_space_trajectories(p_trajectories)
			report_end_point_results(p_trajectories)
		end
		if (use_overlay_D_pos_Euler_trajectories)
			D_pos_trajectories = calculate_D_pos_trajectories();
			plot_D_pos_space_trajectories_in_p_space(D_pos_trajectories)
			#report_end_point_results(p_trajectories)
		end
	end


	if (use_plot_over_p && use_add_trajectories_to_plot)
		if (critic_dimensions == 2)
			add_trajectories_to_linear_p_plot(exp_results[1],sub_task_id_to_plot);
		elseif (critic_dimensions == 4)
			#TODO: plotting wrong trajectories here
			add_biased_trajectories_to_linear_p_plot(exp_results[1],sub_task_id_to_plot);
		end
	end
end # end function plot_linear_model_flow_vectors()

function run_linear_wta_model_flow(local_similarity=0.7)
	setup_plot_wta_D_basic_variables(local_similarity);
	calculate_linear_wta_model_flow_vectors();
	plot_linear_wta_model_flow_vectors();
end
