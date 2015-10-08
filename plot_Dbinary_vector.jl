using PyPlot;
using Distributions;
using LaTeXStrings;

### Useful functions
## There are a number of alternative ways to calculate pdf and cdf inverse
dist_pdf(x) = pdf(Normal(0,1), x);
dist_cdf(x) = cdf(Normal(0,1), x);
# Note: inv_cdf(x) != 1.0 / cdf(Normal(0,1), x); #Not 1/fn but inverse function!!
include("inverse_cdf.jl"); #contains invnorm(), consider switching to invphi()
invphi(p) = sqrt(2) * erfinv(2 * p - 1.0)


## Plotting over D, D~ (+ve), and p optional
use_plot_over_D_pos = true :: Bool;
use_plot_over_D = false :: Bool;
use_plot_over_p = true :: Bool;
use_add_trajectories_to_plot = false :: Bool;


## Space over which vector field is calculated / plotted
no_points = 20;
#no_points = 10;
#no_y_points = no_points - 1;
# The no_y_points is to ensure that I plot the vector field in the right direction,
#	 julia is column major but matplot lib is row major which causes confusion!
#	Set no_y_points = no_points - 1; to check if an error is thrown, no error means
#		that the array access is correct.
no_y_points = no_points;
p = linspace(0, 1, no_points);
p_y = linspace(0, 1, no_y_points);
d_a = linspace(-3,3, no_points);
d_b = linspace(-3,3, no_points);

D_pos_scale = 20.0 :: Float64;
D_scale = 20.0 :: Float64;
p_scale = 1.0 :: Float64;

#debug vars
Da = zeros(no_points);
Db = zeros(no_y_points);

## Vector flow field variables
deriv_p_a = zeros(no_points, no_y_points);
deriv_p_b = zeros(no_points, no_y_points);
p_deriv_D_a = zeros(no_points, no_y_points);
p_deriv_D_b = zeros(no_points, no_y_points);
deriv_D_a = zeros(no_points, no_y_points);
deriv_D_b = zeros(no_points, no_y_points);
deriv_D_a_pos = zeros(no_points, no_y_points);
deriv_D_b_pos = zeros(no_points, no_y_points);


# Confusion parameter
critic_dimensions = 2;
# perfect critic (overwritten if any of the following are active)
C = eye(critic_dimensions)
#=
# equal mix critic
c = 1 / critic_dimensions; # currently equal confusion mix of all true critics
C = ones(critic_dimensions,critic_dimensions)
C *= c
A = eye(critic_dimensions) - C=#


# Probabilistic presentation of individual tasks
prob_task = ones(1,critic_dimensions);
prob_task /= critic_dimensions;
#prob_task = [1, 0.001, 10, 10]; # manual tweaking
#prob_task /= sum(prob_task); # normalise, so I can use arbitrary units
# this influences Confustion matrix
for k = 1:critic_dimensions
	C[k,:] = prob_task;
end
A = eye(critic_dimensions) - C;


# Input representation similarity parameter
a = 0.9;
S = [1 a; a 1]

# Output correlation with +ve D
O = [1; -1];

# Noise and external bias
sigma = 1;
R_ext = 1;


for i = 1:no_points
	for j = 1:(no_y_points)
		#####
		#
		# Calculation of change of difference in outputs
		#
		# positive association in plotting of D with reward (use d_a as d_a^~)
		if (use_plot_over_D_pos)
			# *2 for R^{true} = (2p-1)
			temp_a = 4 * cdf(Normal(0,sigma),d_a[i]) * (1 - cdf(Normal(0,sigma),d_a[i]))
			temp_b = 4 * cdf(Normal(0,sigma),d_b[j]) * (1 - cdf(Normal(0,sigma),d_b[j]))
			# equations for R^{true} = (2p-1)
			temp_a += A[1,1] * (2 * cdf(Normal(0,sigma), d_a[i]) - 1) * (2 * cdf(Normal(0,sigma),d_a[i]) - 1);
			temp_a += A[1,2] * (2 * cdf(Normal(0,sigma), d_b[j]) - 1) * (2 * cdf(Normal(0,sigma),d_a[i]) - 1);

			temp_b += A[2,1] * (2 * cdf(Normal(0,sigma), d_a[i]) - 1) * (2 * cdf(Normal(0,sigma),d_b[j]) - 1);
			temp_b += A[2,2] * (2 * cdf(Normal(0,sigma), d_b[j]) - 1) * (2 * cdf(Normal(0,sigma),d_b[j]) - 1);

			# Bias from other tasks
			if(critic_dimensions > 2)
				# a_multiplier assumes equal for all
				a_multiplier = (critic_dimensions - 2) / critic_dimensions
				#=temp_a += (2 * cdf(Normal(0,sigma),d_a[i]) - 1) * (-0.5 * R_ext);
				temp_b += (2 * cdf(Normal(0,sigma),d_b[j]) - 1) * (-0.5 * R_ext);=#
				#=temp_a += (2 * cdf(Normal(0,sigma),d_a[i]) - 1) * (-a_multiplier * R_ext);
				temp_b += (2 * cdf(Normal(0,sigma),d_b[j]) - 1) * (-a_multiplier * R_ext);=#
				for(k = 3:critic_dimensions)
					temp_a += (2 * cdf(Normal(0,sigma),d_a[i]) - 1) * (A[1,k] * R_ext);
					temp_b += (2 * cdf(Normal(0,sigma),d_b[j]) - 1) * (A[2,k] * R_ext);
				end
			end

			#TODO: Multiply by probability of occurence of each task
			#

			# putting it all together
			deriv_D_a[i,j] = S[1,1] * temp_a + S[1,2] * temp_b;
			deriv_D_b[i,j] = S[2,1] * temp_a + S[2,2] * temp_b;

			#TODO: multiply again by output encoding to give +ve D for success representation
			#deriv_D_a_pos[i,j] *= O[1];
			#deriv_D_b_pos[i,j] *= O[2];
		end

		######
		#
		# no correction for -ve association in plotting of D with reward (use d_a as d_a)
		if (use_plot_over_D)
			#TODO

		end


		#####
		#
		# Calculation of change of probability of outcome
		#
		if (use_plot_over_p)
			## the following has not been updates to follow binary output rules yet!!
			Da[i] = invphi(p[i]);
			Db[j] = invphi(p_y[j]);
			p_temp_a = 4 * cdf(Normal(0,sigma),Da[i]) * (1 - cdf(Normal(0,sigma),Da[i]))
			p_temp_b = 4 * cdf(Normal(0,sigma),Db[j]) * (1 - cdf(Normal(0,sigma),Db[j]))
			# equations for R^{true} = (2p-1)
			p_temp_a += A[1,1] * (2 * p[i] - 1) * (2 * p[i] - 1);
			p_temp_a += A[1,2] * (2 * p[j] - 1) * (2 * p[i] - 1);

			p_temp_b += A[2,1] * (2 * p[i] - 1) * (2 * p[j] - 1);
			p_temp_b += A[2,2] * (2 * p[j] - 1) * (2 * p[j] - 1);


			# Bias from other tasks
			if(critic_dimensions > 2)
				a_multiplier = (critic_dimensions - 2) / critic_dimensions
				#=p_temp_a += (2 * p[i] - 1) * (-0.5 * R_ext);
				p_temp_b += (2 * p[j] - 1) * (-0.5 * R_ext);=#
				#=p_temp_a += (2 * p[i] - 1) * (-a_multiplier * R_ext);
				p_temp_b += (2 * p[j] - 1) * (-a_multiplier * R_ext);=#
				for(k = 3:critic_dimensions)
					p_temp_a += (2 * p[i] - 1) * (A[1,k] * R_ext);
					p_temp_b += (2 * p[j] - 1) * (A[2,k] * R_ext);
				end
			end

			# putting it all together
			p_deriv_D_a[i,j] = S[1,1] * p_temp_a + S[1,2] * p_temp_b;
			p_deriv_D_b[i,j] = S[2,1] * p_temp_a + S[2,2] * p_temp_b;

			deriv_p_a[i,j] = pdf(Normal(0,sigma), Da[i]) * p_deriv_D_a[i,j];
			deriv_p_b[i,j] = pdf(Normal(0,sigma), Db[j]) * p_deriv_D_b[i,j];
		end
	end
end

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
	#TODO
end


if (use_plot_over_D)
	## Difference in outputs view
	figure();
	#streamplot(d_a,d_b,deriv_D_a',deriv_D_b');
	quiver(d_a,d_b,deriv_D_a',deriv_D_b', units="width", scale=20.0);
	xtxt = latexstring("D_1");
	ytxt = latexstring("D_2");
	xlabel(xtxt)
	ylabel(ytxt) # L"D_2"
	title("Similarity s=$a");
	if (critic_dimensions > 2)
		titletxt = latexstring();
		title("Similarity s=$a, R_ext = $R_ext, no external processes = $(critic_dimensions-2)");
	end

	#plot(d_a, Db_null);
	## x=0 and y=0 lines for visual inspection
	#=origin = zeros(no_points);
	origin_space = linspace(-100,100,no_points);
	plot(origin, origin_space);
	plot(origin_space, origin);=#
end


if (use_plot_over_p)
	## probabilistic view
	figure();
	##streamplot(d_a,d_b,deriv_D_a',deriv_D_b');
	quiver(p,p_y,deriv_p_a',deriv_p_b', units="width", scale=1.0);
	xtxt = latexstring("p_1");
	ytxt = latexstring("p_2");
	xlabel(xtxt)
	ylabel(ytxt) # L"D_2"
	aa = abs(a);
	title("Similarity s=$aa");
	if (critic_dimensions > 2)
		titletxt = latexstring();
		title("Similarity s=$aa, R_ext = $R_ext, no external processes = $(critic_dimensions-2)");
	end
end


function add_trajectories_to_linear_p_plot(latest_experiment_results, sub_task_id)
	include("parameters_critic_simulations.jl"); # dont' change the paramters in between calls!


	for j = 1:no_subjects
		local_prop_sub_1_correct = zeros(no_blocks_in_experiment);
		local_prop_sub_2_correct = zeros(no_blocks_in_experiment);
		for i = 1:no_blocks_in_experiment
			#scatter(latest_experiment_results.subjects_task[j,sub_task_id].blocks[i].proportion_task_correct[1], latest_experiment_results.subjects_task[j,sub_task_id].blocks[i].proportion_task_correct[2], marker="o", c="c")
			local_prop_sub_1_correct[i] = latest_experiment_results.subjects_task[j,sub_task_id].blocks[i].proportion_task_correct[1];
			local_prop_sub_2_correct[i] = latest_experiment_results.subjects_task[j,sub_task_id].blocks[i].proportion_task_correct[2];
		end
		plot(local_prop_sub_1_correct, local_prop_sub_2_correct, "r")
		#print("",local_prop_sub_1_correct, local_prop_sub_2_correct, "\n-----\n")
	end

	for j = 1:no_subjects
		for i = 1:no_blocks_in_experiment
			# start point
			scatter(latest_experiment_results.subjects_task[j,sub_task_id].blocks[1].proportion_task_correct[1], latest_experiment_results.subjects_task[j,sub_task_id].blocks[1].proportion_task_correct[2], marker="s", c="r", s=40)
			# end point
			scatter(latest_experiment_results.subjects_task[j,sub_task_id].blocks[end].proportion_task_correct[1], latest_experiment_results.subjects_task[j,sub_task_id].blocks[end].proportion_task_correct[2], marker="D", c="g", s=60)
		end
	end

	axis([-0.005,1.005,-0.005,1.005]);
end


if (use_plot_over_p && use_add_trajectories_to_plot)
	sub_task_id_to_plot = 1;
	add_trajectories_to_linear_p_plot(exp_results[1],sub_task_id_to_plot);
end
