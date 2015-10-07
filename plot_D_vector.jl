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


## Space over which vector field is calculated / plotted
no_points = 30;
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


## Reward learning component to flow is an option in probabilistic diagram
plot_unbiased_learning = true :: Bool;
# Success signal is necessary for unbiased learning component
# constant Success signal
#S_1 = +1.0 / 10.0;
#S_2 = -1.0 / 10.0;
# linear decay of Success signal
#S_1 = 1.0 - p;
#S_2 = -p - 1.0;
# Stationary Success signal
S_1 = 1.0 - (2*p-1.0);
S_2 = -1.0 - (2*p-1.0);

## Rho represents error in estimation of Success signal, used in probabilistic calculation
rho_a = zeros(no_points,no_y_points);
for i = 1:no_points
	for j = 1:(no_y_points)
		rho_a[i,j] = (p[i] - p[j]);
		#rho_a[i,j] = p[i] - ((p[i] + p[j] + 2.0 * p_ext) / 4.0);
		#rho_a[i,j] = p[i] - ((p[j] + p_ext) / 2.0);
	end
end
rho_b = zeros(no_points,no_y_points);
for i = 1:no_points
	for j = 1:(no_y_points)
		rho_b[i,j] = (p[j] - p[i]);
		#rho_b[i,j] = p[j] - ((p[i] + p[j] + 2.0 * p_ext) / 4.0);
		#rho_b[i,j] = p[j] - ((p[i] + p_ext) / 2.0);
	end
end


## Constants
xa_norm_sq = 1.0;
p_ext = 0.30;

# Confusion parameter
critic_dimensions = 2;
c = 1 / critic_dimensions; # currently equal confusion mix of all true critics
C = ones(critic_dimensions,critic_dimensions)
C *= c
#C = eye(critic_dimensions) # perfect critic
A = eye(critic_dimensions) - C

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
a = 0; #-0.9;
S = [1 a; a 1]

# Output correlation with +ve D
O = [1; -1];

# Noise and external bias
sigma = 1; #sqrt(1); #sqrt(100);
#rho_ext = -0.5;
R_ext = 1; #20; #1.001;


for i = 1:no_points
	for j = 1:(no_y_points)
		#####
		#
		# Calculation of change of difference in outputs
		#
		# for R^{true} = p
		# temp_a = sigma^2 * pdf(Normal(0,sigma), d_a[i]);
		# temp_b = sigma^2 * pdf(Normal(0,sigma), d_b[j]);
		# # equations for R^{true} = p
		# temp_a += A[1,1] * cdf(Normal(0,sigma), d_a[i]) * d_a[i];
		# temp_a += A[1,2] * cdf(Normal(0,sigma), d_b[j]) * d_a[i];
		# temp_b += A[2,1] * cdf(Normal(0,sigma), d_a[i]) * d_b[j];
		# temp_b += A[2,2] * cdf(Normal(0,sigma), d_b[j]) * d_b[j];

		# *2 for R^{true} = (2p-1)
		temp_a = sigma^2 * pdf(Normal(0,sigma), (d_a[i])) * 2;
		temp_b = sigma^2 * pdf(Normal(0,sigma), (d_b[j])) * 2;
		# equations for R^{true} = (2p-1)
		temp_a += A[1,1] * (2 * cdf(Normal(0,sigma), (d_a[i])) - 1) * (d_a[i]);
		temp_a += A[1,2] * (2 * cdf(Normal(0,sigma), (d_b[j])) - 1) * (d_a[i]);

		temp_b += A[2,1] * (2 * cdf(Normal(0,sigma), (d_a[i])) - 1) * (d_b[j]);
		temp_b += A[2,2] * (2 * cdf(Normal(0,sigma), (d_b[j])) - 1) * (d_b[j]);

		# Bias from other tasks
		if(critic_dimensions > 2)
			# a_multiplier assumes equal for all
			a_multiplier = (critic_dimensions - 2) / critic_dimensions
			#=temp_a += d_a[i] * (-0.5 * R_ext);
			temp_b += d_b[j] * (-0.5 * R_ext);=#
			#=temp_a += d_a[i] * (-a_multiplier * R_ext);
			temp_b += d_b[j] * (-a_multiplier * R_ext);=#
			for(k = 3:critic_dimensions)
				temp_a += d_a[i] * (A[1,k] * R_ext);
				temp_b += d_b[j] * (A[2,k] * R_ext);
			end
		end

		# Multiply by probability of occurence of each task
		temp_a *= prob_task[1];
		temp_b *= prob_task[2];

		# putting it all together
		deriv_D_a[i,j] = ( O[1] * S[1,1] * temp_a + O[2] * S[1,2] * temp_b ) * O[1];
		deriv_D_b[i,j] = ( O[1] * S[2,1] * temp_a + O[2] * S[2,2] * temp_b ) * O[2];

		# multiply again by output encoding to give +ve D for success representation
		#deriv_D_a[i,j] *= O[1];
		#deriv_D_b[i,j] *= O[2];

		#####
		#
		# Calculation of change of probability of outcome
		#
		# -ve for task B is for the opposite sign on rho
		# unbounded post firing rate equations
		#=deriv_p_a[i,j] = dist_pdf(invnorm(p[i])) * xa_norm_sq * rho[i,j] * invnorm(p[i]);
		deriv_p_b[i,j] = dist_pdf(invnorm(p[j])) * xa_norm_sq * (-rho[i,j]) * invnorm(p[j]);=#
		# binary output neurons
		#=deriv_p_a[i,j] = dist_pdf(invnorm(p[i])) * xa_norm_sq * rho[i,j] * p[i];
		deriv_p_b[i,j] = - dist_pdf(invnorm(p[j])) * xa_norm_sq * rho[i,j] * p[j];=#
		# binary output neurons - alternative
		#=deriv_p_a[i,j] = dist_pdf(invnorm(p[i])) * xa_norm_sq * rho[i,j] * p[i] * (1-p[i]);
		deriv_p_b[i,j] = - dist_pdf(invnorm(p[j])) * xa_norm_sq * rho[i,j] * p[j] * (1-p[j]);=#

		# binary output neurons - new (in first write-up)
		# effects of unsupervised bias
		#deriv_p_a[i,j] = dist_pdf(invnorm(p[i])) * xa_norm_sq * rho_a[i,j] * (2 * p[i] - 1);
		#deriv_p_b[i,j] = dist_pdf(invnorm(p[j])) * xa_norm_sq * rho_b[i,j] * (2 * p[j] - 1);


		Da[i] = invphi(p[i]);
		Db[j] = invphi(p_y[j]);
		p_temp_a = sigma^2 * pdf(Normal(0,sigma), Da[i]) * 2;
		p_temp_b = sigma^2 * pdf(Normal(0,sigma), Db[j]) * 2;
		# equations for R^{true} = (2p-1)
		p_temp_a += A[1,1] * (2 * p[i] - 1) * Da[i];
		p_temp_a += A[1,2] * (2 * p[j] - 1) * Da[i];

		p_temp_b += A[2,1] * (2 * p[i] - 1) * Db[j];
		p_temp_b += A[2,2] * (2 * p[j] - 1) * Db[j];

		# Bias from other tasks
		if(critic_dimensions > 2)
			a_multiplier = (critic_dimensions - 2) / critic_dimensions
			#=p_temp_a += Da[i] * (-0.5 * R_ext);
			p_temp_b += Db[j] * (-0.5 * R_ext);=#
			#=p_temp_a += Da[i] * (-a_multiplier * R_ext);
			p_temp_b += Db[j] * (-a_multiplier * R_ext);=#
			for(k = 3:critic_dimensions)
				p_temp_a += Da[i] * (A[1,k] * R_ext);
				p_temp_b += Db[j] * (A[2,k] * R_ext);
			end
		end

		# Multiply by probability of occurence of each task
		p_temp_a *= prob_task[1];
		p_temp_b *= prob_task[2];

		# putting it all together
		p_deriv_D_a[i,j] = S[1,1] * p_temp_a + S[1,2] * p_temp_b;
		p_deriv_D_b[i,j] = S[2,1] * p_temp_a + S[2,2] * p_temp_b;

		deriv_p_a[i,j] = pdf(Normal(0,sigma), Da[i]) * p_deriv_D_a[i,j];
		deriv_p_b[i,j] = pdf(Normal(0,sigma), Db[j]) * p_deriv_D_b[i,j];


		#=if(plot_unbiased_learning)
			# include effects of regular learning signal
			#deriv_p_a[i,j] = dist_pdf(invnorm(p[i])) * xa_norm_sq * ( ( p[i]*S_1[i] - (1-p[i])*S_2[i] ) + ( rho_a[i,j] * (2 * p[i] - 1) ) );
			#deriv_p_b[i,j] = dist_pdf(invnorm(p[j])) * xa_norm_sq * ( ( p[j]*S_1[j] - (1-p[j])*S_2[j] ) + ( rho_b[i,j] * (2 * p[j] - 1) ) );
			deriv_p_a[i,j] += dist_pdf(invnorm(p[i])) * xa_norm_sq * ( ( p[i]*S_1[i] - (1-p[i])*S_2[i] ) );
			deriv_p_b[i,j] += dist_pdf(invnorm(p[j])) * xa_norm_sq * ( ( p[j]*S_1[j] - (1-p[j])*S_2[j] ) );
		end=#
		# wta binary output for principal choice, second choice is normalised by magnitude of first
		#=deriv_p_a[i,j] = dist_pdf(invnorm(p[i])) * xa_norm_sq * rho[i,j] * (p[i] + (1-p[i]) * p[i]);
		deriv_p_b[i,j] = - dist_pdf(invnorm(p[j])) * xa_norm_sq * rho[i,j] * (p[j] + (1-p[j]) * p[j]);=#
		# Rescaled outputs - new (in first write-up)
		#=deriv_p_a[i,j] = dist_pdf(invnorm(p[i])) * xa_norm_sq * rho[i,j] * ( (2 * p[i] - 1) * (1 - p[i]) );
		deriv_p_b[i,j] = dist_pdf(invnorm(p[j])) * xa_norm_sq * (-rho[i,j]) * ( (2 * p[j] - 1) * (1 - p[j]) );=#
		# basic flow field
		#deriv_p_a[i,j] = xa_norm_sq * rho[i,j] * (2 * p[i] - 1);
		#deriv_p_b[i,j] = xa_norm_sq * (-rho[i,j]) * (2 * p[j] - 1);
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

sub_task_id_to_plot = 2;

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

#add_trajectories_to_linear_p_plot(exp_results[1],sub_task_id_to_plot);
