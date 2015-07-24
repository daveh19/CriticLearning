using PyPlot;
using Distributions;

### Useful functions
## There are a number of alternative ways to calculate pdf and cdf inverse
dist_pdf(x) = pdf(Normal(0,1), x);
dist_cdf(x) = cdf(Normal(0,1), x);
# Note: inv_cdf(x) != 1.0 / cdf(Normal(0,1), x); #Not 1/fn but inverse function!!
include("inverse_cdf.jl"); #contains invnorm(), consider switching to invphi()
invphi(p) = sqrt(2) * erfinv(2 * p - 1.0)


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
d_a = linspace(-2.5,2.5, no_points);
d_b = linspace(-2.5,2.5, no_points);

## Vector flow field variables
deriv_p_a = zeros(no_points, no_y_points);
deriv_p_b = zeros(no_points, no_y_points);
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
c = 0.5
C = [1-c c; c 1-c];
A = eye(2) - C;

# Input representation similarity parameter
a = 0; #0.9;
S = [1 a; a 1]

# Noise and external bias
sigma = 1;
#rho_ext = -0.5;

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
		temp_a = sigma^2 * pdf(Normal(0,sigma), d_a[i]) * 2; 
		temp_b = sigma^2 * pdf(Normal(0,sigma), d_b[j]) * 2;
		# equations for R^{true} = (2p-1)
		temp_a += A[1,1] * (2 * cdf(Normal(0,sigma), d_a[i]) - 1) * d_a[i];
		temp_a += A[1,2] * (2 * cdf(Normal(0,sigma), d_b[j]) - 1) * d_a[i];
		temp_b += A[2,1] * (2 * cdf(Normal(0,sigma), d_a[i]) - 1) * d_b[j];
		temp_b += A[2,2] * (2 * cdf(Normal(0,sigma), d_b[j]) - 1) * d_b[j];

		# # Bias from other tasks (change how it's formulated)
		# temp_a += rho_ext * d_a[i];
		# temp_b += rho_ext * d_b[j];

		# putting it all together
		deriv_D_a[i,j] = S[1,1] * temp_a + S[1,2] * temp_b;
		deriv_D_b[i,j] = S[2,1] * temp_a + S[2,2] * temp_b;


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
		deriv_p_a[i,j] = dist_pdf(invnorm(p[i])) * xa_norm_sq * rho_a[i,j] * (2 * p[i] - 1);
		deriv_p_b[i,j] = dist_pdf(invnorm(p[j])) * xa_norm_sq * rho_b[i,j] * (2 * p[j] - 1);
		if(plot_unbiased_learning)
			# include effects of regular learning signal
			#deriv_p_a[i,j] = dist_pdf(invnorm(p[i])) * xa_norm_sq * ( ( p[i]*S_1[i] - (1-p[i])*S_2[i] ) + ( rho_a[i,j] * (2 * p[i] - 1) ) );
			#deriv_p_b[i,j] = dist_pdf(invnorm(p[j])) * xa_norm_sq * ( ( p[j]*S_1[j] - (1-p[j])*S_2[j] ) + ( rho_b[i,j] * (2 * p[j] - 1) ) );
			deriv_p_a[i,j] += dist_pdf(invnorm(p[i])) * xa_norm_sq * ( ( p[i]*S_1[i] - (1-p[i])*S_2[i] ) );
			deriv_p_b[i,j] += dist_pdf(invnorm(p[j])) * xa_norm_sq * ( ( p[j]*S_1[j] - (1-p[j])*S_2[j] ) );
		end
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

figure();
#streamplot(d_a,d_b,deriv_D_a',deriv_D_b');
quiver(d_a,d_b,deriv_D_a',deriv_D_b');
#plot(d_a, Db_null);
## x=0 and y=0 lines for visual inspection
#=origin = zeros(no_points);
origin_space = linspace(-100,100,no_points);
plot(origin, origin_space);
plot(origin_space, origin);=#

