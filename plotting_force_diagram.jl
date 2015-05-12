using PyPlot;
using Distributions;

include("inverse_cdf.jl"); #contains invnorm(), consider switching to invphi()

no_points = 101;
#no_points = 10;
#no_y_points = no_points - 1; 
# The no_y_points is to ensure that I plot the vector field in the right direction,
#	 julia is column major but matplot lib is row major which causes confusion! 
#	Set no_y_points = no_points - 1; to check if an error is thrown, no error means
#		that the array access is correct.
no_y_points = no_points;
p = linspace(0, 1, no_points);
p_y = linspace(0, 1, no_y_points);

## There are a number of alternative ways to calculate pdf and cdf inverse
dist_pdf(x) = pdf(Normal(0,1), x);
dist_cdf(x) = cdf(Normal(0,1), x);
# Note: inv_cdf(x) != 1.0 / cdf(Normal(0,1), x); #Not 1/fn but inverse function!!
invphi(p) = sqrt(2) * erfinv(2 * p - 1.0)

## Constants
xa_norm_sq = 10.0;
p_ext = 0.30;

## Success signal is necessary for unbiased learning
plot_unbiased_learning = true :: Bool;
# constant Success signal
#S_1 = +1.0 / 10.0;
#S_2 = -1.0 / 10.0;
# linear decay of Success signal
#S_1 = 1.0 - p;
#S_2 = -p - 1.0;
# Stationary Success signal
S_1 = 1.0 - (2*p-1.0);
S_2 = -1.0 - (2*p-1.0);

## Rho represents error in estimation of Success signal
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

## Calculate vector flow field
deriv_p_a = zeros(no_points, no_y_points);
deriv_p_b = zeros(no_points, no_y_points);
for i = 1:no_points
	for j = 1:(no_y_points)
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
		#deriv_p_a[i,j] = dist_pdf(invnorm(p[i])) * xa_norm_sq * rho_a[i,j] * (2 * p[i] - 1);
		#deriv_p_b[i,j] = dist_pdf(invnorm(p[j])) * xa_norm_sq * rho_b[i,j] * (2 * p[j] - 1);
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
filename_base = string("vector_field_", filename_change);
filename_quiver = string("quiver_",filename_base,".pdf")
filename_stream = string("stream_",filename_base,".pdf")

figure();
quiver(p,p,deriv_p_a', deriv_p_b');
#quiver(p,p_y,deriv_p_a', deriv_p_b');
savefig(filename_quiver);

figure();
streamplot(p,p,deriv_p_a',deriv_p_b');
#streamplot(p,p_y,deriv_p_a',deriv_p_b')
savefig(filename_stream);

