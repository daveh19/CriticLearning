using PyPlot;
using Distributions;

include("inverse_cdf.jl");

no_points = 101;
p = linspace(0, 1, no_points);


dist_pdf(x) = pdf(Normal(0,1), x);
dist_cdf(x) = cdf(Normal(0,1), x);
#inv_cdf(x) = 1.0 / cdf(Normal(0,1), x); #Not 1/fn but inverse function!!
invphi(p) = sqrt(2) * erfinv(2 * p - 1.0)

xa_norm_sq = 1.0;

rho = zeros(no_points,no_points);
for i = 1:no_points
	for j = 1:no_points
		rho[i,j] = (p[i] - p[j]) / 2.0;
	end
end

deriv_p_a = zeros(no_points, no_points);
deriv_p_b = zeros(no_points, no_points);
for i = 1:no_points
	for j = 1:no_points
		# unbounded post firing rate equations
		#=deriv_p_a[i,j] = dist_pdf(invnorm(p[i])) * xa_norm_sq * rho[i,j] * invnorm(p[i]);
		deriv_p_b[i,j] = - dist_pdf(invnorm(p[j])) * xa_norm_sq * rho[i,j] * invnorm(p[j]);=#
		# binary output neurons
		#=deriv_p_a[i,j] = dist_pdf(invnorm(p[i])) * xa_norm_sq * rho[i,j] * p[i];
		deriv_p_b[i,j] = - dist_pdf(invnorm(p[j])) * xa_norm_sq * rho[i,j] * p[j];=#
		# binary output neurons - alternative
		#=deriv_p_a[i,j] = dist_pdf(invnorm(p[i])) * xa_norm_sq * rho[i,j] * p[i] * (1-p[i]);
		deriv_p_b[i,j] = - dist_pdf(invnorm(p[j])) * xa_norm_sq * rho[i,j] * p[j] * (1-p[j]);=#
		# wta binary output for principal choice, second choice is normalised by magnitude of first
		deriv_p_a[i,j] = dist_pdf(invnorm(p[i])) * xa_norm_sq * rho[i,j] * (p[i] + (1-p[i]) * p[i]);
		deriv_p_b[i,j] = - dist_pdf(invnorm(p[j])) * xa_norm_sq * rho[i,j] * (p[j] + (1-p[j]) * p[j]);
	end
end

figure();
quiver(p,p,deriv_p_a, deriv_p_b);
savefig("vector_field_scaled_output.pdf");

figure();
streamplot(p,p,deriv_p_a,deriv_p_b)
savefig("quiver_vector_field_output.pdf");