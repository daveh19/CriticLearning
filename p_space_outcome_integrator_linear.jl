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
include("plotting_assist_functions.jl");


function setup_p_space_basic_variables()
  print("Setting generic parameters for space for Euler trajectory integration\n")
  ## Plotting over D, D~ (+ve), and p optional
  global use_plot_over_D_pos = false :: Bool;
  global use_plot_over_D = false :: Bool;
  global use_plot_over_p = true :: Bool;
  global use_overlay_performance_on_D = true :: Bool;
  global use_add_trajectories_to_plot = false :: Bool;
  global sub_task_id_to_plot = 1 ::Int;
  global use_plot_measured_proportion_correct = false ::Bool;

  ## Space over which vector field is calculated / plotted
  global no_points = 25; #30;
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

  global D_pos_scale = 20.0:: Float64;
  global D_scale = 20.0 :: Float64;
  global p_scale = 1.0:: Float64;

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
  global prob_task = ones(1,critic_dimensions);
  prob_task /= critic_dimensions;
  #prob_task = [1, 0.001, 10, 10]; # manual tweaking
  #prob_task /= sum(prob_task); # normalise, so I can use arbitrary units
  # this influences Confustion matrix
  for k = 1:critic_dimensions
	   C[k,:] = prob_task;
   end
  global A = eye(critic_dimensions) - C;

  # Input representation similarity parameter
  global a = 0.99; #0.9;
  global S = [1 a; a 1]

  # Output correlation with +ve D
  global O = [1; -1];

  # Noise and external bias
  global sigma = 1;
  global R_ext = 0;
  print("Done\n")
end


function calculate_p_trajectories()
  print("Calculating forward Euler trajectories in p-space\n")
  ## Tracking of p-space trajectories (forward Euler integrated) over time
  global no_euler_trajectories = 5 :: Int;
  duration_euler_integration = 10000.0 :: Float64;
  dt_euler = 0.1 :: Float64;
  global euler_integration_timesteps = int(duration_euler_integration / dt_euler) :: Int;
  # p_trajectories : [ trajectory_id, p1, p2, time ]
  p_trajectories = zeros(no_euler_trajectories, no_euler_trajectories, 2, euler_integration_timesteps);
  # set initial values for trajectories
  for i = 1:no_euler_trajectories
    for j = 1:no_euler_trajectories
      p_trajectories[i,j,1,1] = i * ( (1.0 / (no_euler_trajectories + 0) ) ) - 0.5 * (1 / (no_euler_trajectories + 1)  );
      p_trajectories[i,j,2,1] = j * ( (1.0 / (no_euler_trajectories + 0) ) ) - 0.5 * (1 / (no_euler_trajectories + 1)  );
    end
  end

  for t = 2:euler_integration_timesteps
    # Loop over time
    for trajectory_id_1 = 1:no_euler_trajectories
      for trajectory_id_2 = 1:no_euler_trajectories
      # Loop over trajectories

		  #####
		  #
		  # Calculation of change of probability of outcome
		  #
      p_a = p_trajectories[trajectory_id_1, trajectory_id_2, 1, t-1];
      p_b = p_trajectories[trajectory_id_1, trajectory_id_2, 2, t-1];
		  Da = invphi(p_a);
		  Db = invphi(p_b);
      #=if (Da == Inf || Da == -Inf || Db == Inf || Db == -Inf)
        print("DEBUG, $trajectory_id_1, $trajectory_id_2, $t, $Da, $Db \n")
      end=#
      #=if(Da==Inf)
        Da = 100;
      end
      if(Da==-Inf)
        Da = -100;
      end
      if(Db==Inf)
        Db = 100;
      end
      if(Db==-Inf)
        Db = -100;
      end=#

		  p_temp_a = sigma^2 * pdf(Normal(0,sigma), Da) * 2;
		  p_temp_b = sigma^2 * pdf(Normal(0,sigma), Db) * 2;
		  # equations for R^{true} = (2p-1)
		  p_temp_a += A[1,1] * (2 * p_a - 1) * Da;
		  p_temp_a += A[1,2] * (2 * p_b - 1) * Da;

		  p_temp_b += A[2,1] * (2 * p_a - 1) * Db;
		  p_temp_b += A[2,2] * (2 * p_b - 1) * Db;

		  # Bias from other tasks
		  if(critic_dimensions > 2)
			  a_multiplier = (critic_dimensions - 2) / critic_dimensions
			  #=p_temp_a += Da[i] * (-0.5 * R_ext);
			  p_temp_b += Db[j] * (-0.5 * R_ext);=#
			  #=p_temp_a += Da[i] * (-a_multiplier * R_ext);
			  p_temp_b += Db[j] * (-a_multiplier * R_ext);=#
			  for(k = 3:critic_dimensions)
				  p_temp_a += Da * (A[1,k] * R_ext);
				  p_temp_b += Db * (A[2,k] * R_ext);
			   end
		  end

		  # Multiply by probability of occurence of each task
		  p_temp_a *= prob_task[1];
		  p_temp_b *= prob_task[2];

		  # putting it all together
		  p_deriv_D_a = (O[1] * S[1,1] * p_temp_a + O[2] * S[1,2] * p_temp_b);
		  p_deriv_D_b = (O[1] * S[2,1] * p_temp_a + O[2] * S[2,2] * p_temp_b);

		  # we need to transform derivatives to D_pos space
		  p_deriv_D_a *= O[1];
		  p_deriv_D_b *= O[2];

		  # and we scale everything by the pdf of the underlying probability
		  deriv_p_a = pdf(Normal(0,sigma), Da) * p_deriv_D_a;
		  deriv_p_b = pdf(Normal(0,sigma), Db) * p_deriv_D_b;

      # Now do a forward Euler update of the trajectory
      p_trajectories[trajectory_id_1, trajectory_id_2, 1, t] = p_trajectories[trajectory_id_1, trajectory_id_2, 1, t-1] + (dt_euler * deriv_p_a);
      p_trajectories[trajectory_id_1, trajectory_id_2, 2, t] = p_trajectories[trajectory_id_1, trajectory_id_2, 2, t-1] + (dt_euler * deriv_p_b);

      if (p_trajectories[trajectory_id_1, trajectory_id_2, 1, t] <= 0)
        p_trajectories[trajectory_id_1, trajectory_id_2, 1, t] = 0+epsilon;
      elseif (p_trajectories[trajectory_id_1, trajectory_id_2, 1, t] >= 1)
        p_trajectories[trajectory_id_1, trajectory_id_2, 1, t] = 1-epsilon;
      end

      if (p_trajectories[trajectory_id_1, trajectory_id_2, 2, t] <= 0)
        p_trajectories[trajectory_id_1, trajectory_id_2, 2, t] = 0+epsilon;
      elseif (p_trajectories[trajectory_id_1, trajectory_id_2, 2, t] >= 1)
        p_trajectories[trajectory_id_1, trajectory_id_2, 2, t] = 1-epsilon;
      end
    end
	 end
  end
  print("Done calculating trajectories\n")
  return p_trajectories;
end

function plot_p_space_trajectories(p_trajectories)
  ## Plotting
  print("Plotting...\n")
  #figure();
  #=for t = 2:euler_integration_timesteps
  # Loop over time
  for trajectory_id_1 = 1:no_euler_trajectories
    for trajectory_id_2 = 1:no_euler_trajectories
      scatter(p_trajectories[trajectory_id_1, trajectory_id_2, 1, :], p_trajectories[trajectory_id_1, trajectory_id_2, 2, :])
    end
  end
  end=#
  for trajectory_id_1 = 1:no_euler_trajectories
    for trajectory_id_2 = 1:no_euler_trajectories
      local_line_1 = zeros(euler_integration_timesteps,1);
      local_line_2 = zeros(euler_integration_timesteps,1);
      for t = 1:euler_integration_timesteps
        local_line_1[t] = p_trajectories[trajectory_id_1, trajectory_id_2, 1, t];
        local_line_2[t] = p_trajectories[trajectory_id_1, trajectory_id_2, 2, t];
      end
      plot(local_line_1, local_line_2)
      scatter(p_trajectories[trajectory_id_1, trajectory_id_2, 1, 1], p_trajectories[trajectory_id_1, trajectory_id_2, 2, 1], marker="s", c="r", s=40, zorder=2);
      scatter(p_trajectories[trajectory_id_1, trajectory_id_2, 1, euler_integration_timesteps], p_trajectories[trajectory_id_1, trajectory_id_2, 2, euler_integration_timesteps], marker="D", c="g", s=40, zorder=3);
    end
  end
  axis([0,1,0,1])

  print("Done\n")
end

function run_local_p_trajectories()
  setup_p_space_basic_variables()
  p_trajectories = calculate_p_trajectories()
  figure()
  plot_p_space_trajectories(p_trajectories)
end
