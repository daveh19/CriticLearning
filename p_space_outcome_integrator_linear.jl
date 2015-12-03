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
invphi(p) = sqrt(2) * erfinv(2 * p - 1.0)
include("plotting_assist_functions.jl");


function setup_p_space_basic_variables(local_a = 0.5, local_c = -1)
  print("Setting generic parameters for space for Euler trajectory integration\n")
  ## Space over which vector field is calculated / plotted
  global no_points = 25; #30;
  global epsilon = 0; #1e-7
  global no_y_points = no_points;

  # Confusion parameter
  global critic_dimensions = 2;
  # perfect critic (overwritten if any of the following are active)
  global C = eye(critic_dimensions)

  # Probabilistic presentation of individual tasks critic
  global prob_task = ones(1,critic_dimensions);
  prob_task /= critic_dimensions;
  # this influences Confustion matrix
  for k = 1:critic_dimensions
	  C[k,:] = prob_task;
  end
  global A = eye(critic_dimensions) - C;

  if(local_c != -1)
    global A = eye(critic_dimensions) - local_c;
  end

  # Input representation similarity parameter
  global a = local_a; #0.5; #0.9;
  global S = [1 a; a 1]

  # Output correlation with +ve D
  global O = [1; -1];

  # Noise and external bias
  global sigma = 1;
  global R_ext = 0;
  print("Done\n")
end


function set_initial_trajectory_points_in_p_space(no_euler_trajectories)
  global p_trajectories;

  # set initial values for trajectories
  for i = 1:no_euler_trajectories
    for j = 1:no_euler_trajectories
      p_trajectories[i,j,1,1] = i * ( (1.0 / (no_euler_trajectories + 0) ) ) - 0.5 * (1 / (no_euler_trajectories + 1)  );
      p_trajectories[i,j,2,1] = j * ( (1.0 / (no_euler_trajectories + 0) ) ) - 0.5 * (1 / (no_euler_trajectories + 1)  );
    end
  end
  #=D_pos_trajectories[1,1,1,1] = 0.3 #i * ( (1.0 / (no_euler_trajectories + 0) ) ) - 0.5 * (1 / (no_euler_trajectories + 1)  );
  D_pos_trajectories[1,1,2,1] = 0.1 #j * ( (1.0 / (no_euler_trajectories + 0) ) ) - 0.5 * (1 / (no_euler_trajectories + 1)  );
=#

  return p_trajectories;
end


function calculate_p_trajectories()
  print("Calculating forward Euler trajectories in p-space\n")
  ## Tracking of p-space trajectories (forward Euler integrated) over time
  global no_euler_trajectories = 5; #1 :: Int;
  duration_euler_integration = 1000.0 :: Float64;
  dt_euler = 0.1 :: Float64;
  global euler_integration_timesteps = int(duration_euler_integration / dt_euler) :: Int;
  # p_trajectories : [ trajectory_id, p1, p2, time ]
  global p_trajectories = zeros(no_euler_trajectories, no_euler_trajectories, 2, euler_integration_timesteps);

  p_trajectories = set_initial_trajectory_points_in_p_space(no_euler_trajectories);

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
      large_bound = 1e6;
      if(Da > large_bound)
        Da = large_bound;
      elseif(Da < -large_bound)
        Da = -large_bound;
      end
      if(Db > large_bound)
        Db = large_bound;
      elseif(Db < -large_bound)
        Db = -large_bound;
      end

		  p_temp_a = sigma^2 * pdf(Normal(0,sigma), Da) * 2;
		  p_temp_b = sigma^2 * pdf(Normal(0,sigma), Db) * 2;
		  # equations for R^{true} = (2p-1)
		  p_temp_a += A[1,1] * (2 * p_a - 1) * Da;
		  p_temp_a += A[1,2] * (2 * p_b - 1) * Da;

		  p_temp_b += A[2,1] * (2 * p_a - 1) * Db;
		  p_temp_b += A[2,2] * (2 * p_b - 1) * Db;
#@bp (p_temp_a == Inf)
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
#@bp isnan(p_deriv_D_a)
		  # we need to transform derivatives to D_pos space
		  p_deriv_D_a *= O[1];
		  p_deriv_D_b *= O[2];
#@bp isnan(p_deriv_D_a)
		  # and we scale everything by the pdf of the underlying probability

      #=if( abs(p_deriv_D_a) < Inf )
        if (pdf(Normal(0,sigma),Da) > 0)
		        deriv_p_a = pdf(Normal(0,sigma), Da) * p_deriv_D_a;
        else
            deriv_p_a = 0.0;
        end
      else
        if (pdf(Normal(0,sigma),Da) == 0)
          deriv_p_a = 0.0;
          print("handle NaN\n")
        elseif (isnan(p_deriv_D_a))
          @bp
          print("handle NaN\n")
        else
          @bp
          print("handle infinity\n")
          deriv_p_a = pdf(Normal(0,sigma), Da) * p_deriv_D_a;
        end
      end=#

      #=if( abs(p_deriv_D_b) < Inf )
        if (pdf(Normal(0,sigma),Db) > 0)
		        deriv_p_b = pdf(Normal(0,sigma), Db) * p_deriv_D_b;
        else
            deriv_p_b = 0.0;
        end
      else
        if (pdf(Normal(0,sigma),Db) == 0)
          deriv_p_b = 0.0;
          print("handle NaN b\n")
        elseif (isnan(p_deriv_D_b))
          @bp
          print("handle another NaN\n")
        else
          @bp
          print("handle infinity b\n")
          deriv_p_b = pdf(Normal(0,sigma), Db) * p_deriv_D_b;
        end
      end=#

      deriv_p_a = pdf(Normal(0,sigma), Da) * p_deriv_D_a;
      deriv_p_b = pdf(Normal(0,sigma), Db) * p_deriv_D_b;

      # Now do a forward Euler update of the trajectory
      if (abs(deriv_p_a) < Inf )
        p_trajectories[trajectory_id_1, trajectory_id_2, 1, t] = p_trajectories[trajectory_id_1, trajectory_id_2, 1, t-1] + (dt_euler * deriv_p_a);
      elseif (deriv_p_a == Inf)
        @bp
        p_trajectories[trajectory_id_1, trajectory_id_2, 1, t] = 1;
      elseif (deriv_p_a == -Inf)
        @bp
        p_trajectories[trajectory_id_1, trajectory_id_2, 1, t] = 0;
      else
        @bp
        print("Catch error!!\n")
      end
      if ( abs(deriv_p_b) < Inf )
        p_trajectories[trajectory_id_1, trajectory_id_2, 2, t] = p_trajectories[trajectory_id_1, trajectory_id_2, 2, t-1] + (dt_euler * deriv_p_b);
      elseif (deriv_p_b == Inf)
        @bp
        p_trajectories[trajectory_id_1, trajectory_id_2, 2, t] = 1;
      elseif (deriv_p_b == -Inf)
        @bp
        p_trajectories[trajectory_id_1, trajectory_id_2, 2, t] = 0;
      else
        @bp
        print("Catch error!\n")
      end

#TODO: consider whether it's really a good idea to bounce back inside the boundary rather than onto it (set epsilon = 0)
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


function report_end_point_results(p_trajectories)
  all_correct = 1 # line 193:
  ball_radius = 0.01 # line 195:

  count_both_correct = 0 # line 196:
  count_task1_correct = 0 # line 197:
  count_task2_correct = 0 # line 198:
  count_other = 0 # line 200:

  for trajectory_id_1 = 1:no_euler_trajectories
    for trajectory_id_2 = 1:no_euler_trajectories
      print("Initial point ",p_trajectories[trajectory_id_1, trajectory_id_2, 1, 1]," ",p_trajectories[trajectory_id_1, trajectory_id_2, 2, 1]," end point ",p_trajectories[trajectory_id_1, trajectory_id_2, 1, end]," ",p_trajectories[trajectory_id_1, trajectory_id_2, 2, end]) # line 204:
      if ( abs(p_trajectories[trajectory_id_1, trajectory_id_2, 1, end] - all_correct) < ball_radius )
        if ( abs(p_trajectories[trajectory_id_1, trajectory_id_2, 2, end] - all_correct) < ball_radius )
          print(" Both win \n")
          count_both_correct += 1;
        else
          print(" Task 1 win \n")
          count_task1_correct += 1;
        end
      elseif ( abs(p_trajectories[trajectory_id_1, trajectory_id_2, 2, end] - all_correct) < ball_radius )
        print(" Task 2 win \n")
        count_task2_correct += 1;
      else
        print(" Both fail \n")
        count_other += 1;
      end
    end
  end

  print("Counts ", count_both_correct, " ", count_task1_correct, " ", count_task2_correct, " ", count_other, "\n")
  print("Done\n")
end


function print_single_trajectory(p_trajectories, trajectory_id_1, trajectory_id_2, t_begin=1,t_end=euler_integration_timesteps)
  for t = t_begin:t_end
    print("$t ", p_trajectories[trajectory_id_1, trajectory_id_2, 1, t], " ", p_trajectories[trajectory_id_1, trajectory_id_2, 2, t]," \n");
  end
  print("\n\n1  : ", p_trajectories[trajectory_id_1, trajectory_id_2, 1, 1], " ", p_trajectories[trajectory_id_1, trajectory_id_2, 2, 1]," \n");
  print("end: ", p_trajectories[trajectory_id_1, trajectory_id_2, 1, end], " ", p_trajectories[trajectory_id_1, trajectory_id_2, 2, end]," \n");
end


function run_local_p_trajectories()
  setup_p_space_basic_variables()
  p_trajectories = calculate_p_trajectories()
  figure()
  plot_p_space_trajectories(p_trajectories)
  report_end_point_results(p_trajectories);
end
