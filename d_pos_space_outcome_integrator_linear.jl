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

include("p_space_outcome_integrator_linear.jl"); # trajectories initialised in p-space in this source file


function setup_D_pos_space_basic_variables(local_a = 0.7, local_c = -1)
  print("Setting generic parameters for space for D+ Euler trajectory integration\n")
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


function set_initial_trajectory_points_in_D_pos_space(no_euler_trajectories::Int, duration_euler_integration::Float64, dt_euler::Float64) #via initialisation in p-space
  global euler_integration_timesteps = int(duration_euler_integration / dt_euler) :: Int;
  # p_trajectories : [ p1, p2, dimenstion(task performance), time ]
  global D_pos_trajectories = zeros(no_euler_trajectories, no_euler_trajectories, 2, euler_integration_timesteps);

  #TODO: may call set_initial_trajectory_points_in_p_space() here then convert to D_pos space, meaning we'll
  #   only have one initialisation function

  p_trajectories = set_initial_trajectory_points_in_p_space(no_euler_trajectories, duration_euler_integration, dt_euler);

  # set initial values for trajectories
  for i = 1:no_euler_trajectories
    for j = 1:no_euler_trajectories
      #=p_trajectories_1 = i * ( (1.0 / (no_euler_trajectories + 0) ) ) - 0.5 * (1 / (no_euler_trajectories + 1)  );
      p_trajectories_2 = j * ( (1.0 / (no_euler_trajectories + 0) ) ) - 0.5 * (1 / (no_euler_trajectories + 1)  );

      D_pos_trajectories_1 = invphi(p_trajectories_1);
      D_pos_trajectories_2 = invphi(p_trajectories_2);=#

      #print("$p_trajectories_1 $p_trajectories_2 $D_pos_trajectories_1 $D_pos_trajectories_2\n")
      D_pos_trajectories[i,j,1,1] = invphi( p_trajectories[i,j,1,1] );
      D_pos_trajectories[i,j,2,1] = invphi( p_trajectories[i,j,2,1] );
    end
  end
  #=D_pos_trajectories[1,1,1,1] = 0.3 #i * ( (1.0 / (no_euler_trajectories + 0) ) ) - 0.5 * (1 / (no_euler_trajectories + 1)  );
  D_pos_trajectories[1,1,2,1] = 0.1 #j * ( (1.0 / (no_euler_trajectories + 0) ) ) - 0.5 * (1 / (no_euler_trajectories + 1)  );
=#

  return D_pos_trajectories;
end

function calculate_D_pos_trajectories()
  print("Calculating forward Euler trajectories in D+ space\n")
  ## Tracking of D+ space trajectories (forward Euler integrated) over time
  global no_euler_trajectories = 50; #50; #1 :: Int;
  duration_euler_integration = 1000.0 :: Float64;
  dt_euler = 0.1 :: Float64;

  D_pos_trajectories = set_initial_trajectory_points_in_D_pos_space(no_euler_trajectories, duration_euler_integration, dt_euler);

  for t = 2:euler_integration_timesteps
    # Loop over time
    for trajectory_id_1 = 1:no_euler_trajectories
      for trajectory_id_2 = 1:no_euler_trajectories
      # Loop over trajectories

		  #####
		  #
		  # Calculation of change of difference in outputs
		  #
			# positive association in plotting of D with reward (use d_a as d_a^~)
      d_a = D_pos_trajectories[trajectory_id_1, trajectory_id_2, 1, t-1];
      d_b = D_pos_trajectories[trajectory_id_1, trajectory_id_2, 2, t-1];

		  temp_a = sigma^2 * pdf(Normal(0,sigma), d_a) * 2;
		  temp_b = sigma^2 * pdf(Normal(0,sigma), d_b) * 2;
		  # equations for R^{true} = (2p-1)
      temp_a += A[1,1] * (2 * cdf(Normal(0,sigma), (d_a)) - 1) * (d_a);
      temp_a += A[1,2] * (2 * cdf(Normal(0,sigma), (d_b)) - 1) * (d_a);

      temp_b += A[2,1] * (2 * cdf(Normal(0,sigma), (d_a)) - 1) * (d_b);
      temp_b += A[2,2] * (2 * cdf(Normal(0,sigma), (d_b)) - 1) * (d_b);

		  # Bias from other tasks
      if(critic_dimensions > 2)
        # a_multiplier assumes equal for all
        a_multiplier = (critic_dimensions - 2) / critic_dimensions
        #temp_a += d_a[i] * (-0.5 * R_ext);
        #temp_b += d_b[j] * (-0.5 * R_ext);
        #temp_a += d_a[i] * (-a_multiplier * R_ext);
        #temp_b += d_b[j] * (-a_multiplier * R_ext);
        for(k = 3:critic_dimensions)
          temp_a += d_a * (A[1,k] * R_ext);
          temp_b += d_b * (A[2,k] * R_ext);
        end
      end

		  # Multiply by probability of occurence of each task
		  temp_a *= prob_task[1];
		  temp_b *= prob_task[2];

      # putting it all together
      deriv_D_a_pos = ( O[1] * S[1,1] * temp_a + O[2] * S[1,2] * temp_b );
      deriv_D_b_pos = ( O[1] * S[2,1] * temp_a + O[2] * S[2,2] * temp_b );

		  # we need to transform derivatives to D_pos space
      deriv_D_a_pos *= O[1];
      deriv_D_b_pos *= O[2];

      # Now do a forward Euler update of the trajectory
      D_pos_trajectories[trajectory_id_1, trajectory_id_2, 1, t] = D_pos_trajectories[trajectory_id_1, trajectory_id_2, 1, t-1] + (dt_euler * deriv_D_a_pos);
      D_pos_trajectories[trajectory_id_1, trajectory_id_2, 2, t] = D_pos_trajectories[trajectory_id_1, trajectory_id_2, 2, t-1] + (dt_euler * deriv_D_b_pos);
    end
	 end
  end
  print("Done calculating trajectories\n")
  return D_pos_trajectories;
end

function plot_D_pos_space_trajectories(D_pos_trajectories)
  ## Plotting
  print("Plotting...\n")
  for trajectory_id_1 = 1:no_euler_trajectories
    for trajectory_id_2 = 1:no_euler_trajectories
      local_line_1 = zeros(euler_integration_timesteps,1);
      local_line_2 = zeros(euler_integration_timesteps,1);
      for t = 1:euler_integration_timesteps
        local_line_1[t] = D_pos_trajectories[trajectory_id_1, trajectory_id_2, 1, t];
        local_line_2[t] = D_pos_trajectories[trajectory_id_1, trajectory_id_2, 2, t];
      end
      plot(local_line_1, local_line_2)
      scatter(D_pos_trajectories[trajectory_id_1, trajectory_id_2, 1, 1], D_pos_trajectories[trajectory_id_1, trajectory_id_2, 2, 1], marker="s", c="r", s=40, zorder=2);
      scatter(D_pos_trajectories[trajectory_id_1, trajectory_id_2, 1, euler_integration_timesteps], D_pos_trajectories[trajectory_id_1, trajectory_id_2, 2, euler_integration_timesteps], marker="D", c="g", s=40, zorder=3);
    end
  end
  axis([-20,20,-20,20])

  print("Done\n")
end

function plot_D_pos_space_trajectories_in_p_space(D_pos_trajectories)
  ## Plotting
  print("Plotting...\n")
  for trajectory_id_1 = 1:no_euler_trajectories
    for trajectory_id_2 = 1:no_euler_trajectories
      local_line_1 = zeros(euler_integration_timesteps,1);
      local_line_2 = zeros(euler_integration_timesteps,1);
      for t = 1:euler_integration_timesteps
        local_line_1[t] = cdf(Normal(0,1), D_pos_trajectories[trajectory_id_1, trajectory_id_2, 1, t]);
        local_line_2[t] = cdf(Normal(0,1), D_pos_trajectories[trajectory_id_1, trajectory_id_2, 2, t]);
      end
      plot(local_line_1, local_line_2)
      scatter(cdf(Normal(0,1), D_pos_trajectories[trajectory_id_1, trajectory_id_2, 1, 1]), cdf(Normal(0,1), D_pos_trajectories[trajectory_id_1, trajectory_id_2, 2, 1]), marker="s", c="r", s=40, zorder=2);
      scatter(cdf(Normal(0,1), D_pos_trajectories[trajectory_id_1, trajectory_id_2, 1, euler_integration_timesteps]), cdf(Normal(0,1), D_pos_trajectories[trajectory_id_1, trajectory_id_2, 2, euler_integration_timesteps]), marker="D", c="g", s=40, zorder=3);
    end
  end
  axis([0,1,0,1])

  print("Done\n")
end



function report_D_pos_trajectory_end_point_results(D_pos_trajectories)
  all_correct = 1
  ball_radius = 0.01

  count_both_correct = 0
  count_task1_correct = 0
  count_task2_correct = 0
  count_midline = 0
  count_fail = 0

  global D_pos_trajectory_initial_points_1 = zeros(no_euler_trajectories, no_euler_trajectories);
  global D_pos_trajectory_initial_points_2 = zeros(no_euler_trajectories, no_euler_trajectories);
  global D_pos_trajectory_end_points = zeros(no_euler_trajectories, no_euler_trajectories);

  figure()
  for trajectory_id_1 = 1:no_euler_trajectories
    for trajectory_id_2 = 1:no_euler_trajectories
      D_pos_trajectory_initial_points_1[trajectory_id_1, trajectory_id_2] = dist_cdf(D_pos_trajectories[trajectory_id_1, trajectory_id_2, 1, 1]);
      D_pos_trajectory_initial_points_2[trajectory_id_1, trajectory_id_2] = dist_cdf(D_pos_trajectories[trajectory_id_1, trajectory_id_2, 2, 1]);

      print("Initial point ",dist_cdf(D_pos_trajectories[trajectory_id_1, trajectory_id_2, 1, 1])," ",dist_cdf(D_pos_trajectories[trajectory_id_1, trajectory_id_2, 2, 1])," end point ",dist_cdf(D_pos_trajectories[trajectory_id_1, trajectory_id_2, 1, end])," ",dist_cdf(D_pos_trajectories[trajectory_id_1, trajectory_id_2, 2, end]) );
      if ( abs( dist_cdf(D_pos_trajectories[trajectory_id_1, trajectory_id_2, 1, end]) - all_correct ) < ball_radius )
        if ( abs( dist_cdf(D_pos_trajectories[trajectory_id_1, trajectory_id_2, 2, end]) - all_correct ) < ball_radius )
          print(" Both win \n")
          count_both_correct += 1;
          D_pos_trajectory_end_points[trajectory_id_1, trajectory_id_2] = 1;
          #scatter(dist_cdf(D_pos_trajectories[trajectory_id_1, trajectory_id_2, 1, 1]), dist_cdf(D_pos_trajectories[trajectory_id_1, trajectory_id_2, 2, 1]), marker="s", c="r", s=40, zorder=2)
        else
          print(" Task 1 win \n")
          count_task1_correct += 1;
          D_pos_trajectory_end_points[trajectory_id_1, trajectory_id_2] = 2;
          #scatter(dist_cdf(D_pos_trajectories[trajectory_id_1, trajectory_id_2, 1, 1]), dist_cdf(D_pos_trajectories[trajectory_id_1, trajectory_id_2, 2, 1]), marker="s", c="g", s=40, zorder=2)
        end
      elseif ( abs( dist_cdf(D_pos_trajectories[trajectory_id_1, trajectory_id_2, 2, end]) - all_correct ) < ball_radius )
        print(" Task 2 win \n")
        count_task2_correct += 1;
        D_pos_trajectory_end_points[trajectory_id_1, trajectory_id_2] = 3;
        #scatter(dist_cdf(D_pos_trajectories[trajectory_id_1, trajectory_id_2, 1, 1]), dist_cdf(D_pos_trajectories[trajectory_id_1, trajectory_id_2, 2, 1]), marker="s", c="b", s=40, zorder=2)
      elseif ( abs( dist_cdf(D_pos_trajectories[trajectory_id_1, trajectory_id_2, 2, end]) - dist_cdf(D_pos_trajectories[trajectory_id_1, trajectory_id_2, 2, 1]) ) < ball_radius )
        print(" approximate midline \n");
        D_pos_trajectory_end_points[trajectory_id_1, trajectory_id_2] = 4;
        count_midline += 1;
        #scatter(dist_cdf(D_pos_trajectories[trajectory_id_1, trajectory_id_2, 1, 1]), dist_cdf(D_pos_trajectories[trajectory_id_1, trajectory_id_2, 2, 1]), marker="s", c="c", s=40, zorder=2)
      else
        print(" Both fail \n")
        count_fail += 1;
        D_pos_trajectory_end_points[trajectory_id_1, trajectory_id_2] = 5;
        #scatter(dist_cdf(D_pos_trajectories[trajectory_id_1, trajectory_id_2, 1, 1]), dist_cdf(D_pos_trajectories[trajectory_id_1, trajectory_id_2, 2, 1]), marker="s", c="w", s=40, zorder=2)
      end
    end
  end

  #pcolor(D_pos_trajectory_initial_points_1,D_pos_trajectory_initial_points_2,D_pos_trajectory_end_points, cmap="RdBu", vmin=0, vmax=5);
  #pcolormesh(D_pos_trajectory_initial_points_1,D_pos_trajectory_initial_points_2,D_pos_trajectory_end_points, cmap="RdBu", vmin=0, vmax=5);
  imshow(D_pos_trajectory_end_points', cmap="RdBu", vmin=0, vmax=5, interpolation="nearest", origin="lower", extent=[0, 1, 0, 1]);
  #RdBu
  #YlOrRd_r
  xlabel("task 1 initial performance")
  ylabel("task 2 initial performance")
  title("End point classification of trajectories");
  colorbar();

  print("Counts both correct: ", count_both_correct, ", task 1 correct: ", count_task1_correct, ", task 2 correct: ", count_task2_correct, ", midline convergence: ", count_midline, ", lost in the middle: ", count_fail, "\n")
  print("Done\n")
end


function print_single_D_pos_trajectory(D_pos_trajectories, trajectory_id_1, trajectory_id_2, t_begin=1,t_end=euler_integration_timesteps)
  for t = t_begin:t_end
    print("$t ", D_pos_trajectories[trajectory_id_1, trajectory_id_2, 1, t], " ", D_pos_trajectories[trajectory_id_1, trajectory_id_2, 2, t]," \n");
  end
  print("\n\n1  : ", D_pos_trajectories[trajectory_id_1, trajectory_id_2, 1, 1], " ", D_pos_trajectories[trajectory_id_1, trajectory_id_2, 2, 1]," \n");
  print("end: ", D_pos_trajectories[trajectory_id_1, trajectory_id_2, 1, end], " ", D_pos_trajectories[trajectory_id_1, trajectory_id_2, 2, end]," \n");
end


function run_local_D_pos_trajectories(local_similarity = 0.5)
  setup_D_pos_space_basic_variables(local_similarity);
  D_pos_trajectories = calculate_D_pos_trajectories();
  figure();
  plot_D_pos_space_trajectories(D_pos_trajectories);

  figure();
  plot_D_pos_space_trajectories_in_p_space(D_pos_trajectories);

  report_D_pos_trajectory_end_point_results(D_pos_trajectories);
end
