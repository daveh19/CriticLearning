using PyPlot;
using Distributions;
using LaTeXStrings;
using Debug;

use_linear_outputs = true :: Bool;
no_similarities = 11;
use_trajectory_tracing_only = true :: Bool;
use_show_plots = false :: Bool;

use_integrate_in_p_space = false :: Bool;

if (use_linear_outputs)
  if(use_integrate_in_p_space)
    include("p_space_outcome_integrator_linear.jl");
  else
    include("d_pos_space_outcome_integrator_linear.jl");
  end
  include("plot_D_vector.jl");
else
  include("plot_Dbinary_vector.jl");
end

similarity_set = linspace(0,1,no_similarities);

## could just loop over trajectory calculations
if (use_trajectory_tracing_only)
  for i = 1:no_similarities
    similarity = similarity_set[i];
    if (use_integrate_in_p_space)
      setup_p_space_basic_variables(similarity)
      p_trajectories = calculate_p_trajectories()
      if(use_show_plots)
        figure()
        plot_p_space_trajectories(p_trajectories)
      end
      report_p_trajectory_end_point_results(p_trajectories, similarity)
    else
      setup_D_pos_space_basic_variables(similarity);
      D_pos_trajectories = calculate_D_pos_trajectories()
      if(use_show_plots)
        #figure()
        #plot_D_pos_space_trajectories(D_pos_trajectories);
        figure()
        plot_D_pos_space_trajectories_in_p_space(D_pos_trajectories);
      end
      report_D_pos_trajectory_end_point_results(D_pos_trajectories, similarity);
    end
  end
else
  ## or loop over plotting of vector fields (which costs almost nothing)
  for i = 1:no_similarities
    similarity = similarity_set[i];
    if (use_linear_outputs)
      setup_plot_D_basic_variables(similarity);
      if (use_integrate_in_p_space)
        # set a variable here which is used in plot_D_vector.jl to choose integration space
        global use_overlay_p_Euler_trajectories = true;
        global use_overlay_D_pos_Euler_trajectories = false;
      else
        global use_overlay_p_Euler_trajectories = false;
        global use_overlay_D_pos_Euler_trajectories = true;
      end
      use_plot_over_p = true;
      calculate_linear_model_flow_vectors();
      plot_linear_model_flow_vectors();
    else
      setup_plot_D_binary_basic_variables(similarity);
      use_plot_over_p = true;
      calculate_binary_model_flow_vectors();
      plot_binary_model_flow_vectors();
    end
  end
end

print("End of loop over similarity matrices\n");
