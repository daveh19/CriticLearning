using PyPlot;
using Distributions;
using LaTeXStrings;
using Debug;

include("p_space_outcome_integrator_linear.jl");
include("plot_D_vector.jl");

no_similarities = 11;
use_trajectory_tracing_only = true :: Bool;
use_show_plots = false :: Bool;

similarity_set = linspace(0,1,no_similarities);

## could just loop over trajectory calculations
if (use_trajectory_tracing_only)
  for i = 1:no_similarities
    similarity = similarity_set[i];
    setup_p_space_basic_variables(similarity)
    p_trajectories = calculate_p_trajectories()
    if(use_show_plots)
      figure()
      plot_p_space_trajectories(p_trajectories)
    end
    report_end_point_results(p_trajectories)
  end
else
  ## or loop over plotting of vector fields (which costs almost nothing)
  for i = 1:no_similarities
    similarity = similarity_set[i];
    setup_plot_D_basic_variables(similarity);
    use_plot_over_p = true;
    calculate_linear_model_flow_vectors();
    plot_linear_model_flow_vectors();
  end
end

print("End of loop over similarity matrices\n");
