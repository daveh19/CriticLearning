using PyPlot
using Distributions

function generate_two_reward_sequences(sequence_length = 100, noise_sigma = 0.1, switch_point = 0)
  sequence_id = zeros(sequence_length,1);
  sequence_value = zeros(sequence_length,1);
  element_count = zeros(2,1);

  mean_values = [0.9 0.9; -0.7 0.3]'
  for i = 1:sequence_length
    sequence_id[i] = round(Int,(rand(Uniform(0,1)) < 0.5 ? 1 : 2));
    if i < switch_point
      sequence_value[i] = mean_values[round(Int,sequence_id[i]),1]; #(sequence_id[i]*4) - 1.5;
    else
      sequence_value[i] = mean_values[round(Int,sequence_id[i]),2]; #(sequence_id[i]*2) - 2;
    end

    sequence_value[i] += rand(Normal(0,1)) .* noise_sigma;
    # sequence_value[i] = 3.0 + rand(Normal(0,1)) .* noise_sigma;
    element_count[round(Int,sequence_id[i])] += 1;
  end


  print("els: ", element_count, "\n", sequence_id)
  # figure()
  # plot(sequence_value, "g", linewidth=2, label="Combined R signals")
  figure()
  # x1 = linspace(1,100, element_count[1]);
  # x2 = linspace(1,100, element_count[2]);
  time = linspace(1,sequence_length, sequence_length);
  time1 = time[sequence_id .== 1];
  time2 = time[sequence_id .== 2];
  y1 = sequence_value[sequence_id .== 1];
  y2 = sequence_value[sequence_id .== 2];
  plot(time1, y1, "r", linewidth=2, label="Task 1 R signal")
  plot(time2, y2, "b", linewidth=2, label="Task 2 R signal")
  scatter(linspace(1,sequence_length,sequence_length), sequence_value);

  return [sequence_id sequence_value];
end

function kalman_initialise(c = 0.99)
  reward_estimate = [0.0; 0.0];
  error_covariance = [1 c; c 1];
  k_dict = Dict("corrected_reward_estimate" => reward_estimate, "corrected_error_covariance" => error_covariance);
  return k_dict;
end

function kalman_host()
  # Basic simulation tracking stuff
  srand(1);
  no_data_points = 3000;
  tracking_updated_reward_estimates = zeros(2,no_data_points); # for plotting!
  tracking_corrected_reward_estimates = zeros(2,no_data_points);
  #tracking_updated_error_covariance = zeros(2,no_data_points);
  # tracking_corrected_error_covariance = zeros(2,no_data_points);

  # Kalman filter parameters
  process_noise_model = [1. 0.1; 0.1 1.]; #[10.01 1.10; 1.10 10.01]; # process noise
  sigma_1_sq = sigma_2_sq = 100.0; #150.0; # observation noise
  observation_noise_model = [sigma_1_sq 0 ; 0 sigma_2_sq];

  initial_covariance = 0.999;
  tau = 5.;

  # Data generation
  data_gen_noise = 0.2;

  k_dict = kalman_initialise(initial_covariance);
  data_matrix = generate_two_reward_sequences(no_data_points, data_gen_noise, 3001);

  for i = 1:no_data_points
    print("\ntrial: ", i)
    kalman_update_prediction(k_dict, process_noise_model, tau)

    kalman_update_correction(k_dict, data_matrix[i,:], observation_noise_model)

    tracking_updated_reward_estimates[:,i] = k_dict["updated_reward_estimate"];
    tracking_corrected_reward_estimates[:,i] = k_dict["corrected_reward_estimate"];
    print("\n Predicted reward: \t", k_dict["updated_reward_estimate"])
    print("\n Corrected reward: \t", k_dict["corrected_reward_estimate"])
    print("\n Predicted covariance: \t", k_dict["updated_error_covariance"])
    print("\n Corrected covariance: \t", k_dict["corrected_error_covariance"])
  end

  figure() # plot reward estimates (predictions)
  plot(linspace(1,no_data_points,no_data_points), tracking_updated_reward_estimates[1,:], "r", linewidth=3, label="Kalman reward 1 estimates")
  plot(linspace(1,no_data_points,no_data_points), tracking_updated_reward_estimates[2,:], "g", linewidth=2, label="Kalman reward 2 estimates")
  scatter(linspace(1,no_data_points,no_data_points), data_matrix[:,2], color="b", label="Data points")
  # figure() # plot covariance estimates (predictions) [ugh matrices!]
  # plot(linspace(1,100,no_data_points), tracking_updated_error_covariance[1,:], "r", linewidth=3, label="Kalman covariance 1 estimates")
  # plot(linspace(1,100,no_data_points), tracking_updated_error_covariance[2,:], "g", linewidth=2, label="Kalman covariance 2 estimates")

  # print("", k_dict);
  # return data_matrix;
  return k_dict;
end

function kalman_update_prediction(k_dict, process_noise_model, tau)
  # modified to use (1-1/tau) in the matrix A
  #tau = 5.;
  A = (1.-(1./tau)) * eye(2);

  # play with actually having a lossy update (this is mathematically incorrect, and pointless in actual use)
  #k_dict["updated_reward_estimate"] = A * k_dict["corrected_reward_estimate"];
  # reward estimate stays the same as the expectation of the noise term corresponds to an extra (1/tau) entry in the matrix
  k_dict["updated_reward_estimate"] = k_dict["corrected_reward_estimate"];

  k_dict["updated_error_covariance"] = (A * k_dict["corrected_error_covariance"] * transpose(A)) + process_noise_model;
end


function kalman_update_correction(k_dict, data_row, observation_noise_model)
  task_id = round(Int, data_row[1]);
  reward_value = data_row[2];
  observed_reward = zeros(2,1);
  observed_reward[task_id] = reward_value;
  # using a local copy of the observation noise allows us to play with zeroing and infinite entries
  local_observation_noise_model = deepcopy(observation_noise_model);

  # Debugging: try combining monitors to make a single (dual) prediction for testing
  observed_reward[1] = reward_value;
  observed_reward[2] = reward_value;

  # Code for identifying row of observation_noise_model or K to modify to account
  #   for effectively infinite variance in the non-presented dimension.
  if task_id == 1
    non_task_id = 2;
  elseif task_id == 2
    non_task_id = 1;
  else
    print("\nERROR: this shouldn't happen\n");
  end

  print("\n Actual observed reward: \t", observed_reward);

  ## First approach: modify observation_noise_model directly
  # local_observation_noise_model[non_task_id,non_task_id] = Inf
  # K = k_dict["updated_error_covariance"] * inv(k_dict["updated_error_covariance"] + local_observation_noise_model);
  ## Second approach: modify K instead
  K = k_dict["updated_error_covariance"] * inv(k_dict["updated_error_covariance"] + observation_noise_model);
  print("\n K1 ", K);
  # Now manually set non task_id row of K to zeros
  #K[:,non_task_id] = 0.0;
  print("\n K2 ", K);

  # update reward estimate according to observation
  k_dict["corrected_reward_estimate"] = k_dict["updated_reward_estimate"] + K * (observed_reward - k_dict["updated_reward_estimate"]);

  # update covariance matrix according to observation
  # Updating rule for Optimal Kalman gain function (not what we have with modified K)
  # k_dict["corrected_error_covariance"] = (1 - K) * k_dict["updated_error_covariance"];
  # Full updating of covariance rule
  # local_observation_noise_model[non_task_id,non_task_id] = 0.0;
  print("\n local_observation_noise_model: \t", local_observation_noise_model);
  k_dict["corrected_error_covariance"] = ( (eye(2) - K) * k_dict["updated_error_covariance"] * transpose(eye(2) - K) ) + ( K * local_observation_noise_model * transpose(K) );
end
