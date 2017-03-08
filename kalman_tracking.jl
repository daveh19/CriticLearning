using PyPlot
using Distributions

function generate_two_reward_sequences(sequence_length = 100, noise_sigma = 0.1, switch_point = 0, reward_contingencies = [0.8; 0.8])
  sequence_id = zeros(Int,sequence_length,1);
  sequence_value = zeros(sequence_length,1);
  element_count = zeros(Int,2,1);

  mean_values = [0.9 0.9; -0.7 0.3]'
  reward_probabilities = reward_contingencies;
  for i = 1:sequence_length
    # this is the task association, which is a random sequence
    sequence_id[i] = (rand(Uniform(0,1)) < 0.5 ? 1 : 2);

    if i < switch_point
      position_in_contingencies_sequence = 1;
    else
      position_in_contingencies_sequence = 2;
    end

    # sequence value is a mean + gaussian noise
    # sequence_value[i] = mean_values[sequence_id[i],position_in_contingencies_sequence];
    # sequence_value[i] += rand(Normal(0,1)) .* noise_sigma;

    # sequence value is +/-1 (reward) based on probability correct (reward_probabilities)
    sequence_value[i] = (rand(Uniform(0,1)) < reward_probabilities[sequence_id[i],position_in_contingencies_sequence] ? +1 : -1);


    element_count[sequence_id[i]] += 1;
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
  # plot(time1, y1, "r", linewidth=2, label="Task 1 R signal")
  # plot(time2, y2, "b", linewidth=2, label="Task 2 R signal")
  # scatter(linspace(1,sequence_length,sequence_length), sequence_value);
  scatter(time1, y1, color="r", label="Task 1 R signal")
  scatter(time2, y2, color="g", label="Task 1 R signal")

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
  no_data_points = 6000;
  switch_contingencies_point = 3001;
  tracking_updated_reward_estimates = zeros(2,no_data_points); # for plotting!
  tracking_corrected_reward_estimates = zeros(2,no_data_points);
  #tracking_updated_error_covariance = zeros(2,no_data_points);
  # tracking_corrected_error_covariance = zeros(2,no_data_points);

  # Kalman filter parameters
  process_noise_model = [1. 0.01; 0.01 1.]; #[10.01 1.10; 1.10 10.01]; # process noise
  sigma_1_sq = sigma_2_sq = 1000.0; #150.0; # observation noise
  observation_noise_model = [sigma_1_sq 0 ; 0 sigma_2_sq];

  initial_covariance = 0.999;
  tau = 3.;

  # Data generation
  data_gen_noise = 0.2;
  reward_contingencies = [0.9 0.9; 0.3 0.6]';

  k_dict = kalman_initialise(initial_covariance);
  data_matrix = generate_two_reward_sequences(no_data_points, data_gen_noise, switch_contingencies_point, reward_contingencies);

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

  # plot contingencies up until switch point
  plot(linspace(1,switch_contingencies_point,switch_contingencies_point), ones(switch_contingencies_point,1)*reward_contingencies[1,1]*2 - 1, "c")
  plot(linspace(1,switch_contingencies_point,switch_contingencies_point), ones(switch_contingencies_point,1)*reward_contingencies[2,1]*2 - 1, "m")
  # plot contingencies following switch point
  plot(linspace(switch_contingencies_point,no_data_points, no_data_points-switch_contingencies_point), ones(no_data_points-switch_contingencies_point,1)*reward_contingencies[1,2]*2 - 1, "c")
  plot(linspace(switch_contingencies_point,no_data_points, no_data_points-switch_contingencies_point), ones(no_data_points-switch_contingencies_point,1)*reward_contingencies[2,2]*2 - 1, "m")

  # play: for split processes 0 gets presented 50% of the time, subtract this from the running average
  plot(linspace(1,switch_contingencies_point,switch_contingencies_point), (ones(switch_contingencies_point,1)*reward_contingencies[1,1]*2 - 1) /2.0, "c")
  plot(linspace(1,switch_contingencies_point,switch_contingencies_point), (ones(switch_contingencies_point,1)*reward_contingencies[2,1]*2 - 1) /2.0, "m")
  # plot contingencies following switch point
  plot(linspace(switch_contingencies_point,no_data_points, no_data_points-switch_contingencies_point), (ones(no_data_points-switch_contingencies_point,1)*reward_contingencies[1,2]*2 - 1) /2.0, "c")
  plot(linspace(switch_contingencies_point,no_data_points, no_data_points-switch_contingencies_point), (ones(no_data_points-switch_contingencies_point,1)*reward_contingencies[2,2]*2 - 1) /2.0, "m")


  # plot data points and two kalman following processes
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
  task_id = round(Int,data_row[1]);
  reward_value = data_row[2];
  observed_reward = zeros(2,1); # default value is zero
  observed_reward[task_id] = reward_value;
  # using a local copy of the observation noise allows us to play with zeroing and infinite entries
  local_observation_noise_model = deepcopy(observation_noise_model);

  # Debugging: try combining monitors to make a single (dual) prediction for testing
  # observed_reward[1] = reward_value;
  # observed_reward[2] = reward_value;

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
