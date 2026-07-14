function official_pmbm_recurring_birth(dataset_path, output_path, mtt_root, tcl_root)
% Evaluate the official Gaussian PMBM implementation on the shared benchmark.

pkg load statistics;
addpath(fullfile(mtt_root, 'PMBM filter'));
addpath(fullfile(mtt_root, 'GOSPA code'));
addpath(fullfile(mtt_root, 'Assignment'));
addpath(fullfile(tcl_root, 'Assignment_Algorithms', 'k-Best_2D_Assignment'));
addpath(fullfile(tcl_root, 'Sample_Code', '2D_Assignment'));
addpath(fullfile(tcl_root, 'Container_Classes'));
load(dataset_path);

T_pruning = 0.0001;
T_pruningPois = 1e-5;
Nhyp_max = 200;
gating_threshold = 20;
existence_threshold = 1e-5;
existence_estimation_threshold = 0.4;
c_gospa = 10;
scan_count = size(target_states, 3);
seed_count = size(target_states, 4);
intensity_clutter = clutter_rate / surveillance_area;
tracker_names = {'pmbm_broad', 'pmbm_oracle'};

output_dir = fileparts(output_path);
if ~isempty(output_dir) && ~exist(output_dir, 'dir')
    mkdir(output_dir);
end
file_id = fopen(output_path, 'w');
fprintf(file_id, ['seed,tracker,scans,total_births,total_confirmed_births,' ...
    'rms_gospa,rms_localization,rms_missed,rms_false,early_rms_gospa,' ...
    'late_rms_gospa,mean_cardinality_error,mean_false_targets,' ...
    'mean_missed_targets,final_birth_atoms,recurrent_birth_atoms,' ...
    'spurious_birth_atoms,birth_region_error,runtime_seconds\n']);

for tracker_index = 1:length(tracker_names)
    tracker_name = tracker_names{tracker_index};
    if strcmp(tracker_name, 'pmbm_broad')
        weights_b = birth_rate;
        means_b = base_mean;
        covs_b = base_covariance;
    else
        weights_b = repmat(birth_rate / size(oracle_means, 2), 1, size(oracle_means, 2));
        means_b = oracle_means;
        covs_b = oracle_covariances;
    end

    for seed_index = 1:seed_count
        started = tic;
        filter_pred.weightPois = weights_b;
        filter_pred.meanPois = means_b;
        filter_pred.covPois = covs_b;
        filter_pred.tracks = cell(0, 1);
        filter_pred.globHyp = [];
        filter_pred.globHypWeight = [];

        powered_gospa = zeros(1, scan_count);
        powered_localization = zeros(1, scan_count);
        powered_missed = zeros(1, scan_count);
        powered_false = zeros(1, scan_count);
        cardinality_error = zeros(1, scan_count);
        false_count = zeros(1, scan_count);
        missed_count = zeros(1, scan_count);

        for scan = 1:scan_count
            measurement_count = measurement_counts(scan, seed_index);
            z = measurement_values(:, 1:measurement_count, scan, seed_index);
            filter_upd = PoissonMBMtarget_update(
                filter_pred, z, H, R, p_d, scan, gating_threshold, ...
                intensity_clutter, Nhyp_max);
            X_estimate = PoissonMBMtarget_estimate1(
                filter_upd, existence_estimation_threshold);
            if isempty(X_estimate)
                estimate_positions = zeros(2, 0);
            else
                estimate_states = reshape(X_estimate, 4, []);
                estimate_positions = estimate_states(1:2, :);
            end
            alive = logical(target_alive(:, scan, seed_index));
            truth_positions = target_states(1:2, alive, scan, seed_index);
            truth_positions = reshape(truth_positions, 2, []);
            [distance, ~, decomposition] = GOSPA(
                truth_positions, estimate_positions, 2, c_gospa, 2);
            powered_gospa(scan) = distance^2;
            powered_localization(scan) = decomposition.localisation;
            powered_missed(scan) = decomposition.missed;
            powered_false(scan) = decomposition.false;
            cardinality_error(scan) = abs(size(truth_positions, 2) - size(estimate_positions, 2));
            missed_count(scan) = decomposition.missed / (c_gospa^2 / 2);
            false_count(scan) = decomposition.false / (c_gospa^2 / 2);

            filter_upd = PoissonMBMtarget_pruning(
                filter_upd, T_pruning, T_pruningPois, Nhyp_max, existence_threshold);
            filter_pred = PoissonMBMtarget_pred(
                filter_upd, F, Q, p_s, weights_b, means_b, covs_b);
        end

        split = min(32, scan_count);
        rms_gospa = sqrt(mean(powered_gospa));
        rms_localization = sqrt(mean(powered_localization));
        rms_missed = sqrt(mean(powered_missed));
        rms_false = sqrt(mean(powered_false));
        early_rms_gospa = sqrt(mean(powered_gospa(1:split)));
        if split < scan_count
            late_rms_gospa = sqrt(mean(powered_gospa(split + 1:end)));
        else
            late_rms_gospa = 0;
        end
        runtime_seconds = toc(started);
        fprintf(file_id, ['%d,%s,%d,NaN,NaN,%.12g,%.12g,%.12g,%.12g,' ...
            '%.12g,%.12g,%.12g,%.12g,%.12g,NaN,NaN,NaN,NaN,%.12g\n'], ...
            seed_values(seed_index), tracker_name, scan_count, rms_gospa, ...
            rms_localization, rms_missed, rms_false, early_rms_gospa, ...
            late_rms_gospa, mean(cardinality_error), mean(false_count), ...
            mean(missed_count), runtime_seconds);
        fprintf('%s seed %d/%d: RMS GOSPA %.3f\n', ...
            tracker_name, seed_index, seed_count, rms_gospa);
    end
end

fclose(file_id);
fprintf('wrote %s\n', output_path);
end
