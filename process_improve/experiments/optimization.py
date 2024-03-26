"""
IDEAS from earlier code written in 2015/2016.
RSM optimization.



    function out = optimization_caller()

% Leave this part intact. Copy it, and paste it below, altering the values.
% Repeat that once per factor.
template = struct;
template.descriptive_name = 'Parameter name';
template.argument_name = 'Name as used in a function';
template.parameter_type = 'Categorical | Numerical';
template.default_value = 'Text or numeric value';
template.levels = 'A cell array of level names, only for categorical factor. Numerical: ignored';
template.level_mapping = 'A vector of numeric values that are used to map the levels, e.g. [1, 2, 3]. Numerical: ignored';
template.starting_lower_bound = 'A numeric factor will be evaluated starting at this low value. Categorical: ignored';
template.starting_upper_bound = 'A numeric factor will be evaluated starting at this upper value, Categorical: ignored';
template.absolute_lower_bound = 'A numeric factor cannot go lower than this. Categorical: ignored';
template.absolute_upper_bound = 'A numeric factor cannot go above this. Categorical: ignored';


% 0. OVERALL settings
% ========================
settings = struct;

% This is your function that will be called to run the optimization.
% It will have only 1 input, a structure. That structure will have the names
% of your factors. If you have specified ``factors`` (see below) as
%   factors.A
%   factors.my_variable
%   etc
%
% that is what will be passed into the function.
%
% This will almost certainly mean that you must write another MATLAB file
% function which receives this input, and translates it into the right form to
% actually run the experiment/simulation.
%
% The function must return a single scalar value.
settings.expt_function = @my_function_call;
settings.expt_function__n_outputs = 3;

% Number of starting runs to kick-off the D-optimal design
settings.n_start_runs = 20;

% If you start with a full factorial, then the above setting of
% ``n_start_runs`` is ignored. It will run the full set of potentially many,
% many runs
settings.start_full_factorial = 0;



% 1. SPECIFY parameters
% ========================
% Reference spectrum type
A = template;
A.descriptive_name = 'Reference type';
A.argument_name = 'reference_type';
A.parameter_type = 'Categorical';
A.default_value = 'average';
A.levels = {'average', 'median', 'max', 'average2'};
A.level_mapping = [1, 2, 3, 4];

% Number of intervals
B = template;
B.descriptive_name = 'Number of intervals';
B.argument_name = 'number_intervals';
B.parameter_type = 'Numerical';
B.default_value = 50;
B.starting_lower_bound = 60;
B.starting_upper_bound = 120;
B.absolute_lower_bound = 10;
B.absolute_upper_bound = 100;

% Number of intervals: as a categorical variable
B = template;
B.descriptive_name = 'Number of intervals';
B.argument_name = 'number_intervals';
B.parameter_type = 'Categorical';
B.default_value = 50;
B.levels = {50, 70, 90, 110};
B.level_mapping = [50, 70, 90, 110];


% Filling mode
C = template;
C.descriptive_name = 'Filling mode';
C.argument_name = 'filling_mode';
C.parameter_type = 'Categorical';
C.default_value = 0;
C.levels = {0, 1};
C.level_mapping = [0, 1];


% Coshift preprocessing
D = template;
D.descriptive_name = 'Coshift';
D.argument_name = 'coshift_preprocessing';
D.parameter_type = 'Categorical';
D.default_value = 0;
D.levels = {0, 1};
D.level_mapping = [0, 1];

% Max allowed shift for the Co-shift preprocessing
E = template;
E.descriptive_name = 'Maximum allowed shift';
E.argument_name = 'max_coshift_allowed';
E.parameter_type = 'Numerical';
E.default_value = 50;
E.starting_lower_bound = 40;
E.starting_upper_bound = 100;
E.absolute_lower_bound = 10;
E.absolute_upper_bound = 100;

% Alignment method
F = template;
F.descriptive_name = 'Alignment';
F.argument_name = 'alignment_method';
F.parameter_type = 'Categorical';
F.default_value = 0;
F.levels = {0, 1, 3};
F.level_mapping = [0, 1, 3];


% Maximum shift correction in data
G = template;
G.descriptive_name = 'Maxshift';
G.argument_name = 'maxshift_type';
G.parameter_type = 'Categorical';
G.default_value = 'b';
G.levels = {'b', 'f'};
G.level_mapping = [0, 1];


factors = struct;
factors.A = A;
factors.B = B;
factors.C = C;
factors.D = D;
%factors.E = E;  % <-- not used anyway
factors.F = F;
%factors.G = G;

fields = fieldnames(factors);
n_factors = numel(fields);

% 2. CALCULATE all combinations
% ========================
% Also, while do this, call the ``cordexch`` function, to see what that gives:


type_of_experiment = 'interaction';


starting_set = cell(1, numel(fields));
categorical = zeros(1, n_factors);
levels = zeros(1, n_factors);
for p = 1 : n_factors
    factor = factors.(fields{p});
    if strcmp(factor.parameter_type, 'Numerical')
        this_levels = [factor.starting_lower_bound, factor.starting_upper_bound];
        factors.(fields{p}).current_mins_one = factor.starting_lower_bound;
        factors.(fields{p}).current_plus_one = factor.starting_upper_bound;
        levels(p) = 2;
    elseif strcmp(factor.parameter_type, 'Categorical')
        this_levels = factor.level_mapping;
        levels(p) = numel(factor.level_mapping);
        categorical(p) = 1;
    end
    starting_set{p} = this_levels;
end

% Based on code from https://nl.mathworks.com/matlabcentral/fileexchange/10064-allcomb-varargin
% but modified for our situation.
ii = 1 : 1 : numel(fields);
A = zeros(0, numel(fields)) ;
[A{ii}] = ndgrid(starting_set{ii});
combinations = reshape(cat(numel(fields) + 1, A{:}), [], numel(fields));

% Coordinate exchange experiments:
coord_exch_full_set = cordexch(n_factors, size(combinations, 1), type_of_experiment, ...
         'categorical', find(categorical), 'levels', levels, 'display', 'off');



% 3. SELECT the subset to run
% =========================

% TODO: select a Plackett-Burham subset

% Use the D-optimal approach to select a subset of the full set in
% ``combinations`` as a starting candidate set.

if settings.start_full_factorial
    start_set = combinations;

else
    start_set = rowexch(n_factors, settings.n_start_runs, type_of_experiment, ...
         'categorical', find(categorical), 'levels', levels, 'display', 'off');

    % Original method: use "candexch" to select, but this ignores all the
    % levels of a categorical factor, and only picks the extremes (first and
    % last levels of such a factor).
    %subset = candexch(coord_exch_full_set, settings.n_start_runs, 'display', 'off', ...
    %              'tries', 100, 'maxiter', 1000);
    %start_set = combinations(subset, :);
end

% Remove the duplicates
start_set = unique(start_set, 'rows', 'stable');


% 4. RUN the experiments
n_runs = size(start_set, 1);
outputs = zeros(n_runs, settings.expt_function__n_outputs);
for k = 1:n_runs
    disp(['Run number: ', num2str(k), ' of ', num2str(n_runs)])
    fn_input = translate_coded_to_realworld_structure(start_set(k, :), factors);
    fn_input
    %try
    outputs(k, :) = settings.expt_function(fn_input);
    %catch
    %
    %    outputs(k, :) = NaN;
    %end
    outputs(k, :)
end
out = [start_set outputs];

% 5. SHOW results
%------------------------------
figure;
subplot(settings.expt_function__n_outputs, n_factors, 1)
count = 1;
for yout = 1:settings.expt_function__n_outputs
    for p = 1:n_factors
        factor = factors.(fields{p});
        subplot(settings.expt_function__n_outputs, n_factors, count)
        count = count + 1;
        xs = start_set(:, p);
        ys = outputs(:, yout);
        plot(xs, ys, 'o')
        hold on
        b = regress(ys, [ones(size(xs, 1), 1), xs]);
        plot(xs, b(1) + b(2) .* xs, 'k-', 'linewidth', 2)
        grid on
        title(factor.descriptive_name)
    end

    fprintf('For output %d: the highest and lowest values are:\n', yout)
    [min_y, idx_min] = min(ys);
    fprintf('Low : y = %f; settings=\n', min_y)
    disp(translate_coded_to_realworld_structure(start_set(idx_min, :), factors))


    [max_y, idx_max] = max(ys);
    fprintf('High: y = %f; settings=\n', max_y)
    disp(translate_coded_to_realworld_structure(start_set(idx_max, :), factors))
end

a=2;


% 6. IDENTIFY next optimization
%------------------------------

function output = translate_coded_to_realworld_structure(input, factors)
% Translates a numeric vector, ``input``, where each entry is the numeric
% factor settings [which may include categorical values], back to the mixed
% numeric and string representation. Returns a structure.
%   range = upper_bound - lower_bound
%   center_value = 0.5 * (upper_bound + lower_bound)
%   coded_value = (actual_value - center_value) / (0.5 * range)
%   actual_value = coded_value * (0.5 * range) + lower_bound
fields = fieldnames(factors);
output = struct;
for p = 1 : numel(fields)
    factor = factors.(fields{p});
    if strcmp(factor.parameter_type, 'Numerical')

        % To get exact matching, without roundoff
        if input(p) == -1
            value = factor.current_mins_one;
        elseif input(p) == +1
            value = factor.current_plus_one;
        else
            range = (factor.current_plus_one - factor.current_mins_one);
            center_value = 0.5*(factor.current_plus_one + factor.current_mins_one);
            value = input(p) * (0.5 * range) + center_value;
        end

    elseif strcmp(factor.parameter_type, 'Categorical')
        value = factor.levels{input(p)};
        % Do NOT remove the 'find' part here, or try to simplify it. It is
        % correct as it is.
        %temp = %(factor.levels(find(factor.level_mapping == input(p))));
        %value = temp{1};
    end
    output.(factor.argument_name) = value;
end
"""
