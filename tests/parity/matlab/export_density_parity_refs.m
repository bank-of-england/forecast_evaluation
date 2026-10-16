% export_density_parity_refs.m
% Exports MATLAB reference outputs for the Python density-parity tests
% (tests/parity). Run from MATLAB; writes .mat files (-v7) into
% tests/data/matlab_parity. Requires Statistics and Optimization toolboxes.
%
% Usage:
%   run('N:\Users\333415\Repos\forecast_evaluation\tests\parity\matlab\export_density_parity_refs.m')

%% Section 0: setup
here   = fileparts(mfilename('fullpath'));
root   = fileparts(fileparts(fileparts(here)));
mtp    = fullfile(root, 'notebooks', 'MTP', 'IaR_GaR_April_2026');
addpath(genpath(fullfile(mtp, 'functions')));
outdir = fullfile(root, 'tests', 'data', 'matlab_parity');
if ~exist(outdir, 'dir')
    mkdir(outdir);
end
rng(0, 'twister');
written = {};
fprintf('Exporting parity references to %s\n', outdir);

%% Section A: distributions (skew-t, TPN)
try
    [Lg, Sg, Ag, Dg] = ndgrid([0 2], [0.5 1.5], [-3 0 2], [5 10 30]);
    skt_params = [Lg(:) Sg(:) Ag(:) Dg(:)];   % columns: loc, scale, shape, df
    skt_x   = linspace(-5, 5, 21);
    skt_tau = [0.01 0.05 0.25 0.5 0.75 0.95 0.99];
    nP = size(skt_params, 1);
    skt_pdf = nan(nP, numel(skt_x));
    skt_cdf = nan(nP, numel(skt_x));
    skt_q   = nan(nP, numel(skt_tau));
    for i = 1:nP
        p = skt_params(i, :);
        skt_pdf(i, :) = reshape(dskt(skt_x, p(1), p(2), p(3), p(4)), 1, []);
        skt_cdf(i, :) = reshape(pskt(skt_x, p(1), p(2), p(3), p(4)), 1, []);
        skt_q(i, :)   = reshape(qskt(skt_tau, p(1), p(2), p(3), p(4)), 1, []);
    end

    % TPN pdf / inverse cdf: columns m, s1, s2
    tpn_params = [0 1 1; 2 0.5 1.5; -1 1.2 0.6; 2 1 2];
    tpn_x   = linspace(-4, 6, 21);
    tpn_tau = skt_tau;
    nT = size(tpn_params, 1);
    tpn_pdf = nan(nT, numel(tpn_x));
    tpn_q   = nan(nT, numel(tpn_tau));
    for i = 1:nT
        p = tpn_params(i, :);
        tpn_pdf(i, :) = reshape(ftp(tpn_x, p(1), p(2), p(3)), 1, []);
        tpn_q(i, :)   = reshape(two_part_normal_inverse_cdf(tpn_tau, p(1), p(2), p(3)), 1, []);
    end

    % Moment conversions: columns mean, mode, unc (mean ~= mode to avoid 0/0 in stdtogam)
    mom_inputs = [2.5 2.0 1.0; 1.5 2.0 0.8; 0.3 0.1 2.0; -1.0 -0.5 1.5];
    nM = size(mom_inputs, 1);
    mom_gam  = nan(nM, 1);   % stdtogam(mean, mode, unc)
    mom_sig  = nan(nM, 1);   % mom2g(unc, gam)
    mom_parm = nan(nM, 1);   % momtopar(mean, mode, mom_sig) -> [parm, sig1, sig2]
    mom_sig1 = nan(nM, 1);
    mom_sig2 = nan(nM, 1);
    for i = 1:nM
        mom_gam(i) = stdtogam(mom_inputs(i, 1), mom_inputs(i, 2), mom_inputs(i, 3));
        mom_sig(i) = mom2g(mom_inputs(i, 3), mom_gam(i));
        [mom_parm(i), mom_sig1(i), mom_sig2(i)] = momtopar(mom_inputs(i, 1), mom_inputs(i, 2), mom_sig(i));
    end

    f = fullfile(outdir, 'refs_distributions.mat');
    save(f, 'skt_params', 'skt_x', 'skt_tau', 'skt_pdf', 'skt_cdf', 'skt_q', ...
        'tpn_params', 'tpn_x', 'tpn_tau', 'tpn_pdf', 'tpn_q', ...
        'mom_inputs', 'mom_gam', 'mom_sig', 'mom_parm', 'mom_sig1', 'mom_sig2', '-v7');
    written{end+1} = f;
catch ME
    warning('Section A failed: %s', ME.message);
end

%% Section B: fitting (skew-t, TPN, QR_sm)
try
    % Exact decimal levels so fit_skewt_to_quantiles_all's (quantiles==0.75) lookup succeeds
    qlev = round((1:19) * 5) / 100;
    nPick = 10;
    R = struct();
    R.qlev = qlev;
    series = {'infl', 'inflation'; 'gdp', 'gdp'};
    for s = 1:size(series, 1)
        tag = series{s, 1};
        src = load(fullfile(mtp, 'Outputs', sprintf('predicted_quantiles_%s_OOS.mat', series{s, 2})), 'pred_q');
        pq  = src.pred_q;   % nOrig x 19 x H
        assert(size(pq, 2) == numel(qlev), 'pred_q(%s) has %d quantile columns', tag, size(pq, 2));
        [nO, ~, H] = size(pq);
        fprintf('Section B: %s pred_q is %d x %d x %d\n', tag, nO, size(pq, 2), H);

        % Deterministic spread of (origin, horizon) candidates; keep first nPick fully observed
        oCand = unique(round(linspace(1, nO, 4 * nPick)));
        hCand = unique(round(linspace(1, H, nPick)));
        picks = zeros(0, 2);
        for c = 1:numel(oCand)
            o = oCand(c);
            h = hCand(mod(c - 1, numel(hCand)) + 1);
            if all(isfinite(pq(o, :, h)))
                picks(end+1, :) = [o h]; %#ok<AGROW>
            end
            if size(picks, 1) == nPick, break; end
        end
        nK = size(picks, 1);

        targets   = nan(nK, numel(qlev));
        qi_params = nan(nK, 4);  qi_fitq = nan(nK, numel(qlev));  qi_obj = nan(nK, 1);  qi_time = nan(nK, 1);
        fa_params = nan(nK, 4);  fa_fitq = nan(nK, numel(qlev));  fa_obj = nan(nK, 1);  fa_time = nan(nK, 1);
        tp_params = nan(nK, 3);  tp_fitq = nan(nK, numel(qlev));  tp_obj = nan(nK, 1);  tp_time = nan(nK, 1);
        for k = 1:nK
            tq = reshape(pq(picks(k, 1), :, picks(k, 2)), 1, []);
            targets(k, :) = tq;

            % MTP QuantilesInterpolation (fmincon on 5 quantiles, df rounded)
            tic;
            [lc, sc, sh, df] = QuantilesInterpolation(tq, qlev);
            qi_time(k) = toc;
            qi_params(k, :) = [lc sc sh df];
            qi_fitq(k, :) = reshape(qskt(qlev, lc, sc, sh, df), 1, []);
            qi_obj(k) = sum((qi_fitq(k, :) - tq).^2);

            % fit_skewt_to_quantiles_all (lsqnonlin over df = 5..30)
            tic;
            [fp, fq] = fit_skewt_to_quantiles_all(reshape(tq, 1, 1, [], 1), qlev, qlev, 1, 1, 30);
            fa_time(k) = toc;
            fa_params(k, :) = reshape(fp(1, :, 1, 1), 1, []);
            fa_fitq(k, :) = reshape(fq(1, :, 1, 1), 1, []);
            fa_obj(k) = sum((reshape(qskt(qlev, fa_params(k, 1), fa_params(k, 2), fa_params(k, 3), fa_params(k, 4)), 1, []) - tq).^2);

            % Two-piece normal (fmincon on q25/q50/q75)
            tic;
            [tpar, tfq] = fit_tpn_to_quantiles_SM(tq, qlev);
            tp_time(k) = toc;
            tp_params(k, :) = tpar;
            tp_fitq(k, :) = reshape(tfq, 1, []);
            tp_obj(k) = sum((tp_fitq(k, :) - sort(tq)).^2);
        end

        R.([tag '_picks'])     = picks;   % [origin, horizon] (1-based)
        R.([tag '_targets'])   = targets;
        R.([tag '_qi_params']) = qi_params;  R.([tag '_qi_fitq']) = qi_fitq;
        R.([tag '_qi_obj'])    = qi_obj;     R.([tag '_qi_time']) = qi_time;
        R.([tag '_fa_params']) = fa_params;  R.([tag '_fa_fitq']) = fa_fitq;
        R.([tag '_fa_obj'])    = fa_obj;     R.([tag '_fa_time']) = fa_time;
        R.([tag '_tp_params']) = tp_params;  R.([tag '_tp_fitq']) = tp_fitq;
        R.([tag '_tp_obj'])    = tp_obj;     R.([tag '_tp_time']) = tp_time;

        if s == 1
            infl_targets_for_qrsm = targets(1:min(5, nK), :);
        end
    end

    % Diagnostic: df distribution of the production skew-t fits
    sk = load(fullfile(mtp, 'Outputs', 'sktparam', 'skewtparam_inflation_OOS.mat'), 'df_skt');
    dfv = sk.df_skt(isfinite(sk.df_skt));
    [R.df_skt_values, ~, ic] = unique(dfv(:));
    R.df_skt_counts = accumarray(ic, 1);

    % QR_sm semi-parametric draws (rng(123) before each call)
    nQ = size(infl_targets_for_qrsm, 1);
    R.qrsm_inputs = infl_targets_for_qrsm;
    R.qrsm_mean = nan(nQ, 1);  R.qrsm_std = nan(nQ, 1);  R.qrsm_skew = nan(nQ, 1);
    R.qrsm_cdf_at_q = nan(nQ, numel(qlev));
    for k = 1:nQ
        rng(123);
        draws = QR_sm(infl_targets_for_qrsm(k, :), qlev);
        draws = draws(:);
        R.qrsm_mean(k) = mean(draws);
        R.qrsm_std(k)  = std(draws);
        R.qrsm_skew(k) = skewness(draws, 0);
        qs_sorted = sort(infl_targets_for_qrsm(k, :));
        for j = 1:numel(qlev)
            R.qrsm_cdf_at_q(k, j) = mean(draws <= qs_sorted(j));
        end
    end

    f = fullfile(outdir, 'refs_fitting.mat');
    save(f, '-struct', 'R', '-v7');
    written{end+1} = f;
catch ME
    warning('Section B failed: %s', ME.message);
end

%% Section C: calibration tests (Knueppel, Mitchell-Weale, RS, GMM)
try
    R = struct();
    sPit = RandStream('mt19937ar', 'Seed', 1);
    pit_unif   = rand(sPit, 120, 1);
    pit_beta22 = betainv(rand(sPit, 120, 1), 2, 2);
    pit_pow13  = rand(sPit, 80, 1).^1.3;
    pit_pow13([10 40 70]) = NaN;
    cases = {'unif', pit_unif; 'beta22', pit_beta22; 'pow13', pit_pow13};

    lags = -1;  prewhite = 0;  z_l = 0.05;  z_u = 0.95;
    rvec_rs = linspace(0, 1, 1000);
    bootMC  = 200;
    R.lags = lags;  R.prewhite = prewhite;  R.z_l = z_l;  R.z_u = z_u;
    R.rvec_rs = rvec_rs;  R.bootMC = bootMC;

    sEtaRS = RandStream('mt19937ar', 'Seed', 2);
    for c = 1:size(cases, 1)
        tag = cases{c, 1};
        x = cases{c, 2};
        R.(['pit_' tag]) = x;
        xrow = x';                   % MTP tests take row vectors (they transpose internally)
        xc = x(~isnan(x));

        % nwX on the Knueppel moment matrix
        s_pit = sqrt(12) * (xc - 0.5);
        zmom = [s_pit (s_pit.^2 - 1) (s_pit.^3) (s_pit.^4 - 1.8)];
        [R.(['nw_auto_' tag]), R.(['nw_auto_k_' tag])] = nwX(zmom, 0, -1);
        [R.(['nw_pw_' tag]),   R.(['nw_pw_k_' tag])]   = nwX(zmom, 1, -1);
        [R.(['nw_k3_' tag]),   R.(['nw_k3_k_' tag])]   = nwX(zmom, 0, 3);

        [R.(['knu_stat_' tag]), R.(['knu_pval_' tag])] = alpha0_1234_NW(xrow, lags, prewhite);
        R.(['mw_uncens_' tag]) = MW_alpha0_1234_NW(xrow, lags, prewhite, z_l, z_u);
        R.(['mw_freq_' tag])   = freqTestInCensoredRegion(xrow, lags, prewhite, z_l, z_u);
        R.(['mw_stat_' tag])   = R.(['mw_uncens_' tag]) + R.(['mw_freq_' tag]);
        R.(['mw_pval_' tag])   = 1 - chi2cdf(R.(['mw_stat_' tag]), 5);

        % Rossi-Sekhposyan with injected multipliers
        el_rs = floor(numel(xc)^(1/3));
        eta_rs = randn(sEtaRS, numel(xc) - el_rs + 1, bootMC);
        [cv_rs, st_rs, lg_rs, KS_rs, CVM_rs] = rs_test_eta(xc, rvec_rs, eta_rs);
        R.(['rs_eta_' tag])    = eta_rs;
        R.(['rs_el_' tag])     = el_rs;
        R.(['rs_ks_' tag])     = KS_rs;
        R.(['rs_cvm_' tag])    = CVM_rs;
        R.(['rs_cv_ks_' tag])  = cv_rs.ks;     % [1% 5% 10%]
        R.(['rs_cv_cvm_' tag]) = cv_rs.cvm;
        R.(['rs_ks_logic_' tag])  = double(lg_rs.ks_array);
        R.(['rs_cvm_logic_' tag]) = double(lg_rs.cvm_array);
        assert(abs(st_rs.ks_stat - KS_rs) < 1e-12);
    end

    % GMM (Galvao-Mantoan-Mitchell) single-sample path on synthetic T x H PITs
    sG = RandStream('mt19937ar', 'Seed', 3);
    Tg = 60;  Hg = 4;
    gmm_z = rand(sG, Tg, Hg);
    gmm_z(:, Hg) = gmm_z(:, Hg).^1.2;   % mild miscalibration at the last horizon
    rvec_gmm = 0:0.01:1;                % coarser than production 0:0.001:1 to keep closedform_h1_cv cheap
    el_g = floor(Tg^(1/3));
    [Kv_sim_h1, CVMv_sim_h1] = closedform_h1_cv_anchored(rvec_gmm, 200000, 987654);
    gmm_eta = randn(sG, Tg - el_g + 1 + Hg, bootMC);
    gmm_KS = nan(1, Hg);  gmm_CVM = nan(1, Hg);
    for h = 1:Hg
        [gmm_KS(h), gmm_CVM(h)] = rs_stats_only(gmm_z(:, h), rvec_rs);
    end
    [g_rej, g_rej_bonf, g_sup, g_bonf, g_h, g_w, g_wi] = gmm_statistic_eta( ...
        gmm_z, gmm_KS, gmm_CVM, Hg, rvec_gmm, el_g, bootMC, Kv_sim_h1, CVMv_sim_h1, gmm_eta);
    R.gmm_z = gmm_z;  R.gmm_rvec = rvec_gmm;  R.gmm_el = el_g;  R.gmm_eta = gmm_eta;
    R.gmm_Kv_sim_h1 = Kv_sim_h1;  R.gmm_CVMv_sim_h1 = CVMv_sim_h1;
    R.gmm_KS = gmm_KS;  R.gmm_CVM = gmm_CVM;
    R.gmm_w_decreasing = g_w;  R.gmm_wi_increasing = g_wi;
    % Sup statistics, rows: [unweighted; decreasing w; increasing wi], cols: [KS CVM]
    R.gmm_sup_stats = [max(gmm_KS) max(gmm_CVM); max(gmm_KS .* g_w) max(gmm_CVM .* g_w); ...
                       max(gmm_KS .* g_wi) max(gmm_CVM .* g_wi)];
    R.gmm_cv_sup  = g_sup;    % same row/col layout as gmm_sup_stats (95%)
    R.gmm_cv_bonf = g_bonf;   % 2 x H: [KS; CVM]
    R.gmm_cv_h    = g_h;      % 2 x H: per-horizon 95% [KS; CVM]
    R.gmm_rej_ks  = double(g_rej(1, :));       R.gmm_rej_cvm  = double(g_rej(2, :));
    R.gmm_rej_bonf_ks = double(g_rej_bonf(1, :));  R.gmm_rej_bonf_cvm = double(g_rej_bonf(2, :));

    f = fullfile(outdir, 'refs_calibration.mat');
    save(f, '-struct', 'R', '-v7');
    written{end+1} = f;
catch ME
    warning('Section C failed: %s', ME.message);
end

%% Section D: scoring (quantile-grid CRPS/wCRPS, qwps, skew-t value grid, sharpness)
try
    R = struct();
    tau_grid = (1:999)' / 1000;
    v_right  = tau_grid.^2;
    v_left   = (1 - tau_grid).^2;
    crps_from_qgrid   = @(q, y) 2 * mean((double(y < q) - tau_grid) .* (q - y));
    wcrpsR_from_qgrid = @(q, y) 2 * mean(v_right .* (double(y < q) - tau_grid) .* (q - y));
    wcrpsL_from_qgrid = @(q, y) 2 * mean(v_left  .* (double(y < q) - tau_grid) .* (q - y));
    sharp_lev = [0.05 0.25 0.35 0.65 0.75 0.95];
    % widths: [q65-q35, q75-q25, q25-q05, q95-q75, q95-q05]
    sharp_widths = @(q) [q(4)-q(3), q(5)-q(2), q(2)-q(1), q(6)-q(5), q(6)-q(1)];
    J_qwps = 1000;
    nDraw  = 2000;
    u_draw = ((1:nDraw)' - 0.5) / nDraw;   % deterministic "draws" for qwps (no ties -> no jitter)

    R.tau_grid = tau_grid;  R.sharp_lev = sharp_lev;
    R.y_obs = [-2 -0.3 0 0.7 3];
    R.J_qwps = J_qwps;  R.u_draw = u_draw;
    R.norm_params = [0 1; 1 0.5; -0.5 2];                        % [mu sigma]
    R.skt_params  = [0 0.5 -3 5; 2 1.5 2 10; 0 1.5 0 30; 2 0.5 2 5];  % subset of Section A grid
    nY = numel(R.y_obs);

    fams = {'norm', 'skt'};
    for fi = 1:numel(fams)
        fam = fams{fi};
        P = R.([fam '_params']);
        nP = size(P, 1);
        crps_q = nan(nP, nY);  wR_q = nan(nP, nY);  wL_q = nan(nP, nY);
        qw = nan(nP, nY, 6);
        widths = nan(nP, 5);
        for i = 1:nP
            if strcmp(fam, 'norm')
                qfun = @(p) norminv(p, P(i, 1), P(i, 2));
            else
                qfun = @(p) reshape(qskt(p, P(i, 1), P(i, 2), P(i, 3), P(i, 4)), size(p));
            end
            q = qfun(tau_grid);
            xdraw = qfun(u_draw);
            widths(i, :) = sharp_widths(qfun(sharp_lev));
            for j = 1:nY
                y = R.y_obs(j);
                crps_q(i, j) = crps_from_qgrid(q, y);
                wR_q(i, j)   = wcrpsR_from_qgrid(q, y);
                wL_q(i, j)   = wcrpsL_from_qgrid(q, y);
                for QW = 1:6
                    rng(0, 'twister');
                    qw(i, j, QW) = qwps(xdraw, y, J_qwps, QW);
                end
            end
        end
        R.([fam '_crps_qgrid'])   = crps_q;
        R.([fam '_wcrpsR_qgrid']) = wR_q;
        R.([fam '_wcrpsL_qgrid']) = wL_q;
        R.([fam '_qwps'])         = qw;      % nP x nY x QW(1..6)
        R.([fam '_sharp_widths']) = widths;
    end

    % Skew-t value-grid CRPS / right wCRPS / sharpness (forecast_evaluation_inflation.m RASS block)
    q_sharp_lev = [0.05, 0.10, 0.25, 0.35, 0.50, 0.65, 0.75, 0.90, 0.95];
    P = R.skt_params;  nP = size(P, 1);
    vg_crps = nan(nP, nY);  vg_wcrps = nan(nP, nY);  vg_qntl = nan(nP, numel(q_sharp_lev));
    for i = 1:nP
        lc_v = P(i, 1);  sc_v = P(i, 2);  sh_v = P(i, 3);  df_v = P(i, 4);
        nG   = 2000;
        x_lo = lc_v - 15 * sc_v;
        x_hi = lc_v + max(15, 8 * abs(sh_v)) * sc_v;
        xg   = linspace(x_lo, x_hi, nG)';
        pdg  = dskt(xg, lc_v, sc_v, sh_v, df_v);
        cdg  = min(max(cumtrapz(xg, pdg), 0), 1);
        [cu, ia] = unique(cdg, 'stable');
        xu = xg(ia);
        vg_qntl(i, :) = interp1(cu, xu, q_sharp_lev, 'linear', NaN);
        for j = 1:nY
            act = R.y_obs(j);
            vg_crps(i, j)  = trapz(xg, (cdg - double(xg >= act)).^2);
            vg_wcrps(i, j) = 2 * trapz(xg, ...
                cdg.^2 .* (double(xg > act) - cdg) .* (xg - act) .* pdg);
        end
    end
    R.skt_vg_crps = vg_crps;  R.skt_vg_wcrpsR = vg_wcrps;
    R.skt_vg_qntl = vg_qntl;  R.q_sharp_lev = q_sharp_lev;

    % GDP RASS path (forecast_evaluation_growth.m): 19 quantiles -> linear interp/extrap to tau_grid
    quant_qr = (0.05:0.05:0.95)';
    R.interp19_quant = quant_qr;
    R.interp19_qvals = [norminv(quant_qr, 0.5, 1.2), ...
                        reshape(qskt(quant_qr, 2, 1.5, -3, 5), [], 1)];   % 19 x 2
    nC = size(R.interp19_qvals, 2);
    R.interp19_pit = nan(nC, nY);  R.interp19_crps = nan(nC, nY);  R.interp19_wcrpsL = nan(nC, nY);
    R.interp19_qntl = nan(nC, numel(q_sharp_lev));
    for c = 1:nC
        q_vals = R.interp19_qvals(:, c);
        R.interp19_qntl(c, :) = interp1(quant_qr, q_vals, q_sharp_lev, 'linear', 'extrap');
        q_fine = interp1(quant_qr, q_vals, tau_grid, 'linear', 'extrap');
        for j = 1:nY
            act = R.y_obs(j);
            pit = interp1(q_vals, quant_qr, act, 'linear', 'extrap');
            R.interp19_pit(c, j)    = min(max(pit, 0), 1);
            R.interp19_crps(c, j)   = crps_from_qgrid(q_fine, act);
            R.interp19_wcrpsL(c, j) = wcrpsL_from_qgrid(q_fine, act);
        end
    end

    f = fullfile(outdir, 'refs_scoring.mat');
    save(f, '-struct', 'R', '-v7');
    written{end+1} = f;
catch ME
    warning('Section D failed: %s', ME.message);
end

%% Section E: end-to-end evaluation refs (manual step, see comment block at end of file)
fprintf(['\nSection E (manual): refs_eval_inflation.mat / refs_eval_growth.mat must be saved from\n' ...
         'forecast_evaluation_inflation.m (insert after lines 199 and 376) and\n' ...
         'forecast_evaluation_growth.m (insert after lines 261 and 382).\n' ...
         'Copy the snippets from the comment block at the end of:\n  %s\n'], [mfilename('fullpath') '.m']);

%% Summary
fprintf('\nWrote %d reference file(s):\n', numel(written));
for k = 1:numel(written)
    fprintf('  %s\n', written{k});
end

%% ===================== Local functions =====================

function [rs_boot_crit_value, rs_test_stats, rs_logic, KS, CVM] = rs_test_eta(z, rvec, eta)
% mirrors functions/rs_test.m lines 11-35, eta injected (diary/Results.out side effect dropped)
T = size(z,1);
cumcumz = (repmat(z,1,size(rvec,2)) < repmat(rvec,size(z,1),1)) - repmat(rvec,size(z,1),1);
v = sum(cumcumz,1)/sqrt(T);

KS = max(abs(v));
CVM = mean(v.^2);

rs_test_stats.ks_stat  = KS;
rs_test_stats.cvm_stat = CVM;

el = floor(T.^(1/3));
bootMC = 200;

tableboot = CVfinalbootstrapInoue_eta(el, bootMC, z, rvec, eta);
rs_boot_crit_value.ks = tableboot(:,1)';
rs_boot_crit_value.cvm = tableboot(:,2)';
KS_logic = (KS > rs_boot_crit_value.ks);
CVM_logic = (CVM > rs_boot_crit_value.cvm);

rs_logic.ks_array = KS_logic;
rs_logic.cvm_array = CVM_logic;
end

function result = CVfinalbootstrapInoue_eta(el, bootMC, pit, rvec, eta)
% mirrors functions/CVfinalbootstrapInoue.m lines 16-42, eta injected
% eta(:, bootrep) replaces randn(P-el+1,1) drawn at each replication
KSv = zeros(bootMC,1); CVMv = zeros(bootMC,1);

P = size(pit,1);
size_rvec = length(rvec);

for bootrep = 1:bootMC
    z = 1/sqrt(el)*eta(:, bootrep);
    emp_cdf = (repmat(pit,1,size_rvec) <= repmat(rvec,P,1));
    mean_emp_cdf = mean(emp_cdf,1); %#ok<NASGU>

    K_star = zeros(1,size_rvec);
    for j = 1:P-el+1
         K_star = K_star + (P^(-1/2)*z(j,1)).*(sum(emp_cdf(j:j+el-1,:) - repmat(rvec,el,1),1));
    end

    KSv(bootrep,1) = max(abs(K_star'));
    CVMv(bootrep,1) = mean(K_star'.^2);
end
KSv = sort(KSv,'ascend');       cvKv = KSv(bootMC*[0.99 0.95 0.90]);
CVMv = sort(CVMv,'ascend');     cvMv = CVMv(bootMC*[0.99 0.95 0.90]);

result = [cvKv cvMv];
end

function [KS, CVM] = rs_stats_only(z, rvec)
% mirrors functions/rs_test.m lines 11-16 (statistics only, no bootstrap)
T = size(z,1);
cumcumz = (repmat(z,1,size(rvec,2)) < repmat(rvec,size(z,1),1)) - repmat(rvec,size(z,1),1);
v = sum(cumcumz,1)/sqrt(T);
KS = max(abs(v));
CVM = mean(v.^2);
end

function [rej, rej_bonf, tableboot_sup, tableboot_bonf, tableboot_h, w_row, wi] = gmm_statistic_eta(z, Kv, CVMv, H, rvec, el, bootMC, Kv_sim_h1, CVMv_sim_h1, eta)
% mirrors galvao_mantoan_mitchell_test_package/size_statistic_h2_shifted_h1rankfix.m lines 23-62, eta injected
% rej / rej_bonf rows: [KS; CVM], cols: [unweighted, decreasing w, increasing wi]
QVrejvecs=zeros(1,3);
CVMrejvecs=QVrejvecs;
QVrejvecs_bonf=QVrejvecs;
CVMrejvecs_bonf=QVrejvecs;

indmaxKv=find(Kv==max(Kv), 1 );
indmaxCvM=find(CVMv==max(CVMv), 1 );
gstep=(1/H)/((H+1)/2);
wi = zeros(1,H);
for h=1:H
    wi(h)=gstep*h;
end
w=flip(wi',1);

[tableboot_sup, tableboot_bonf, tableboot_h] = CVfinalbootstrsuptest_shifted_h1rankfix_eta(el, bootMC, z, rvec, w', wi, Kv_sim_h1, CVMv_sim_h1, eta);

QVrejvecs(1,1) = Kv(indmaxKv) > tableboot_sup(1,1);
CVMrejvecs(1,1) = CVMv(indmaxCvM) > tableboot_sup(1,2);

QVrejvecs(1,2) = max(Kv.*w') > tableboot_sup(2,1);
CVMrejvecs(1,2) = max(CVMv.*w') > tableboot_sup(2,2);

QVrejvecs(1,3) = max(Kv.*wi) > tableboot_sup(3,1);
CVMrejvecs(1,3) = max(CVMv.*wi) > tableboot_sup(3,2);

QVrejvecs_bonf(1,1)=Kv(indmaxKv) > tableboot_bonf(1,indmaxKv);
CVMrejvecs_bonf(1,1)=CVMv(indmaxCvM) > tableboot_bonf(2,indmaxCvM);

QVrejvecs_bonf(1,2) = max(Kv.*w') > max(tableboot_bonf(1,:).*w');
CVMrejvecs_bonf(1,2) = max(CVMv.*w') > max(tableboot_bonf(2,:).*w');

QVrejvecs_bonf(1,3) = max(Kv.*wi) > max(tableboot_bonf(1,:).*wi);
CVMrejvecs_bonf(1,3) = max(CVMv.*wi) > max(tableboot_bonf(2,:).*wi);

rej = [QVrejvecs; CVMrejvecs];
rej_bonf = [QVrejvecs_bonf; CVMrejvecs_bonf];
w_row = w';
end

function [result_sup, result_bonf, result_h] = CVfinalbootstrsuptest_shifted_h1rankfix_eta(el, bootMC, pit, rvec, w, wi, Kv_sim_h1, CVMv_sim_h1, eta)
% mirrors galvao_mantoan_mitchell_test_package/CVfinalbootstrsuptest_shifted_h1rankfix.m lines 24-94, eta injected
% eta ((P-el+1+H) x bootMC) replaces randn(n_windows + H, bootMC) at line 44
H=size(pit,2);
KSv = zeros(bootMC,H); CVMv = zeros(bootMC,H);

P = size(pit,1);
size_rvec = length(rvec);
n_windows = P - el + 1;

cs = cell(1,H);
for zvec = 1:H
    emp_cdf = (pit(:,zvec) <= rvec);
    mean_emp_cdf = mean(emp_cdf,1);
    centered = emp_cdf - mean_emp_cdf;
    cs{zvec} = [zeros(1,size_rvec); cumsum(centered,1)];
end

blocksums = cell(1,H);
for zvec = 1:H
    blocksums{zvec} = cs{zvec}(el+1:el+n_windows,:) - cs{zvec}(1:n_windows,:);
end

ETA_shared = (1/sqrt(el)) * eta;

for zvec = 1:H
    ETA_h = ETA_shared(zvec+1 : zvec+n_windows, :);

    K_star_all = (P^(-1/2)) * (ETA_h' * blocksums{zvec});
    KSv(:,zvec) = max(abs(K_star_all),[],2);
    CVMv(:,zvec) = mean(K_star_all.^2,2);
end

N_sim = numel(Kv_sim_h1);
[~, sortIdxK] = sort(KSv(:,1), 'ascend');
ranksK = zeros(bootMC,1); ranksK(sortIdxK) = 1:bootMC;
mapIdxK = max(1, min(N_sim, round(ranksK/bootMC * N_sim)));
KSv(:,1) = Kv_sim_h1(mapIdxK);

[~, sortIdxM] = sort(CVMv(:,1), 'ascend');
ranksM = zeros(bootMC,1); ranksM(sortIdxM) = 1:bootMC;
mapIdxM = max(1, min(N_sim, round(ranksM/bootMC * N_sim)));
CVMv(:,1) = CVMv_sim_h1(mapIdxM);

KSv_sup_max = max(KSv,[],2);
CVMv_sup_max = max(CVMv,[],2);

KSv_sup_w = max(KSv.*w,[],2);
CVMv_sup_w = max(CVMv.*w,[],2);

KSv_sup_wi = max(KSv.*wi,[],2);
CVMv_sup_wi = max(CVMv.*wi,[],2);

KSv = sort(KSv,'ascend');       cvKv_h = KSv(bootMC*0.95,:);
CVMv = sort(CVMv,'ascend');     cvMv_h = CVMv(bootMC*0.95,:);

KSv_sup_max = sort(KSv_sup_max,'ascend');       cvKv_max = KSv_sup_max(bootMC*0.95);
CVMv_sup_max = sort(CVMv_sup_max,'ascend');     cvMv_max = CVMv_sup_max(bootMC*0.95);

KSv_sup_w = sort(KSv_sup_w,'ascend');       cvKv_w = KSv_sup_w(bootMC*0.95);
CVMv_sup_w = sort(CVMv_sup_w,'ascend');     cvMv_w = CVMv_sup_w(bootMC*0.95);

KSv_sup_wi = sort(KSv_sup_wi,'ascend');       cvKv_wi = KSv_sup_wi(bootMC*0.95);
CVMv_sup_wi = sort(CVMv_sup_wi,'ascend');     cvMv_wi = CVMv_sup_wi(bootMC*0.95);

KSv_bonf = sort(KSv,'ascend');       cvKv_bonf = KSv_bonf(round(bootMC*(1-(0.05/H)),0),:);
CVMv_bonf = sort(CVMv,'ascend');     cvMv_bonf = CVMv_bonf(round(bootMC*(1-(0.05/H)),0),:);

result_sup = [cvKv_max cvMv_max; cvKv_w cvMv_w; cvKv_wi cvMv_wi];
result_bonf=[cvKv_bonf;cvMv_bonf];
result_h=[cvKv_h; cvMv_h];
end

%% ===================== Section E: manual snippets =====================
% Paste these into the MTP evaluation scripts and run them once. Datetimes
% are converted to datenum and cell arrays unpacked so scipy.io.loadmat can read them.
%
% --- forecast_evaluation_inflation.m -----------------------------------
% (E1) Insert AFTER line 199  "originDT_all{1} = eval_origins;"
%      (BEFORE line 200 "clearvars modevar meanvar dispvar ..." which deletes the BoE params):
%
%   pe_boe_mode = modevar;  pe_boe_mean = meanvar;  pe_boe_unc = dispvar;   % (h+1) x nQOrig_fc
%   pe_fc_origins = datenum(fc_origins);
%
% (E2) Insert AFTER line 376  "          mu_bv sigma_bv t_idx q_grid;"
%      (end of BVAR block, BEFORE "RUN TESTS FOR EACH MODEL"):
%
%   pe_models = models;  pe_eval_horizons = cfg.eval_horizons;  pe_eval_origins = datenum(eval_origins);
%   pe_actualvar = actualvar;  pe_tau_grid = tau_grid;  pe_q_sharp_lev = q_sharp_lev;
%   pe_zinf_fanchart  = zinf_all{1};  pe_zinf_qr  = zinf_all{2};  pe_zinf_bvar  = zinf_all{3};
%   pe_crps_fanchart  = crps_all{1};  pe_crps_qr  = crps_all{2};  pe_crps_bvar  = crps_all{3};
%   pe_wcrps_fanchart = wcrps_all{1}; pe_wcrps_qr = wcrps_all{2}; pe_wcrps_bvar = wcrps_all{3};   % right tail, v=tau^2
%   pe_qntl_fanchart  = qntl_all{1};  pe_qntl_qr  = qntl_all{2};  pe_qntl_bvar  = qntl_all{3};    % nHor x nOrig x 9
%   save(fullfile('N:\Users\333415\Repos\forecast_evaluation\tests\data\matlab_parity', 'refs_eval_inflation.mat'), ...
%       '-regexp', '^pe_', '-v7');
%
% --- forecast_evaluation_growth.m --------------------------------------
% (E1) Insert AFTER line 261  "originDT_all{1} = eval_origins;"
%      (BEFORE line 263 "clearvars modevar_fc meanvar_fc dispvar_fc ..."):
%
%   pe_boe_mode = modevar_fc;  pe_boe_mean = meanvar_fc;  pe_boe_unc = dispvar_fc;   % 13 x nQOrig_fc
%   pe_fc_origins = datenum(fc_origins);
%
% (E2) Insert AFTER line 382  "          ih j k h_idx t_idx mu_bv sigma_bv act q_grid;"
%      (end of BVAR block, BEFORE "RUN TESTS FOR EACH MODEL"):
%
%   pe_models = models;  pe_eval_horizons = cfg.eval_horizons;  pe_eval_origins = datenum(eval_origins);
%   pe_actualvar = actualvar;  pe_tau_grid = tau_grid;  pe_q_sharp_lev = q_sharp_lev;   % actualvar has Covid NaNs applied
%   pe_zinf_fanchart  = zinf_all{1};  pe_zinf_qr  = zinf_all{2};  pe_zinf_bvar  = zinf_all{3};
%   pe_crps_fanchart  = crps_all{1};  pe_crps_qr  = crps_all{2};  pe_crps_bvar  = crps_all{3};
%   pe_wcrps_fanchart = wcrps_all{1}; pe_wcrps_qr = wcrps_all{2}; pe_wcrps_bvar = wcrps_all{3};   % left tail, v=(1-tau)^2
%   pe_qntl_fanchart  = qntl_all{1};  pe_qntl_qr  = qntl_all{2};  pe_qntl_bvar  = qntl_all{3};
%   save(fullfile('N:\Users\333415\Repos\forecast_evaluation\tests\data\matlab_parity', 'refs_eval_growth.mat'), ...
%       '-regexp', '^pe_', '-v7');
