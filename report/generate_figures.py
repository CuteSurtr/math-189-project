"""Generate all figures for the Math 189 report from data_csv/."""
import os, sys, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy import stats
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from pathlib import Path
from io import StringIO

sns.set_theme(style='whitegrid', palette='muted', font_scale=1.1)
plt.rcParams.update({
    'figure.figsize': (10, 4),
    'figure.dpi': 150,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.15,
})

PROJECT_DIR = Path(__file__).resolve().parent.parent
CSV_DIR = PROJECT_DIR / 'data_csv'
FIG_DIR = Path(__file__).resolve().parent / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)

GAMES = {
    730:    ('Counter-Strike 2',       'popular'),
    570:    ('Dota 2',                 'popular'),
    578080: ('PUBG: Battlegrounds',    'popular'),
    271590: ('Grand Theft Auto V',     'popular'),
    230410: ('Warframe',               'popular'),
    252490: ('Rust',                   'decline'),
    346110: ('ARK: Survival Evolved',  'decline'),
    304390: ('For Honor',              'decline'),
    381210: ('Dead by Daylight',       'decline'),
    359550: ('Rainbow Six Siege',      'decline'),
    275850: ("No Man's Sky",           'volatile'),
    1085660:('Destiny 2',              'volatile'),
    105600: ('Terraria',               'volatile'),
    582010: ('Monster Hunter: World',  'volatile'),
    218620: ('PAYDAY 2',               'volatile'),
}

CSV_NAME_MAP = {
    'Counter-Strike (CSGO)': 'Counter-Strike 2',
    'ARK Survival Evolved':  'ARK: Survival Evolved',
    'Monster Hunter World':  'Monster Hunter: World',
    'PUBG Battlegrounds':    'PUBG: Battlegrounds',
}
NAME_TO_INFO = {name: (app_id, stratum) for app_id, (name, stratum) in GAMES.items()}

PANEL_END_DATE = '2024-09-30'

# ── Load panel ────────────────────────────────────────────────────────────────
records = []
for pf in sorted(CSV_DIR.glob('*_players.csv')):
    csv_name = pf.stem.replace('_players', '')
    game_name = CSV_NAME_MAP.get(csv_name, csv_name)
    if game_name not in NAME_TO_INFO:
        continue
    app_id, stratum = NAME_TO_INFO[game_name]
    sf = CSV_DIR / f'{csv_name}_sentiment.csv'
    if not sf.exists():
        continue
    pdf = pd.read_csv(pf, parse_dates=['week'])
    sdf = pd.read_csv(sf, parse_dates=['week'])
    m = pdf.merge(sdf, on='week', how='inner')
    m = m[(m['players'] >= 100) & (m['week'] <= PANEL_END_DATE)]
    if m.empty:
        continue
    m['app_id'] = app_id
    m['game'] = game_name
    m['stratum'] = stratum
    woy = m['week'].dt.isocalendar().week.astype(int)
    m['season_sale'] = (((woy >= 25) & (woy <= 27)) | ((woy >= 51) | (woy <= 1))).astype(int)
    records.append(m)

panel = pd.concat(records, ignore_index=True)
panel = panel[['app_id','game','stratum','week','log_players','players',
               'neg_sentiment','pos_sentiment','season_sale']]

print(f"Panel: {panel.shape}, Games: {panel['game'].nunique()}")

# ── Fig 1: Player count trends by stratum ─────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=False)
for ax, stratum in zip(axes, ['popular', 'decline', 'volatile']):
    sub = panel[panel['stratum'] == stratum]
    for g in sub['game'].unique():
        gd = sub[sub['game'] == g].sort_values('week')
        ax.plot(gd['week'], gd['log_players'], label=g, linewidth=0.9)
    ax.set_title(stratum.title())
    ax.set_xlabel('Date')
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.tick_params(axis='x', rotation=45)
    if stratum == 'popular':
        ax.set_ylabel('log(Players)')
    ax.legend(fontsize=6, loc='lower left')
fig.suptitle('Weekly Log Player Counts by Stratum', fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(FIG_DIR / 'fig1_player_trends.pdf')
plt.close()
print("Fig 1 done")

# ── Fig 2: Sentiment distributions ────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, stratum in zip(axes, ['popular', 'decline', 'volatile']):
    sub = panel[panel['stratum'] == stratum]
    ax.hist(sub['neg_sentiment'], bins=40, alpha=0.7, color='salmon', edgecolor='white')
    ax.set_title(f'{stratum.title()} Stratum')
    ax.set_xlabel('Negative Sentiment')
    ax.set_ylabel('Count')
fig.suptitle('Distribution of Weekly Negative Sentiment by Stratum', fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(FIG_DIR / 'fig2_sentiment_dist.pdf')
plt.close()
print("Fig 2 done")

# ── Fig 3: Dual-axis time series for one game ────────────────────────────────
game_ex = 'Dota 2'
gd = panel[panel['game'] == game_ex].sort_values('week')
fig, ax1 = plt.subplots(figsize=(10, 4))
ax1.plot(gd['week'], gd['log_players'], color='steelblue', linewidth=1)
ax1.set_ylabel('log(Players)', color='steelblue')
ax1.tick_params(axis='y', labelcolor='steelblue')
ax2 = ax1.twinx()
ax2.plot(gd['week'], gd['neg_sentiment'], color='salmon', linewidth=0.8, alpha=0.8)
ax2.set_ylabel('Neg. Sentiment', color='salmon')
ax2.tick_params(axis='y', labelcolor='salmon')
ax1.set_title(f'{game_ex}: Weekly Player Counts and Negative Sentiment')
ax1.xaxis.set_major_locator(mdates.YearLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
fig.tight_layout()
fig.savefig(FIG_DIR / 'fig3_timeseries_dota2.pdf')
plt.close()
print("Fig 3 done")

# ── Fig 4: Cross-correlation ─────────────────────────────────────────────────
max_lag = 12
ccf_results = []
for g in panel['game'].unique():
    gd = panel[panel['game'] == g].sort_values('week')
    x = gd['neg_sentiment'].values
    y = gd['log_players'].values
    x = (x - x.mean()) / (x.std() + 1e-12)
    y = (y - y.mean()) / (y.std() + 1e-12)
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            cc = np.corrcoef(x[:len(x)-lag], y[lag:])[0, 1] if lag < len(x) else 0
        else:
            cc = np.corrcoef(x[-lag:], y[:len(y)+lag])[0, 1] if -lag < len(y) else 0
        ccf_results.append({'game': g, 'lag': lag, 'ccf': cc})

ccf_df = pd.DataFrame(ccf_results)
mean_ccf = ccf_df.groupby('lag')['ccf'].mean()

fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(mean_ccf.index, mean_ccf.values, color='steelblue', alpha=0.7)
ax.axhline(0, color='black', linewidth=0.5)
n_avg = panel.groupby('game').size().mean()
ci = 1.96 / np.sqrt(n_avg)
ax.axhline(ci, color='red', linestyle='--', linewidth=0.8, label=f'95% CI (±{ci:.3f})')
ax.axhline(-ci, color='red', linestyle='--', linewidth=0.8)
ax.set_xlabel('Lag (weeks)')
ax.set_ylabel('Mean Cross-Correlation')
ax.set_title('Cross-Correlation: Negative Sentiment → log(Players)')
ax.legend()
fig.tight_layout()
fig.savefig(FIG_DIR / 'fig4_crosscorrelation.pdf')
plt.close()
print("Fig 4 done")

# ── Fig 5: Correlation heatmap ────────────────────────────────────────────────
num_cols = ['log_players', 'neg_sentiment', 'pos_sentiment', 'season_sale']
corr_pearson = panel[num_cols].corr(method='pearson')

fig, ax = plt.subplots(figsize=(6, 5))
mask = np.triu(np.ones_like(corr_pearson, dtype=bool), k=1)
sns.heatmap(corr_pearson, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
            mask=mask, square=True, ax=ax, vmin=-1, vmax=1)
ax.set_title('Pearson Correlation Matrix')
fig.tight_layout()
fig.savefig(FIG_DIR / 'fig5_correlation_matrix.pdf')
plt.close()
print("Fig 5 done")

# ── Fig 6: Granger causality heatmap ─────────────────────────────────────────
max_gc_lag = 4
gc_pvals = {}
for g in panel['game'].unique():
    gd = panel[panel['game'] == g].sort_values('week')
    y = gd['log_players'].diff().dropna()
    x = gd['neg_sentiment'].iloc[1:]
    data = pd.DataFrame({'y': y.values, 'x': x.values}).dropna()
    if len(data) < 30:
        continue
    pvals_by_lag = {}
    for lag in range(1, max_gc_lag + 1):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                res = grangercausalitytests(data[['y', 'x']], maxlag=lag, verbose=False)
            pvals_by_lag[lag] = res[lag][0]['ssr_ftest'][1]
        except:
            pvals_by_lag[lag] = 1.0
    gc_pvals[g] = pvals_by_lag

gc_df = pd.DataFrame(gc_pvals).T
gc_df.columns = [f'Lag {i}' for i in range(1, max_gc_lag + 1)]
gc_df = gc_df.sort_index()

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(gc_df, annot=True, fmt='.3f', cmap='RdYlGn_r', vmin=0, vmax=0.15,
            ax=ax, linewidths=0.5)
ax.set_title('Granger Causality p-values: Sentiment → Player Changes')
ax.set_ylabel('Game')
fig.tight_layout()
fig.savefig(FIG_DIR / 'fig6_granger_heatmap.pdf')
plt.close()
print("Fig 6 done")

# ── Prepare regression data ───────────────────────────────────────────────────
reg = panel.copy()
reg = reg.sort_values(['game', 'week'])
reg['d_log_players'] = reg.groupby('game')['log_players'].diff()
reg['neg_sent_lag1'] = reg.groupby('game')['neg_sentiment'].shift(1)
reg['neg_sent_lag2'] = reg.groupby('game')['neg_sentiment'].shift(2)
reg['neg_sent_lag3'] = reg.groupby('game')['neg_sentiment'].shift(3)
reg['update_flag'] = 0
reg = reg.dropna(subset=['d_log_players', 'neg_sent_lag1'])

# ── Fig 7: Bootstrap CI histogram ────────────────────────────────────────────
import statsmodels.api as sm

games_list = reg['game'].unique()
n_boot = 2000
np.random.seed(189)
boot_betas = []
for b in range(n_boot):
    sampled_games = np.random.choice(games_list, size=len(games_list), replace=True)
    frames = []
    for g in sampled_games:
        frames.append(reg[reg['game'] == g])
    bdf = pd.concat(frames, ignore_index=True)
    y = bdf['d_log_players']
    X = bdf[['neg_sent_lag1', 'update_flag', 'season_sale']]
    X = sm.add_constant(X)
    try:
        res = sm.OLS(y, X).fit()
        boot_betas.append(res.params['neg_sent_lag1'])
    except:
        pass

boot_betas = np.array(boot_betas)
ci_low, ci_high = np.percentile(boot_betas, [2.5, 97.5])

fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(boot_betas, bins=50, color='steelblue', alpha=0.7, edgecolor='white', density=True)
ax.axvline(np.mean(boot_betas), color='red', linestyle='-', linewidth=2, label=f'Mean = {np.mean(boot_betas):.4f}')
ax.axvline(ci_low, color='orange', linestyle='--', linewidth=1.5, label=f'2.5% = {ci_low:.4f}')
ax.axvline(ci_high, color='orange', linestyle='--', linewidth=1.5, label=f'97.5% = {ci_high:.4f}')
ax.axvline(0, color='black', linestyle=':', linewidth=1)
ax.set_xlabel(r'$\hat{\beta}$ (neg_sent_lag1)')
ax.set_ylabel('Density')
ax.set_title(r'Bootstrap Distribution of $\hat{\beta}$ (2,000 replications)')
ax.legend()
fig.tight_layout()
fig.savefig(FIG_DIR / 'fig7_bootstrap_ci.pdf')
plt.close()
print("Fig 7 done")

# ── Fig 8: LOESS comparison ──────────────────────────────────────────────────
from statsmodels.nonparametric.smoothers_lowess import lowess

x_sent = reg['neg_sent_lag1'].values
y_dlp = reg['d_log_players'].values
loess_result = lowess(y_dlp, x_sent, frac=0.3, return_sorted=True)

ols_res = sm.OLS(y_dlp, sm.add_constant(x_sent)).fit()

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(x_sent, y_dlp, alpha=0.05, s=5, color='grey')
ax.plot(loess_result[:, 0], loess_result[:, 1], color='red', linewidth=2, label='LOESS')
x_range = np.linspace(x_sent.min(), x_sent.max(), 100)
ax.plot(x_range, ols_res.params[0] + ols_res.params[1] * x_range,
        color='steelblue', linewidth=2, linestyle='--', label='OLS')
ax.set_xlabel('Lagged Negative Sentiment')
ax.set_ylabel('Δ log(Players)')
ax.set_title('LOESS vs. Linear OLS: Sentiment → Player Change')
ax.legend()
fig.tight_layout()
fig.savefig(FIG_DIR / 'fig8_loess.pdf')
plt.close()
print("Fig 8 done")

# ── Fig 9: Out-of-sample prediction ──────────────────────────────────────────
split_date = pd.Timestamp('2023-06-01')
train = reg[reg['week'] < split_date]
test = reg[reg['week'] >= split_date]

y_train = train['d_log_players']
X_train = sm.add_constant(train[['neg_sent_lag1', 'update_flag', 'season_sale']])
y_test = test['d_log_players']
X_test = sm.add_constant(test[['neg_sent_lag1', 'update_flag', 'season_sale']])

ols_train = sm.OLS(y_train, X_train).fit()
y_pred = ols_train.predict(X_test)

rmse_model = np.sqrt(np.mean((y_test - y_pred)**2))
rmse_base = np.sqrt(np.mean(y_test**2))

fig, ax = plt.subplots(figsize=(8, 4))
ax.scatter(y_test, y_pred, alpha=0.1, s=8, color='steelblue')
lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
ax.plot(lims, lims, 'r--', linewidth=1)
ax.set_xlabel('Actual Δ log(Players)')
ax.set_ylabel('Predicted Δ log(Players)')
ax.set_title(f'Out-of-Sample: RMSE = {rmse_model:.4f} (baseline {rmse_base:.4f})')
fig.tight_layout()
fig.savefig(FIG_DIR / 'fig9_oos_prediction.pdf')
plt.close()
print("Fig 9 done")

# ── Fig 10: Diagnostics – residual ACF ────────────────────────────────────────
full_y = reg['d_log_players']
full_X = sm.add_constant(reg[['neg_sent_lag1', 'update_flag', 'season_sale']])
full_res = sm.OLS(full_y, full_X).fit()
resid = full_res.resid

from statsmodels.graphics.tsaplots import plot_acf
fig, ax = plt.subplots(figsize=(8, 3.5))
plot_acf(resid, lags=30, ax=ax, alpha=0.05)
ax.set_title('Residual Autocorrelation (ACF)')
fig.tight_layout()
fig.savefig(FIG_DIR / 'fig10_residual_acf.pdf')
plt.close()
print("Fig 10 done")

print("\nAll figures generated in:", FIG_DIR)
