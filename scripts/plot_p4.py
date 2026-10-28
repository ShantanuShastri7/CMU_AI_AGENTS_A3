import json
import matplotlib.pyplot as plt
import numpy as np

def load_run(path):
    train_records = []
    eval_records = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            if 'train/success_rate' in d:
                train_records.append(d)
            if 'eval/success_rate' in d:
                eval_records.append(d)
    return train_records, eval_records

def extract_metric(records, key):
    steps = []
    vals = []
    for r in records:
        if key in r and r[key] is not None:
            steps.append(r['step'])
            vals.append(r[key])
    return np.array(steps), np.array(vals)

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

# =====================================================================
# Study 4.1: Group Size (p2a G=1 vs p41 G=2 vs p2b G=8)
# =====================================================================
train_p2a, eval_p2a = load_run('runs/p2a_number_search_reinforce.jsonl')
train_p41, eval_p41 = load_run('runs/p41_number_search_grpo_g2.jsonl')
train_p2b, eval_p2b = load_run('runs/p2b_number_search_grpo.jsonl')

fig, axes = plt.subplots(2, 3, figsize=(18, 10), dpi=300)
fig.suptitle('Part 4.1: Group Size Ablation — REINFORCE (G=1) vs. GRPO (G=2) vs. GRPO (G=8)',
             fontsize=15, fontweight='bold', y=0.98)

c_p2a = '#E65100'  # Deep Orange (G=1)
c_p41 = '#8E24AA'  # Purple (G=2)
c_p2b = '#1565C0'  # Blue (G=8)
c_p2a_alt = '#FFA726'
c_p41_alt = '#BA68C8'
c_p2b_alt = '#64B5F6'

# 1. Reward Quality (Eval)
ax1 = axes[0, 0]
s, sr_p2a = extract_metric(eval_p2a, 'eval/success_rate')
_, p4_p2a = extract_metric(eval_p2a, 'eval/pass@4')
ax1.plot(s, sr_p2a, marker='o', markersize=5, color=c_p2a, linewidth=1.8, label='p2a (G=1): eval/success_rate')
ax1.plot(s, p4_p2a, marker='o', markersize=4, linestyle='--', color=c_p2a_alt, linewidth=1.4, label='p2a (G=1): eval/pass@4')

s, sr_p41 = extract_metric(eval_p41, 'eval/success_rate')
_, p4_p41 = extract_metric(eval_p41, 'eval/pass@4')
ax1.plot(s, sr_p41, marker='s', markersize=5, color=c_p41, linewidth=1.8, label='p41 (G=2): eval/success_rate')
ax1.plot(s, p4_p41, marker='s', markersize=4, linestyle='--', color=c_p41_alt, linewidth=1.4, label='p41 (G=2): eval/pass@4')

s, sr_p2b = extract_metric(eval_p2b, 'eval/success_rate')
_, p4_p2b = extract_metric(eval_p2b, 'eval/pass@4')
ax1.plot(s, sr_p2b, marker='^', markersize=5, color=c_p2b, linewidth=1.8, label='p2b (G=8): eval/success_rate')
ax1.plot(s, p4_p2b, marker='^', markersize=4, linestyle='--', color=c_p2b_alt, linewidth=1.4, label='p2b (G=8): eval/pass@4')

ax1.set_title('Reward Quality: Evaluation Metrics', fontweight='bold', fontsize=11)
ax1.set_xlabel('Training Step', fontsize=10)
ax1.set_ylabel('Rate', fontsize=10)
ax1.set_ylim(-0.05, 1.05)
ax1.legend(loc='lower right', fontsize=8, framealpha=0.9)
ax1.grid(True, linestyle=':', alpha=0.6)

# 2. Reward Quality (Train)
ax2 = axes[0, 1]
s, sr_p2a = extract_metric(train_p2a, 'train/success_rate')
_, pass1_p2a = extract_metric(train_p2a, 'train/pass@1')
ax2.plot(s, sr_p2a, color=c_p2a, linewidth=1.4, alpha=0.85, label='p2a (G=1): train/success_rate')
ax2.plot(s, pass1_p2a, linestyle=':', color=c_p2a_alt, linewidth=1.3, label='p2a (G=1): train/pass@1')

s, sr_p41 = extract_metric(train_p41, 'train/success_rate')
_, pass2_p41 = extract_metric(train_p41, 'train/pass@2')
ax2.plot(s, sr_p41, color=c_p41, linewidth=1.4, alpha=0.85, label='p41 (G=2): train/success_rate')
ax2.plot(s, pass2_p41, linestyle='--', color=c_p41_alt, linewidth=1.3, label='p41 (G=2): train/pass@2')

s, sr_p2b = extract_metric(train_p2b, 'train/success_rate')
_, pass8_p2b = extract_metric(train_p2b, 'train/pass@8')
ax2.plot(s, sr_p2b, color=c_p2b, linewidth=1.4, alpha=0.85, label='p2b (G=8): train/success_rate')
ax2.plot(s, pass8_p2b, linestyle='--', color=c_p2b_alt, linewidth=1.3, label='p2b (G=8): train/pass@8')

ax2.set_title('Reward Quality: Training Metrics', fontweight='bold', fontsize=11)
ax2.set_xlabel('Training Step', fontsize=10)
ax2.set_ylabel('Rate', fontsize=10)
ax2.set_ylim(-0.05, 1.05)
ax2.legend(loc='lower right', fontsize=8, framealpha=0.9)
ax2.grid(True, linestyle=':', alpha=0.6)

# 3. Optimization: Grad Norm
ax3 = axes[0, 2]
s, gn_p2a = extract_metric(train_p2a, 'train/grad_norm')
s41, gn_p41 = extract_metric(train_p41, 'train/grad_norm')
sb, gn_p2b = extract_metric(train_p2b, 'train/grad_norm')
ax3.plot(s, gn_p2a, marker='o', markersize=2.5, color=c_p2a, linewidth=1.2, label='p2a (G=1): train/grad_norm')
ax3.plot(s41, gn_p41, marker='s', markersize=2.5, color=c_p41, linewidth=1.2, label='p41 (G=2): train/grad_norm')
ax3.plot(sb, gn_p2b, marker='^', markersize=2.5, color=c_p2b, linewidth=1.2, label='p2b (G=8): train/grad_norm')
ax3.set_title('Optimization: Gradient Norm', fontweight='bold', fontsize=11)
ax3.set_xlabel('Training Step', fontsize=10)
ax3.set_ylabel('train/grad_norm', fontsize=10)
ax3.legend(loc='upper right', fontsize=8.5, framealpha=0.9)
ax3.grid(True, linestyle=':', alpha=0.6)

# 4. Policy Drift: IW Mean & Max
ax4 = axes[1, 0]
s, iw_m_p2a = extract_metric(train_p2a, 'train/importance_weight_mean')
_, iw_max_p2a = extract_metric(train_p2a, 'train/importance_weight_max')
s41, iw_m_p41 = extract_metric(train_p41, 'train/importance_weight_mean')
_, iw_max_p41 = extract_metric(train_p41, 'train/importance_weight_max')
sb, iw_m_p2b = extract_metric(train_p2b, 'train/importance_weight_mean')
_, iw_max_p2b = extract_metric(train_p2b, 'train/importance_weight_max')

ax4.plot(s, iw_m_p2a, color=c_p2a, linewidth=1.4, label='p2a (G=1): iw_mean')
ax4.plot(s41, iw_m_p41, color=c_p41, linewidth=1.4, label='p41 (G=2): iw_mean')
ax4.plot(sb, iw_m_p2b, color=c_p2b, linewidth=1.4, label='p2b (G=8): iw_mean')
ax4.plot(s, iw_max_p2a, linestyle='--', color=c_p2a_alt, linewidth=1.1, label='p2a: iw_max')
ax4.plot(s41, iw_max_p41, linestyle='--', color=c_p41_alt, linewidth=1.1, label='p41: iw_max')
ax4.plot(sb, iw_max_p2b, linestyle='--', color=c_p2b_alt, linewidth=1.1, label='p2b: iw_max')
ax4.axhline(1.0, color='gray', linestyle=':', alpha=0.8, label='On-policy Reference (1.0)')
ax4.set_title('Policy Drift: Importance Weights (Mean & Max)', fontweight='bold', fontsize=11)
ax4.set_xlabel('Training Step', fontsize=10)
ax4.set_ylabel('Importance Weight', fontsize=10)
ax4.legend(loc='upper right', fontsize=7.5, framealpha=0.9)
ax4.grid(True, linestyle=':', alpha=0.6)

# 5. Policy Drift: IW Std
ax5 = axes[1, 1]
s, iw_std_p2a = extract_metric(train_p2a, 'train/importance_weight_std')
s41, iw_std_p41 = extract_metric(train_p41, 'train/importance_weight_std')
sb, iw_std_p2b = extract_metric(train_p2b, 'train/importance_weight_std')
ax5.plot(s, iw_std_p2a, marker='o', markersize=2.5, color=c_p2a, linewidth=1.2, label='p2a (G=1): train/importance_weight_std')
ax5.plot(s41, iw_std_p41, marker='s', markersize=2.5, color=c_p41, linewidth=1.2, label='p41 (G=2): train/importance_weight_std')
ax5.plot(sb, iw_std_p2b, marker='^', markersize=2.5, color=c_p2b, linewidth=1.2, label='p2b (G=8): train/importance_weight_std')
ax5.set_title('Policy Drift: Importance Weight Std', fontweight='bold', fontsize=11)
ax5.set_xlabel('Training Step', fontsize=10)
ax5.set_ylabel('train/importance_weight_std', fontsize=10)
ax5.legend(loc='upper right', fontsize=8.5, framealpha=0.9)
ax5.grid(True, linestyle=':', alpha=0.6)

# 6. Data Health: Degenerate Groups & Entropy Proxy
ax6 = axes[1, 2]
s41_dg, degen_p41 = extract_metric(train_p41, 'train/frac_degenerate_groups')
sb_dg, degen_p2b = extract_metric(train_p2b, 'train/frac_degenerate_groups')
s41_ent, ent_p41 = extract_metric(train_p41, 'train/entropy_proxy')
sb_ent, ent_p2b = extract_metric(train_p2b, 'train/entropy_proxy')
sa_ent, ent_p2a = extract_metric(train_p2a, 'train/entropy_proxy')

ax6.plot(s41_dg, degen_p41, marker='s', markersize=2.5, color=c_p41, linewidth=1.4, label='p41 (G=2): frac_degenerate_groups')
ax6.plot(sb_dg, degen_p2b, marker='^', markersize=2.5, color=c_p2b, linewidth=1.4, label='p2b (G=8): frac_degenerate_groups')
ax6.plot(sa_ent, ent_p2a, linestyle=':', color=c_p2a, linewidth=1.2, label='p2a (G=1): entropy_proxy')
ax6.plot(s41_ent, ent_p41, linestyle=':', color=c_p41, linewidth=1.2, label='p41 (G=2): entropy_proxy')
ax6.plot(sb_ent, ent_p2b, linestyle=':', color=c_p2b, linewidth=1.2, label='p2b (G=8): entropy_proxy')

ax6.set_title('Data Health: Degenerate Groups & Entropy Proxy', fontweight='bold', fontsize=11)
ax6.set_xlabel('Training Step', fontsize=10)
ax6.set_ylabel('Metric Value', fontsize=10)
ax6.legend(loc='center right', fontsize=7.5, framealpha=0.9)
ax6.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('assets/p41_diagnostic_plots.png', bbox_inches='tight')
print('Study 4.1 diagnostic plots saved to assets/p41_diagnostic_plots.png')
plt.close()

# =====================================================================
# Study 4.3: Importance Weighting Under Higher Staleness
# (p43_iw PPO vs p43_noiw Cross-Entropy)
# =====================================================================
train_p43_iw, eval_p43_iw = load_run('runs/p43_number_search_async_stale4_iw.jsonl')
train_p43_noiw, eval_p43_noiw = load_run('runs/p43_number_search_async_stale4_noiw.jsonl')

fig, axes = plt.subplots(2, 3, figsize=(18, 10), dpi=300)
fig.suptitle('Part 4.3: Importance Weighting Ablation (Staleness 4) — PPO (with IW) vs. Cross-Entropy (No IW)',
             fontsize=15, fontweight='bold', y=0.98)

c_iw = '#00796B'      # Teal (PPO with IW)
c_iw_alt = '#4DB6AC'
c_noiw = '#D32F2F'    # Crimson / Red (No IW Cross-Entropy)
c_noiw_alt = '#E57373'

# 1. Reward Quality (Eval)
ax1 = axes[0, 0]
s, sr_iw = extract_metric(eval_p43_iw, 'eval/success_rate')
_, p4_iw = extract_metric(eval_p43_iw, 'eval/pass@4')
ax1.plot(s, sr_iw, marker='o', markersize=5, color=c_iw, linewidth=2, label='p43_iw (PPO): eval/success_rate')
ax1.plot(s, p4_iw, marker='o', markersize=4, linestyle='--', color=c_iw_alt, linewidth=1.5, label='p43_iw (PPO): eval/pass@4')

s, sr_noiw = extract_metric(eval_p43_noiw, 'eval/success_rate')
_, p4_noiw = extract_metric(eval_p43_noiw, 'eval/pass@4')
ax1.plot(s, sr_noiw, marker='s', markersize=5, color=c_noiw, linewidth=2, label='p43_noiw (CE): eval/success_rate')
ax1.plot(s, p4_noiw, marker='s', markersize=4, linestyle='--', color=c_noiw_alt, linewidth=1.5, label='p43_noiw (CE): eval/pass@4')

ax1.set_title('Reward Quality: Evaluation Metrics', fontweight='bold', fontsize=11)
ax1.set_xlabel('Training Step', fontsize=10)
ax1.set_ylabel('Rate', fontsize=10)
ax1.set_ylim(-0.05, 1.05)
ax1.legend(loc='lower right', fontsize=8.5, framealpha=0.9)
ax1.grid(True, linestyle=':', alpha=0.6)

# 2. Reward Quality (Train)
ax2 = axes[0, 1]
s, sr_iw = extract_metric(train_p43_iw, 'train/success_rate')
_, p8_iw = extract_metric(train_p43_iw, 'train/pass@8')
ax2.plot(s, sr_iw, color=c_iw, linewidth=1.5, label='p43_iw (PPO): train/success_rate')
ax2.plot(s, p8_iw, linestyle='--', color=c_iw_alt, linewidth=1.5, label='p43_iw (PPO): train/pass@8')

s, sr_noiw = extract_metric(train_p43_noiw, 'train/success_rate')
_, p8_noiw = extract_metric(train_p43_noiw, 'train/pass@8')
ax2.plot(s, sr_noiw, color=c_noiw, linewidth=1.5, label='p43_noiw (CE): train/success_rate')
ax2.plot(s, p8_noiw, linestyle='--', color=c_noiw_alt, linewidth=1.5, label='p43_noiw (CE): train/pass@8')

ax2.set_title('Reward Quality: Training Metrics', fontweight='bold', fontsize=11)
ax2.set_xlabel('Training Step', fontsize=10)
ax2.set_ylabel('Rate', fontsize=10)
ax2.set_ylim(-0.05, 1.05)
ax2.legend(loc='lower right', fontsize=8.5, framealpha=0.9)
ax2.grid(True, linestyle=':', alpha=0.6)

# 3. Optimization: Grad Norm
ax3 = axes[0, 2]
s_iw, gn_iw = extract_metric(train_p43_iw, 'train/grad_norm')
s_noiw, gn_noiw = extract_metric(train_p43_noiw, 'train/grad_norm')
ax3.plot(s_iw, gn_iw, marker='o', markersize=3, color=c_iw, linewidth=1.4, label='p43_iw (PPO): train/grad_norm')
ax3.plot(s_noiw, gn_noiw, marker='s', markersize=3, color=c_noiw, linewidth=1.4, label='p43_noiw (CE): train/grad_norm')
ax3.set_title('Optimization: Gradient Norm', fontweight='bold', fontsize=11)
ax3.set_xlabel('Training Step', fontsize=10)
ax3.set_ylabel('train/grad_norm', fontsize=10)
ax3.legend(loc='upper right', fontsize=8.5, framealpha=0.9)
ax3.grid(True, linestyle=':', alpha=0.6)

# 4. Policy Drift: Importance Weights (Mean, Max & Min)
ax4 = axes[1, 0]
s_iw, iw_m_iw = extract_metric(train_p43_iw, 'train/importance_weight_mean')
_, iw_max_iw = extract_metric(train_p43_iw, 'train/importance_weight_max')
_, iw_min_iw = extract_metric(train_p43_iw, 'train/importance_weight_min')

s_noiw, iw_m_noiw = extract_metric(train_p43_noiw, 'train/importance_weight_mean')
_, iw_max_noiw = extract_metric(train_p43_noiw, 'train/importance_weight_max')
_, iw_min_noiw = extract_metric(train_p43_noiw, 'train/importance_weight_min')

ax4.plot(s_iw, iw_m_iw, color=c_iw, linewidth=1.6, label='p43_iw: iw_mean (~1.0)')
ax4.plot(s_iw, iw_max_iw, linestyle='--', color=c_iw_alt, linewidth=1.3, label='p43_iw: iw_max')
ax4.plot(s_noiw, iw_m_noiw, color=c_noiw, linewidth=1.6, label='p43_noiw: iw_mean (~1.0)')
ax4.plot(s_noiw, iw_max_noiw, linestyle='--', color=c_noiw_alt, linewidth=1.5, label='p43_noiw: iw_max (surges to 7.28!)')
ax4.axhline(1.0, color='gray', linestyle=':', alpha=0.8, label='On-policy Reference (1.0)')

ax4.set_title('Policy Drift: Importance Weights (Mean & Max)', fontweight='bold', fontsize=11)
ax4.set_xlabel('Training Step', fontsize=10)
ax4.set_ylabel('Importance Weight', fontsize=10)
ax4.legend(loc='upper left', fontsize=8, framealpha=0.9)
ax4.grid(True, linestyle=':', alpha=0.6)

# 5. Policy Drift: IW Standard Deviation
ax5 = axes[1, 1]
s_iw, iw_std_iw = extract_metric(train_p43_iw, 'train/importance_weight_std')
s_noiw, iw_std_noiw = extract_metric(train_p43_noiw, 'train/importance_weight_std')
ax5.plot(s_iw, iw_std_iw, marker='o', markersize=3, color=c_iw, linewidth=1.5, 
         label=f'p43_iw (PPO): mean={np.mean(iw_std_iw):.4f}')
ax5.plot(s_noiw, iw_std_noiw, marker='s', markersize=3, color=c_noiw, linewidth=1.5, 
         label=f'p43_noiw (CE): mean={np.mean(iw_std_noiw):.4f} (max={np.max(iw_std_noiw):.4f})')

ax5.set_title('Policy Drift: Importance Weight Std', fontweight='bold', fontsize=11)
ax5.set_xlabel('Training Step', fontsize=10)
ax5.set_ylabel('train/importance_weight_std', fontsize=10)
ax5.legend(loc='upper left', fontsize=8.5, framealpha=0.9)
ax5.grid(True, linestyle=':', alpha=0.6)

# 6. Data Health: Degenerate Groups & Entropy Proxy
ax6 = axes[1, 2]
s_iw_dg, degen_iw = extract_metric(train_p43_iw, 'train/frac_degenerate_groups')
s_noiw_dg, degen_noiw = extract_metric(train_p43_noiw, 'train/frac_degenerate_groups')
s_iw_ent, ent_iw = extract_metric(train_p43_iw, 'train/entropy_proxy')
s_noiw_ent, ent_noiw = extract_metric(train_p43_noiw, 'train/entropy_proxy')

ax6.plot(s_iw_dg, degen_iw, color=c_iw, linewidth=1.5, label='p43_iw: frac_degenerate_groups')
ax6.plot(s_noiw_dg, degen_noiw, color=c_noiw, linewidth=1.5, label='p43_noiw: frac_degenerate_groups')
ax6.plot(s_iw_ent, ent_iw, linestyle=':', color=c_iw, linewidth=1.4, label='p43_iw: entropy_proxy')
ax6.plot(s_noiw_ent, ent_noiw, linestyle=':', color=c_noiw, linewidth=1.4, label='p43_noiw: entropy_proxy')

ax6.set_title('Data Health: Degenerate Groups & Entropy Proxy', fontweight='bold', fontsize=11)
ax6.set_xlabel('Training Step', fontsize=10)
ax6.set_ylabel('Metric Value', fontsize=10)
ax6.legend(loc='center right', fontsize=8, framealpha=0.9)
ax6.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('assets/p43_diagnostic_plots.png', bbox_inches='tight')
print('Study 4.3 diagnostic plots saved to assets/p43_diagnostic_plots.png')
