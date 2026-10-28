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

train_p2b, eval_p2b = load_run('runs/p2b_number_search_grpo.jsonl')
train_p3, eval_p3 = load_run('runs/p3_number_search_async.jsonl')

def extract_metric(records, key):
    steps = []
    vals = []
    for r in records:
        if key in r and r[key] is not None:
            steps.append(r['step'])
            vals.append(r[key])
    return np.array(steps), np.array(vals)

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, axes = plt.subplots(2, 4, figsize=(22, 10), dpi=300)
fig.suptitle('Part 3: Standard Diagnostic Plots — Asynchronous GRPO (p3) vs. Synchronous GRPO (p2b)', 
             fontsize=16, fontweight='bold', y=0.98)

# Colors
c_p3 = '#2E7D32'       # Forest Green
c_p3_light = '#81C784' # Light Green
c_p3_dark = '#1B5E20'  # Deep Green
c_p2b = '#1565C0'      # Blue
c_p2b_light = '#64B5F6'# Light Blue
c_warn = '#E65100'     # Amber / Orange

# -------------------------------------------------------------
# 1. Reward Quality: Evaluation Metrics
# -------------------------------------------------------------
ax1 = axes[0, 0]
s_p3_e, sr_p3_e = extract_metric(eval_p3, 'eval/success_rate')
_, p4_p3_e = extract_metric(eval_p3, 'eval/pass@4')
s_p2b_e, sr_p2b_e = extract_metric(eval_p2b, 'eval/success_rate')
_, p4_p2b_e = extract_metric(eval_p2b, 'eval/pass@4')

ax1.plot(s_p3_e, sr_p3_e, marker='o', markersize=6, color=c_p3, linewidth=2, label='p3 (Async): eval/success_rate')
ax1.plot(s_p3_e, p4_p3_e, marker='s', markersize=5, linestyle='--', color=c_p3_light, linewidth=1.8, label='p3 (Async): eval/pass@4')
ax1.plot(s_p2b_e, sr_p2b_e, marker='^', markersize=5, linestyle=':', color=c_p2b, linewidth=1.5, alpha=0.75, label='p2b (Sync): eval/success_rate')
ax1.plot(s_p2b_e, p4_p2b_e, marker='D', markersize=4, linestyle=':', color=c_p2b_light, linewidth=1.5, alpha=0.75, label='p2b (Sync): eval/pass@4')

ax1.set_title('Reward Quality: Evaluation Metrics', fontweight='bold', fontsize=11)
ax1.set_xlabel('Training Step', fontsize=10)
ax1.set_ylabel('Rate', fontsize=10)
ax1.set_ylim(-0.05, 1.05)
ax1.legend(loc='lower right', fontsize=8, framealpha=0.9)
ax1.grid(True, linestyle=':', alpha=0.6)

# -------------------------------------------------------------
# 2. Reward Quality: Training Metrics
# -------------------------------------------------------------
ax2 = axes[0, 1]
s_p3_t, sr_p3_t = extract_metric(train_p3, 'train/success_rate')
_, p8_p3_t = extract_metric(train_p3, 'train/pass@8')
s_p2b_t, sr_p2b_t = extract_metric(train_p2b, 'train/success_rate')
_, p8_p2b_t = extract_metric(train_p2b, 'train/pass@8')

ax2.plot(s_p3_t, sr_p3_t, color=c_p3, linewidth=1.6, label='p3 (Async): train/success_rate')
ax2.plot(s_p3_t, p8_p3_t, linestyle='--', color=c_p3_light, linewidth=1.6, label='p3 (Async): train/pass@8')
ax2.plot(s_p2b_t, sr_p2b_t, linestyle=':', color=c_p2b, alpha=0.7, linewidth=1.3, label='p2b (Sync): train/success_rate')
ax2.plot(s_p2b_t, p8_p2b_t, linestyle=':', color=c_p2b_light, alpha=0.7, linewidth=1.3, label='p2b (Sync): train/pass@8')

ax2.set_title('Reward Quality: Training Metrics', fontweight='bold', fontsize=11)
ax2.set_xlabel('Training Step', fontsize=10)
ax2.set_ylabel('Rate', fontsize=10)
ax2.set_ylim(-0.05, 1.05)
ax2.legend(loc='lower right', fontsize=8, framealpha=0.9)
ax2.grid(True, linestyle=':', alpha=0.6)

# -------------------------------------------------------------
# 3. Optimization: Gradient Norm
# -------------------------------------------------------------
ax3 = axes[0, 2]
s_p3_gn, gn_p3 = extract_metric(train_p3, 'train/grad_norm')
s_p2b_gn, gn_p2b = extract_metric(train_p2b, 'train/grad_norm')

ax3.plot(s_p3_gn, gn_p3, marker='o', markersize=3, color=c_p3, linewidth=1.4, label='p3 (Async): train/grad_norm')
ax3.plot(s_p2b_gn, gn_p2b, marker='^', markersize=3, color=c_p2b, linestyle='--', linewidth=1.3, alpha=0.7, label='p2b (Sync): train/grad_norm')

ax3.set_title('Optimization: Gradient Norm', fontweight='bold', fontsize=11)
ax3.set_xlabel('Training Step', fontsize=10)
ax3.set_ylabel('train/grad_norm', fontsize=10)
ax3.legend(loc='upper right', fontsize=8.5, framealpha=0.9)
ax3.grid(True, linestyle=':', alpha=0.6)

# -------------------------------------------------------------
# 4. Wall-Clock Efficiency: Step Time (Timing Comparison)
# -------------------------------------------------------------
ax4 = axes[0, 3]
s_p3_st, st_p3 = extract_metric(train_p3, 'train/step_time_s')
s_p2b_st, st_p2b = extract_metric(train_p2b, 'train/step_time_s')

ax4.plot(s_p3_st, st_p3, marker='o', markersize=3, color=c_p3, linewidth=1.5, 
         label=f'p3 (Async): mean={np.mean(st_p3):.1f}s')
ax4.plot(s_p2b_st, st_p2b, marker='s', markersize=3, linestyle='--', color=c_p2b, linewidth=1.3, alpha=0.75, 
         label=f'p2b (Sync): mean={np.mean(st_p2b):.1f}s')

ax4.set_title('Timing: Wall-Clock Step Time', fontweight='bold', fontsize=11)
ax4.set_xlabel('Training Step', fontsize=10)
ax4.set_ylabel('train/step_time_s (seconds)', fontsize=10)
ax4.legend(loc='upper right', fontsize=8.5, framealpha=0.9)
ax4.grid(True, linestyle=':', alpha=0.6)

# -------------------------------------------------------------
# 5. Policy Drift: Importance Weights (Distribution Tail)
# -------------------------------------------------------------
ax5 = axes[1, 0]
s_p3_iw, iw_mean = extract_metric(train_p3, 'train/importance_weight_mean')
_, iw_max = extract_metric(train_p3, 'train/importance_weight_max')
_, iw_min = extract_metric(train_p3, 'train/importance_weight_min')

ax5.plot(s_p3_iw, iw_mean, color=c_p3_dark, linewidth=2, label='train/importance_weight_mean')
ax5.plot(s_p3_iw, iw_max, linestyle='--', color=c_warn, linewidth=1.5, label='train/importance_weight_max')
ax5.plot(s_p3_iw, iw_min, linestyle='-.', color='#7B1FA2', linewidth=1.5, label='train/importance_weight_min')
ax5.fill_between(s_p3_iw, iw_min, iw_max, color=c_p3_light, alpha=0.2, label='IW Range [min, max]')
ax5.axhline(1.0, color='gray', linestyle=':', alpha=0.8, label='On-Policy Reference (1.0)')

ax5.set_title('Policy Drift: IW Mean, Min & Max (Heavy Tail)', fontweight='bold', fontsize=11)
ax5.set_xlabel('Training Step', fontsize=10)
ax5.set_ylabel('Importance Weight', fontsize=10)
ax5.legend(loc='upper right', fontsize=8, framealpha=0.9)
ax5.grid(True, linestyle=':', alpha=0.6)

# -------------------------------------------------------------
# 6. Policy Drift: Importance Weight Standard Deviation
# -------------------------------------------------------------
ax6 = axes[1, 1]
s_p3_std, iw_std_p3 = extract_metric(train_p3, 'train/importance_weight_std')
s_p2b_std, iw_std_p2b = extract_metric(train_p2b, 'train/importance_weight_std')

ax6.plot(s_p3_std, iw_std_p3, marker='o', markersize=3, color=c_p3, linewidth=1.5, 
         label=f'p3 (Async): mean={np.mean(iw_std_p3):.4f}')
ax6.plot(s_p2b_std, iw_std_p2b, marker='^', markersize=3, linestyle='--', color=c_p2b, linewidth=1.3, alpha=0.75, 
         label=f'p2b (Sync): mean={np.mean(iw_std_p2b):.4f}')

ax6.set_title('Policy Drift: Importance Weight Std', fontweight='bold', fontsize=11)
ax6.set_xlabel('Training Step', fontsize=10)
ax6.set_ylabel('train/importance_weight_std', fontsize=10)
ax6.legend(loc='upper right', fontsize=8.5, framealpha=0.9)
ax6.grid(True, linestyle=':', alpha=0.6)

# -------------------------------------------------------------
# 7. Data Health: Fraction of Degenerate Groups
# -------------------------------------------------------------
ax7 = axes[1, 2]
s_p3_dg, degen_p3 = extract_metric(train_p3, 'train/frac_degenerate_groups')
s_p2b_dg, degen_p2b = extract_metric(train_p2b, 'train/frac_degenerate_groups')

ax7.plot(s_p3_dg, degen_p3, color=c_p3, linewidth=1.6, label='p3 (Async): train/frac_degenerate_groups')
ax7.plot(s_p2b_dg, degen_p2b, linestyle='--', color=c_p2b, linewidth=1.4, alpha=0.7, label='p2b (Sync): train/frac_degenerate_groups')

ax7.set_title('Data Health: Degenerate Groups Fraction', fontweight='bold', fontsize=11)
ax7.set_xlabel('Training Step', fontsize=10)
ax7.set_ylabel('train/frac_degenerate_groups', fontsize=10)
ax7.set_ylim(-0.05, 1.05)
ax7.legend(loc='center right', fontsize=8, framealpha=0.9)
ax7.grid(True, linestyle=':', alpha=0.6)

# -------------------------------------------------------------
# 8. Data Health: Policy Entropy Proxy
# -------------------------------------------------------------
ax8 = axes[1, 3]
s_p3_ent, ent_p3 = extract_metric(train_p3, 'train/entropy_proxy')
s_p2b_ent, ent_p2b = extract_metric(train_p2b, 'train/entropy_proxy')

ax8.plot(s_p3_ent, ent_p3, color=c_p3, linewidth=1.6, label='p3 (Async): train/entropy_proxy')
ax8.plot(s_p2b_ent, ent_p2b, linestyle='--', color=c_p2b, linewidth=1.4, alpha=0.7, label='p2b (Sync): train/entropy_proxy')

ax8.set_title('Data Health: Entropy Proxy', fontweight='bold', fontsize=11)
ax8.set_xlabel('Training Step', fontsize=10)
ax8.set_ylabel('train/entropy_proxy', fontsize=10)
ax8.legend(loc='upper right', fontsize=8.5, framealpha=0.9)
ax8.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('assets/p3_diagnostic_plots.png', bbox_inches='tight')
print('Standard diagnostic plots for Part 3 saved to assets/p3_diagnostic_plots.png')
