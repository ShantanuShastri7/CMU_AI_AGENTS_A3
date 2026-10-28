import json
import matplotlib.pyplot as plt

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

train_p2a, eval_p2a = load_run('runs/p2a_number_search_reinforce.jsonl')
train_p2b, eval_p2b = load_run('runs/p2b_number_search_grpo.jsonl')

def extract_metric(records, key):
    steps = []
    vals = []
    for r in records:
        if key in r and r[key] is not None:
            steps.append(r['step'])
            vals.append(r[key])
    return steps, vals

# Set styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, axes = plt.subplots(2, 3, figsize=(18, 10), dpi=300)
fig.suptitle('Part 2: Standard Diagnostic Plots — REINFORCE vs. GRPO (Number Search)', fontsize=15, fontweight='bold', y=0.98)

# Colors
c_reinforce = '#E65100'       # Deep Orange
c_reinforce_alt = '#FFA726'   # Light Orange
c_grpo = '#1565C0'            # Deep Blue
c_grpo_alt = '#42A5F5'        # Light Blue

# -------------------------------------------------------------
# 1. Reward Quality: Evaluation
# -------------------------------------------------------------
ax1 = axes[0, 0]
s_a, sr_eval_a = extract_metric(eval_p2a, 'eval/success_rate')
_, p4_eval_a = extract_metric(eval_p2a, 'eval/pass@4')
s_b, sr_eval_b = extract_metric(eval_p2b, 'eval/success_rate')
_, p4_eval_b = extract_metric(eval_p2b, 'eval/pass@4')

ax1.plot(s_a, sr_eval_a, marker='o', markersize=6, color=c_reinforce, linewidth=2, label='REINFORCE: eval/success_rate')
ax1.plot(s_a, p4_eval_a, marker='s', markersize=5, linestyle='--', color=c_reinforce_alt, linewidth=1.8, label='REINFORCE: eval/pass@4')
ax1.plot(s_b, sr_eval_b, marker='^', markersize=6, color=c_grpo, linewidth=2, label='GRPO: eval/success_rate')
ax1.plot(s_b, p4_eval_b, marker='D', markersize=5, linestyle='--', color=c_grpo_alt, linewidth=1.8, label='GRPO: eval/pass@4')

ax1.set_title('Reward Quality: Evaluation Metrics', fontweight='bold', fontsize=11)
ax1.set_xlabel('Training Step', fontsize=10)
ax1.set_ylabel('Rate', fontsize=10)
ax1.set_ylim(-0.05, 1.05)
ax1.legend(loc='lower right', fontsize=8.5, framealpha=0.9)
ax1.grid(True, linestyle=':', alpha=0.6)

# -------------------------------------------------------------
# 2. Reward Quality: Training
# -------------------------------------------------------------
ax2 = axes[0, 1]
s_a, sr_train_a = extract_metric(train_p2a, 'train/success_rate')
_, p1_train_a = extract_metric(train_p2a, 'train/pass@1')
s_b, sr_train_b = extract_metric(train_p2b, 'train/success_rate')
_, p8_train_b = extract_metric(train_p2b, 'train/pass@8')

ax2.plot(s_a, sr_train_a, color=c_reinforce, linewidth=1.6, label='REINFORCE: train/success_rate')
ax2.plot(s_a, p1_train_a, linestyle=':', color=c_reinforce_alt, linewidth=1.4, label='REINFORCE: train/pass@1')
ax2.plot(s_b, sr_train_b, color=c_grpo, linewidth=1.6, label='GRPO: train/success_rate')
ax2.plot(s_b, p8_train_b, linestyle='--', color=c_grpo_alt, linewidth=1.6, label='GRPO: train/pass@8')

ax2.set_title('Reward Quality: Training Metrics', fontweight='bold', fontsize=11)
ax2.set_xlabel('Training Step', fontsize=10)
ax2.set_ylabel('Rate', fontsize=10)
ax2.set_ylim(-0.05, 1.05)
ax2.legend(loc='lower right', fontsize=8.5, framealpha=0.9)
ax2.grid(True, linestyle=':', alpha=0.6)

# -------------------------------------------------------------
# 3. Optimization: Gradient Norm
# -------------------------------------------------------------
ax3 = axes[0, 2]
s_a, gn_a = extract_metric(train_p2a, 'train/grad_norm')
s_b, gn_b = extract_metric(train_p2b, 'train/grad_norm')

ax3.plot(s_a, gn_a, marker='o', markersize=3, color=c_reinforce, linewidth=1.4, label='REINFORCE: train/grad_norm')
ax3.plot(s_b, gn_b, marker='^', markersize=3, color=c_grpo, linewidth=1.4, label='GRPO: train/grad_norm')

ax3.set_title('Optimization: Gradient Norm', fontweight='bold', fontsize=11)
ax3.set_xlabel('Training Step', fontsize=10)
ax3.set_ylabel('train/grad_norm', fontsize=10)
ax3.legend(loc='upper right', fontsize=8.5, framealpha=0.9)
ax3.grid(True, linestyle=':', alpha=0.6)

# -------------------------------------------------------------
# 4. Policy Drift: Importance Weights Mean & Max
# -------------------------------------------------------------
ax4 = axes[1, 0]
s_a, iw_mean_a = extract_metric(train_p2a, 'train/importance_weight_mean')
_, iw_max_a = extract_metric(train_p2a, 'train/importance_weight_max')
s_b, iw_mean_b = extract_metric(train_p2b, 'train/importance_weight_mean')
_, iw_max_b = extract_metric(train_p2b, 'train/importance_weight_max')

ax4.plot(s_a, iw_mean_a, color=c_reinforce, linewidth=1.6, label='REINFORCE: train/importance_weight_mean')
ax4.plot(s_a, iw_max_a, linestyle='--', color=c_reinforce_alt, linewidth=1.2, label='REINFORCE: train/importance_weight_max')
ax4.plot(s_b, iw_mean_b, color=c_grpo, linewidth=1.6, label='GRPO: train/importance_weight_mean')
ax4.plot(s_b, iw_max_b, linestyle='--', color=c_grpo_alt, linewidth=1.2, label='GRPO: train/importance_weight_max')
ax4.axhline(1.0, color='gray', linestyle=':', alpha=0.7, label='On-Policy Benchmark (1.0)')

ax4.set_title('Policy Drift: Importance Weights (Mean & Max)', fontweight='bold', fontsize=11)
ax4.set_xlabel('Training Step', fontsize=10)
ax4.set_ylabel('Importance Weight', fontsize=10)
ax4.legend(loc='upper right', fontsize=8, framealpha=0.9)
ax4.grid(True, linestyle=':', alpha=0.6)

# -------------------------------------------------------------
# 5. Policy Drift: Importance Weight Std
# -------------------------------------------------------------
ax5 = axes[1, 1]
s_a, iw_std_a = extract_metric(train_p2a, 'train/importance_weight_std')
s_b, iw_std_b = extract_metric(train_p2b, 'train/importance_weight_std')

ax5.plot(s_a, iw_std_a, marker='o', markersize=3, color=c_reinforce, linewidth=1.4, label='REINFORCE: train/importance_weight_std')
ax5.plot(s_b, iw_std_b, marker='^', markersize=3, color=c_grpo, linewidth=1.4, label='GRPO: train/importance_weight_std')

ax5.set_title('Policy Drift: Importance Weight Std', fontweight='bold', fontsize=11)
ax5.set_xlabel('Training Step', fontsize=10)
ax5.set_ylabel('train/importance_weight_std', fontsize=10)
ax5.legend(loc='upper right', fontsize=8.5, framealpha=0.9)
ax5.grid(True, linestyle=':', alpha=0.6)

# -------------------------------------------------------------
# 6. Data Health: Degenerate Groups & Entropy Proxy
# -------------------------------------------------------------
ax6 = axes[1, 2]
s_a, degen_a = extract_metric(train_p2a, 'train/frac_degenerate_groups')
s_b, degen_b = extract_metric(train_p2b, 'train/frac_degenerate_groups')
s_a_ent, ent_a = extract_metric(train_p2a, 'train/entropy_proxy')
s_b_ent, ent_b = extract_metric(train_p2b, 'train/entropy_proxy')

ax6.plot(s_a, degen_a, color=c_reinforce, linestyle='-', linewidth=1.5, label='REINFORCE: train/frac_degenerate_groups')
ax6.plot(s_b, degen_b, color=c_grpo, linestyle='-', linewidth=1.5, label='GRPO: train/frac_degenerate_groups')
ax6.plot(s_a_ent, ent_a, color=c_reinforce, linestyle=':', linewidth=1.5, label='REINFORCE: train/entropy_proxy')
ax6.plot(s_b_ent, ent_b, color=c_grpo, linestyle=':', linewidth=1.5, label='GRPO: train/entropy_proxy')

ax6.set_title('Data Health: Degenerate Groups & Entropy Proxy', fontweight='bold', fontsize=11)
ax6.set_xlabel('Training Step', fontsize=10)
ax6.set_ylabel('Metric Value', fontsize=10)
ax6.legend(loc='center right', fontsize=8, framealpha=0.9)
ax6.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('assets/p2_diagnostic_plots.png', bbox_inches='tight')
print('Standard diagnostic plots successfully saved to assets/p2_diagnostic_plots.png')
