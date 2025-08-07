import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import seaborn as sns
from collections import defaultdict
import re

class GradientNormsAnalyzer:
    def __init__(self, json_file_path):
        """
        Initialize the analyzer with path to JSON file containing gradient norms.
        Expected format: one JSON object per line with step info and gradient norms.
        """
        self.json_file_path = Path(json_file_path)
        self.data = []
        self.df = None
        self.load_data()
        
    def load_data(self):
        """Load and parse the JSON data from file."""
        print(f"Loading data from {self.json_file_path}...")
        
        with open(self.json_file_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data_point = json.loads(line.strip())
                    self.data.append(data_point)
                except json.JSONDecodeError as e:
                    print(f"Warning: Could not parse line {line_num}: {e}")
                    
        print(f"Loaded {len(self.data)} data points")
        self.create_dataframe()
    
    def create_dataframe(self):
        """Convert loaded data into a pandas DataFrame for easier analysis."""
        rows = []
        
        for ith, entry in enumerate(self.data):
            # step = entry.get('num_samples', entry.get('epoch', 0))
            step = ith * 5
            
            # Handle different possible data structures
            if 'gradients' in entry:
                grad_data = entry['gradients']
            elif 'grad_norms' in entry:
                grad_data = entry['grad_norms']
            else:
                # Assume all keys except step/epoch are gradient data
                grad_data = {k: v for k, v in entry.items() 
                           if 'stream' not in k and ('q_cells' in k or '0' in k)}
            
            for param_name, norm_value in grad_data.items():
                rows.append({
                    'num_samples': step,
                    'parameter': param_name,
                    'grad_norm': float(norm_value),
                    'layer_type': self.extract_layer_type(param_name),
                    'layer_depth': self.extract_layer_depth(param_name)
                })
       
        self.df = pd.DataFrame(rows)
        print(f"Created DataFrame with {len(self.df)} gradient norm records")
    
    def extract_layer_type(self, param_name):
        """Extract layer type from parameter name."""
        param_name_lower = param_name.lower()
        
        # Handle your specific naming patterns
        if param_name_lower.startswith('embeds.'):
            if '.embed.' in param_name_lower:
                return 'embedding'
            elif '.unembed.' in param_name_lower:
                return 'unembedding'
            elif '.ln_final.' in param_name_lower:
                return 'layer_norm_final'
            elif 'proj_heads_q' in param_name_lower:
                return 'attention_q'
            elif 'proj_heads_k' in param_name_lower:
                return 'attention_k'
            elif 'proj_heads_v' in param_name_lower:
                return 'attention_v'
            elif 'proj_out' in param_name_lower:
                return 'attention_out'
            elif '.layers.' in param_name_lower and ('weight' in param_name_lower or 'bias' in param_name_lower):
                return 'ffn'
            else:
                return 'embeds_other'
        
        elif param_name_lower.startswith('ae_local_blocks.'):
            if 'proj_heads_q' in param_name_lower:
                return 'ae_local_attention_q'
            elif 'proj_heads_k' in param_name_lower:
                return 'ae_local_attention_k'
            elif 'proj_heads_v' in param_name_lower:
                return 'ae_local_attention_v'
            elif 'proj_out' in param_name_lower:
                return 'ae_local_attention_out'
            elif '.layers.' in param_name_lower:
                return 'ae_local_ffn'
            else:
                return 'ae_local_other'
        
        elif param_name_lower.startswith('ae_global_blocks.'):
            if 'proj_heads_q' in param_name_lower:
                return 'ae_global_attention_q'
            elif 'proj_heads_k' in param_name_lower:
                return 'ae_global_attention_k'
            elif 'proj_heads_v' in param_name_lower:
                return 'ae_global_attention_v'
            elif 'proj_out' in param_name_lower:
                return 'ae_global_attention_out'
            elif '.layers.' in param_name_lower:
                return 'ae_global_ffn'
            else:
                return 'ae_global_other'
        
        elif param_name_lower.startswith('ae_adapter.'):
            if 'proj_heads_q' in param_name_lower:
                return 'ae_adapter_attention_q'
            elif 'proj_heads_k' in param_name_lower:
                return 'ae_adapter_attention_k'
            elif 'proj_heads_v' in param_name_lower:
                return 'ae_adapter_attention_v'
            elif 'proj_out' in param_name_lower:
                return 'ae_adapter_attention_out'
            elif '.layers.' in param_name_lower:
                return 'ae_adapter_ffn'
            else:
                return 'ae_adapter_other'
        
        elif param_name_lower.startswith('target_token_engines.'):
            if 'proj_heads_q' in param_name_lower:
                return 'tte_attention_q'
            elif 'proj_heads_k' in param_name_lower:
                return 'tte_attention_k'
            elif 'proj_heads_v' in param_name_lower:
                return 'tte_attention_v'
            elif 'proj_out' in param_name_lower:
                return 'tte_attention_out'
            elif 'embed_aux' in param_name_lower:
                return 'tte_embed_aux'
            elif 'lnorm' in param_name_lower:
                return 'tte_layer_norm'
            elif '.layers.' in param_name_lower:
                return 'tte_ffn'
            else:
                return 'tte_other'
        
        elif param_name_lower.startswith('embed_target_coords.'):
            return 'target_coords_embedding'
        
        elif param_name_lower.startswith('pred_heads.'):
            return 'prediction_head'
        
        # Fallback for standard patterns (if any)
        elif 'embed' in param_name_lower:
            return 'embedding'
        elif 'attention' in param_name_lower or 'attn' in param_name_lower:
            if 'q_proj' in param_name_lower or 'query' in param_name_lower:
                return 'attention_q'
            elif 'k_proj' in param_name_lower or 'key' in param_name_lower:
                return 'attention_k'
            elif 'v_proj' in param_name_lower or 'value' in param_name_lower:
                return 'attention_v'
            elif 'o_proj' in param_name_lower or 'out' in param_name_lower:
                return 'attention_out'
            else:
                return 'attention'
        elif 'layernorm' in param_name_lower or 'layer_norm' in param_name_lower or 'ln' in param_name_lower:
            return 'layernorm'
        else:
            return 'other'
    
    def extract_layer_depth(self, param_name):
        """Extract layer depth/index from parameter name."""
        param_name_lower = param_name.lower()
        
        # Look for patterns specific to your architecture
        patterns = [
            # embeds.0.layers.N.* (transformer layers within embeds)
            r'embeds\.\d+\.layers\.(\d+)\.',
            # embeds.0.unembed.N.* (unembedding layers)
            r'embeds\.\d+\.unembed\.(\d+)\.',
            # embeds.0.ln_final.N.* (final layer norms)
            r'embeds\.\d+\.ln_final\.(\d+)\.',
            # ae_local_blocks.N.* (autoencoder local blocks)
            r'ae_local_blocks\.(\d+)\.',
            # ae_global_blocks.N.* (autoencoder global blocks)
            r'ae_global_blocks\.(\d+)\.',
            # ae_adapter.N.* (autoencoder adapter blocks)
            r'ae_adapter\.(\d+)\.',
            # target_token_engines.0.tte.N.* (target token engine blocks)
            r'target_token_engines\.\d+\.tte\.(\d+)\.',
            # target_token_engines.0.tte.N.block.M.* (nested blocks)
            r'target_token_engines\.\d+\.tte\.(\d+)\.block\.(\d+)\.',
            # pred_heads.0.pred_heads.0.N.* (prediction head layers)
            r'pred_heads\.\d+\.pred_heads\.\d+\.(\d+)\.',
            # Generic patterns for any numbered layers
            r'layer[s]?\.(\d+)',
            r'h\.(\d+)', 
            r'transformer\.(\d+)',
            r'blocks\.(\d+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, param_name_lower)
            if match:
                # For nested patterns (like tte blocks), combine indices
                if len(match.groups()) > 1:
                    # Combine indices: e.g., tte.1.block.2 -> 12 (or 1*10+2)
                    return int(match.group(1)) * 10 + int(match.group(2))
                else:
                    return int(match.group(1))
        
        # Special handling for components without clear depth
        if param_name_lower.startswith('embed_target_coords.'):
            return 0  # Coordinate embeddings at the start
        elif 'total_grad_norm' in param_name_lower:
            return -2  # Special marker for total norm
        elif any(x in param_name_lower for x in ['weathergen', 'stage', 'q_cells']):
            return -3  # Special marker for metadata
        
        return -1  # Unknown depth
    
    def plot_total_gradient_norms(self, figsize=(12, 6)):
        """Plot total gradient norm over training steps."""
        # Calculate total norm per step
        total_norms = []
        steps = []
        
        for ith, entry in enumerate(self.data):
            # step = entry.get('num_samples', entry.get('epoch', 0))
            step = ith * 5
            
            if 'gradients' in entry:
                grad_data = entry['gradients']
            elif 'grad_norms' in entry:
                grad_data = entry['grad_norms']
            else:
                grad_data = {k: v for k, v in entry.items() 
                             if 'q_cells' in k or '0' in k}

            if len(grad_data) == 0:
                continue
            
            # Calculate total norm (L2 norm of all gradients)
            total_norm = np.sqrt(sum(float(v)**2 for v in grad_data.values()))
            total_norms.append(total_norm)
            steps.append(step)
        
        plt.figure(figsize=figsize)
        plt.plot(steps, total_norms, linewidth=1.5, alpha=0.8)
        plt.xlabel('Training Step')
        plt.ylabel('Total Gradient Norm')
        plt.title('Total Gradient Norm vs Training Steps')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("plots/total_grad_norm.png")
        
        return steps, total_norms
    
    def plot_layer_type_norms(self, figsize=(14, 8)):
        """Plot gradient norms grouped by layer type."""
        if self.df is None:
            print("No DataFrame available. Load data first.")
            return
        
        plt.figure(figsize=figsize)
        
        # Get unique layer types
        layer_types = self.df['layer_type'].unique()
        print(layer_types)
        colors = plt.cm.tab10(np.linspace(0, 1, len(layer_types)))
        
        for i, layer_type in enumerate(layer_types):
            layer_data = self.df[self.df['layer_type'] == layer_type]
            
            # Calculate mean gradient norm per step for this layer type
            mean_norms = layer_data.groupby('num_samples')['grad_norm'].mean()
            
            plt.plot(mean_norms.index, mean_norms.values, 
                    label=layer_type, color=colors[i], alpha=0.8)
        
        plt.xlabel('Training Step')
        plt.ylabel('Mean Gradient Norm')
        plt.title('Gradient Norms by Layer Type')
        plt.yscale('log')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("plots/grad_norm_by_layer_type.png")
    
    def plot_layer_depth_analysis(self, figsize=(12, 8)):
        """Plot gradient norms by layer depth."""
        if self.df is None:
            print("No DataFrame available. Load data first.")
            return
        
        # Filter out unknown depths
        depth_data = self.df[self.df['layer_depth'] >= 0]
        
        if len(depth_data) == 0:
            print("No layer depth information found in parameter names.")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
        
        # Plot 1: Mean gradient norm by depth over time
        depths = sorted(depth_data['layer_depth'].unique())
        colors = plt.cm.viridis(np.linspace(0, 1, len(depths)))
        
        for i, depth in enumerate(depths):
            layer_data = depth_data[depth_data['layer_depth'] == depth]
            mean_norms = layer_data.groupby('num_samples')['grad_norm'].mean()
            
            ax1.plot(mean_norms.index, mean_norms.values, 
                    label=f'Layer {depth}', color=colors[i], alpha=0.8)
        
        ax1.set_xlabel('Training Step')
        ax1.set_ylabel('Mean Gradient Norm')
        ax1.set_title('Gradient Norms by Layer Depth')
        ax1.set_yscale('log')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Heatmap of gradient norms by depth and step
        pivot_data = depth_data.groupby(['num_samples', 'layer_depth'])['grad_norm'].mean().unstack()
        
        # Sample data if too many steps for readability
        if len(pivot_data) > 100:
            sample_idx = np.linspace(0, len(pivot_data)-1, 100, dtype=int)
            pivot_data = pivot_data.iloc[sample_idx]
        
        im = ax2.imshow(pivot_data.T, aspect='auto', cmap='viridis', 
                       extent=[pivot_data.index.min(), pivot_data.index.max(),
                              pivot_data.columns.min(), pivot_data.columns.max()])
        ax2.set_xlabel('Training Step')
        ax2.set_ylabel('Layer Depth')
        ax2.set_title('Gradient Norm Heatmap (Layer Depth vs Step)')
        
        cbar = plt.colorbar(im, ax=ax2)
        cbar.set_label('Gradient Norm')
        
        plt.tight_layout()
        plt.savefig("plots/grad_norm_heatmap.png")
    
    def plot_gradient_distribution(self, figsize=(15, 10)):
        """Plot distribution of gradient norms."""
        if self.df is None:
            print("No DataFrame available. Load data first.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Plot 1: Histogram of all gradient norms
        axes[0, 0].hist(np.log10(self.df['grad_norm'].values), bins=50, alpha=0.7)
        axes[0, 0].set_xlabel('Log10(Gradient Norm)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Distribution of Gradient Norms (Log Scale)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Box plot by layer type
        layer_types = self.df['layer_type'].unique()[:10]  # Limit to 10 for readability
        plot_data = [np.log10(self.df[self.df['layer_type'] == lt]['grad_norm'].values) 
                    for lt in layer_types]
        
        axes[0, 1].boxplot(plot_data, labels=layer_types)
        axes[0, 1].set_xlabel('Layer Type')
        axes[0, 1].set_ylabel('Log10(Gradient Norm)')
        axes[0, 1].set_title('Gradient Norm Distribution by Layer Type')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Gradient norms over time (sample of parameters)
        sample_params = self.df['parameter'].unique()[:20]  # Sample 20 parameters
        for param in sample_params:
            param_data = self.df[self.df['parameter'] == param]
            axes[1, 0].plot(param_data['num_samples'], param_data['grad_norm'], 
                          alpha=0.6, linewidth=0.8)
        
        axes[1, 0].set_xlabel('Training Step')
        axes[1, 0].set_ylabel('Gradient Norm')
        axes[1, 0].set_title('Individual Parameter Gradient Norms (Sample)')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Statistics over time
        stats_by_step = self.df.groupby('num_samples')['grad_norm'].agg(['mean', 'std', 'min', 'max'])
       
        axes[1, 1].fill_between(stats_by_step.index, 
                              stats_by_step['mean'] - stats_by_step['std'],
                              stats_by_step['mean'] + stats_by_step['std'],
                              alpha=0.3, label='±1 std')
        axes[1, 1].plot(stats_by_step.index, stats_by_step['mean'], 
                       label='Mean', linewidth=2)
        axes[1, 1].plot(stats_by_step.index, stats_by_step['max'], 
                       label='Max', linewidth=1, alpha=0.8)
        axes[1, 1].plot(stats_by_step.index, stats_by_step['min'], 
                       label='Min', linewidth=1, alpha=0.8)
        
        axes[1, 1].set_xlabel('Training Step')
        axes[1, 1].set_ylabel('Gradient Norm')
        axes[1, 1].set_title('Gradient Norm Statistics Over Time')
        axes[1, 1].set_yscale('log')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("plots/grad_norm_over_time.png")
    
    def generate_summary_report(self):
        """Generate a summary report of gradient norm statistics."""
        if self.df is None:
            print("No DataFrame available. Load data first.")
            return
        
        print("=== GRADIENT NORMS ANALYSIS REPORT ===")
        print(f"Total data points: {len(self.df)}")
        print(f"Training steps: {self.df['num_samples'].nunique()}")
        print(f"Unique parameters: {self.df['parameter'].nunique()}")
        print()
        
        print("Overall Statistics:")
        print(f"Mean gradient norm: {self.df['grad_norm'].mean():.6f}")
        print(f"Median gradient norm: {self.df['grad_norm'].median():.6f}")
        print(f"Min gradient norm: {self.df['grad_norm'].min():.6f}")
        print(f"Max gradient norm: {self.df['grad_norm'].max():.6f}")
        print()
        
        print("Statistics by Layer Type:")
        layer_stats = self.df.groupby('layer_type')['grad_norm'].agg(['count', 'mean', 'std', 'min', 'max'])
        print(layer_stats)
        print()
        
        # Check for potential issues
        print("Potential Issues:")
        very_small = (self.df['grad_norm'] < 1e-6).sum()
        very_large = (self.df['grad_norm'] > 10.0).sum()
        
        if very_small > 0:
            print(f"⚠️  {very_small} gradient norms < 1e-6 (possible vanishing gradients)")
        if very_large > 0:
            print(f"⚠️  {very_large} gradient norms > 10.0 (possible exploding gradients)")
        
        if very_small == 0 and very_large == 0:
            print("✅ No obvious gradient issues detected")

# Usage example
def analyze_gradient_file(json_file_path):
    """
    Main function to analyze gradient norms from a JSON file.
    
    Usage:
    analyze_gradient_file('gradient_norms.jsonl')
    """
    
    analyzer = GradientNormsAnalyzer(json_file_path)
    
    # Generate summary report
    analyzer.generate_summary_report()
    
    # Create all plots
    print("\n=== GENERATING PLOTS ===")
    
    print("1. Total gradient norms over time...")
    analyzer.plot_total_gradient_norms()
    
    print("2. Gradient norms by layer type...")
    analyzer.plot_layer_type_norms()
    
    print("3. Layer depth analysis...")
    analyzer.plot_layer_depth_analysis()
    
    print("4. Gradient distribution analysis...")
    analyzer.plot_gradient_distribution()
    
    return analyzer

# Example usage:
# uv run python src/weathergen/utils/plot_grad_norms.py results/yvhxm2jc/yvhxm2jc_train_metrics.json
if __name__ == '__main__':
    import sys
    analyzer = analyze_gradient_file(sys.argv[1])
