#!/usr/bin/env python3
"""
Export PyTorch SPHNCA model weights to JSON format for JavaScript implementation.
"""

import torch
import json
import argparse
import os
import sys

def export_model_weights(checkpoint_path, output_path):
    """
    Load PyTorch model and export weights to JSON format.
    
    Args:
        checkpoint_path: Path to the .pt checkpoint file
        output_path: Path to save the JSON weights file
    """
    
    # Load the checkpoint
    print(f"Loading checkpoint from: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print("Checkpoint loaded successfully")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return False
    
    # Extract model state dict
    if 'model_state_dict' in checkpoint:
        model_state = checkpoint['model_state_dict']
    elif 'model' in checkpoint:
        model_state = checkpoint['model']
    else:
        # Assume the checkpoint is the model state dict itself
        model_state = checkpoint
    
    print("Model state dict keys:", list(model_state.keys()))
    
    # Extract weights and biases from the neural network layers
    weights_data = {
        'layers': [],
        'config': {
            'input_features': None,
            'hidden_features': None,
            'output_features': None,
            'fire_rate': 0.5,
            'update_rule': 'gated',
            'h': 0.08
        }
    }
    
    # Look for linear layer weights
    layer_weights = {}
    for key, tensor in model_state.items():
        if 'model.' in key and ('weight' in key or 'bias' in key):
            # Extract layer info from key (e.g., 'model.0.weight' -> layer 0, weight)
            parts = key.split('.')
            if len(parts) >= 3:
                layer_idx = int(parts[1])
                param_type = parts[2]  # 'weight' or 'bias'
                
                if layer_idx not in layer_weights:
                    layer_weights[layer_idx] = {}
                
                layer_weights[layer_idx][param_type] = tensor.detach().cpu().numpy().tolist()
    
    # Sort layers by index and add to weights_data
    for layer_idx in sorted(layer_weights.keys()):
        layer_data = layer_weights[layer_idx]
        weights_data['layers'].append({
            'index': layer_idx,
            'weight': layer_data.get('weight'),
            'bias': layer_data.get('bias')
        })
        
        # Infer dimensions from first and last layers
        if layer_idx == 0 and 'weight' in layer_data:
            weights_data['config']['input_features'] = len(layer_data['weight'][0])
            weights_data['config']['hidden_features'] = len(layer_data['weight'])
        elif 'weight' in layer_data:
            weights_data['config']['output_features'] = len(layer_data['weight'])
    
    # Try to extract additional config from checkpoint
    if 'configs' in checkpoint:
        config = checkpoint['configs']
        print(config)
        if 'CELL_FIRE_RATE' in config:
            weights_data['config']['fire_rate'] = config['CELL_FIRE_RATE']
        if 'NCA_UPDATE' in config:
            weights_data['config']['update_rule'] = config['NCA_UPDATE']
        if 'H' in config:
            weights_data['config']['h'] = config['H']
        weights_data['config']['mode'] = "image" if config.get('LOSS') == 'mse_simple' else "texture"
    
    # Save to JSON
    print(f"Saving weights to: {output_path}")
    try:
        with open(output_path, 'w') as f:
            json.dump(weights_data, f, indent=2)
        print("Weights exported successfully!")
        
        # Print summary
        print("\nExported model summary:")
        print(f"  Number of layers: {len(weights_data['layers'])}")
        print(f"  Input features: {weights_data['config']['input_features']}")
        print(f"  Hidden features: {weights_data['config']['hidden_features']}")
        print(f"  Output features: {weights_data['config']['output_features']}")
        print(f"  Fire rate: {weights_data['config']['fire_rate']}")
        print(f"  Update rule: {weights_data['config']['update_rule']}")
        print(f"  Kernel radius (h): {weights_data['config']['h']}")
        print(f"  Mode: {weights_data['config']['mode']}")
        
        return True
        
    except Exception as e:
        print(f"Error saving weights: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Export SPHNCA model weights to JSON')
    parser.add_argument('--checkpoint', '-c', 
                       default='../code/checkpoints/sphnca-07180538-1000.pt',
                       help='Path to PyTorch checkpoint file')
    parser.add_argument('--output', '-o',
                       default='model_weights.json',
                       help='Output JSON file path')
    
    args = parser.parse_args()
    
    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file not found: {args.checkpoint}")
        print("Available checkpoints:")
        checkpoint_dir = os.path.dirname(args.checkpoint)
        if os.path.exists(checkpoint_dir):
            for f in os.listdir(checkpoint_dir):
                if f.endswith('.pt'):
                    print(f"  {os.path.join(checkpoint_dir, f)}")
        sys.exit(1)
    
    # Export weights
    success = export_model_weights(args.checkpoint, args.output)
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()
