import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads=8):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        B, T, _ = x.size()
        Q = self.query(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        
        attended = torch.matmul(weights, V)
        attended = attended.transpose(1, 2).contiguous().view(B, T, self.hidden_size)
        output = self.output(attended)
        return output, weights

class AdaptiveFusion(nn.Module):
    def __init__(self, hidden_size, num_agents=3):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_size * num_agents, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size * 2, num_agents),
            nn.Softmax(dim=1)
        )
        self.transform_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1)
            ) for _ in range(num_agents)
        ])
        self.fusion_net = nn.Sequential(
            nn.Linear(hidden_size * num_agents, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size * 2, hidden_size)
        )
        
    def forward(self, agent_outputs):
        transformed = [net(o) for net, o in zip(self.transform_nets, agent_outputs)]
        combined = torch.cat(transformed, dim=1)
        gates = self.gate_net(combined)
        weighted_sum = sum(gates[:, i:i+1] * transformed[i] for i in range(len(transformed)))
        deep_fused = self.fusion_net(combined)
        final = weighted_sum + deep_fused
        return final, gates

class MetaLearningModule(nn.Module):
    def __init__(self, hidden_size, num_agents=3):
        super().__init__()
        self.meta_extractor = nn.Sequential(
            nn.Linear(hidden_size + num_agents, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size)
        )
        self.modulation = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid()
        )
        
    def forward(self, fused, weights):
        meta_input = torch.cat([fused, weights], dim=1)
        meta_features = self.meta_extractor(meta_input)
        modulation = self.modulation(meta_features)
        return fused * modulation + meta_features * (1 - modulation)

class DistillationModule(nn.Module):
    def __init__(self, hidden_size, temperature=4.0):
        super().__init__()
        self.temperature = temperature
        self.feature_align = nn.ModuleDict({
            'gru_to_lstm': nn.Linear(hidden_size, hidden_size),
            'lstm_to_transformer': nn.Linear(hidden_size, hidden_size),
            'transformer_to_gru': nn.Linear(hidden_size, hidden_size)
        })
        
    def forward(self, gru_features, lstm_features, transformer_features):
        aligned_gru_to_lstm = self.feature_align['gru_to_lstm'](gru_features)
        aligned_lstm_to_transformer = self.feature_align['lstm_to_transformer'](lstm_features)
        aligned_transformer_to_gru = self.feature_align['transformer_to_gru'](transformer_features)
        
        return {
            'gru_to_lstm': aligned_gru_to_lstm,
            'lstm_to_transformer': aligned_lstm_to_transformer,
            'transformer_to_gru': aligned_transformer_to_gru
        }
    
    def compute_distillation_loss(self, teacher_features, student_features):
        teacher_norm = F.normalize(teacher_features, p=2, dim=1)
        student_norm = F.normalize(student_features, p=2, dim=1)
        
        teacher_soft = F.softmax(teacher_norm / self.temperature, dim=1)
        student_soft = F.log_softmax(student_norm / self.temperature, dim=1)
        
        distill_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean')
        feature_loss = F.mse_loss(student_features, teacher_features)
        
        return distill_loss + 0.5 * feature_loss

class CrossSupervisionModule(nn.Module):
    def __init__(self, hidden_size, num_tasks=4):
        super().__init__()
        self.num_tasks = num_tasks
        
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_size, 
            num_heads=8, 
            dropout=0.1,
            batch_first=True
        )
        
        self.consistency_net = nn.Sequential(
            nn.Linear(hidden_size * num_tasks, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_tasks)
        )
        
    def forward(self, task_features_dict):
        task_features = torch.stack(list(task_features_dict.values()), dim=1)
        
        attended_features, attention_weights = self.cross_attention(
            task_features, task_features, task_features
        )
        
        combined_features = attended_features.reshape(attended_features.size(0), -1)
        consistency_scores = self.consistency_net(combined_features)
        
        return attended_features, consistency_scores, attention_weights
    
    def compute_consistency_loss(self, task_predictions_dict):
        task_probs = {}
        for task, pred in task_predictions_dict.items():
            task_probs[task] = F.softmax(pred, dim=1)
        
        consistency_loss = 0.0
        count = 0
        
        task_names = list(task_probs.keys())
        for i in range(len(task_names)):
            for j in range(i+1, len(task_names)):
                task1, task2 = task_names[i], task_names[j]
                kl_loss = F.kl_div(
                    F.log_softmax(task_predictions_dict[task1], dim=1),
                    task_probs[task2],
                    reduction='batchmean'
                )
                consistency_loss += kl_loss
                count += 1
        
        return consistency_loss / count if count > 0 else torch.tensor(0.0, device=list(task_predictions_dict.values())[0].device)

class EnhancedGRUAgent(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, 3, batch_first=True, dropout=0.2, bidirectional=True)
        self.projection = nn.Linear(hidden_size*2, hidden_size)
        self.self_attention = SelfAttention(hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        
    def forward(self, x):
        out, _ = self.gru(x)
        out = self.projection(out)
        attended, _ = self.self_attention(out)
        final = self.norm(out + attended)
        return final[:, -1], final

class EnhancedLSTMAgent(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, 3, batch_first=True, dropout=0.2, bidirectional=True)
        self.projection = nn.Linear(hidden_size*2, hidden_size)
        self.self_attention = SelfAttention(hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.projection(out)
        attended, _ = self.self_attention(out)
        final = self.norm(out + attended)
        return final[:, -1], final

class EnhancedTransformerAgent(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_projection = nn.Linear(input_size, hidden_size)
        self.pos_encoding = nn.Parameter(torch.randn(1000, hidden_size))
        encoder_layer = nn.TransformerEncoderLayer(
            hidden_size, 16, hidden_size*4, dropout=0.2, 
            activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, 6)
        self.norm = nn.LayerNorm(hidden_size)
        
    def forward(self, x):
        B, T, _ = x.size()
        x = self.input_projection(x)
        x = x + self.pos_encoding[:T].unsqueeze(0)
        out = self.transformer(x)
        final = self.norm(out)
        return final[:, -1], final

class SuperMAA(nn.Module):
    def __init__(self, input_size=4, hidden_size=512, available_tasks=['cd1','cd2','cd3','va'],
                 distillation_weight=0.5, cross_supervision_weight=0.3, temperature=4.0):
        super().__init__()
        self.available_tasks = available_tasks
        self.distillation_weight = distillation_weight
        self.cross_supervision_weight = cross_supervision_weight
        
        self.gru_agent = EnhancedGRUAgent(input_size, hidden_size)
        self.lstm_agent = EnhancedLSTMAgent(input_size, hidden_size)
        self.transformer_agent = EnhancedTransformerAgent(input_size, hidden_size)
        
        self.adaptive_fusion = AdaptiveFusion(hidden_size)
        self.meta_learning = MetaLearningModule(hidden_size)
        
        self.distillation = DistillationModule(hidden_size, temperature)
        self.cross_supervision = CrossSupervisionModule(hidden_size, len(available_tasks))
        
        self.task_heads = nn.ModuleDict({
            t: nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_size, hidden_size//2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_size//2, 2)
            ) for t in available_tasks
        })
        
        self.length_heads = nn.ModuleDict({
            f"{t}_length": nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_size, hidden_size//2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_size//2, 1)
            ) for t in available_tasks
        })
        
        self.task_feature_extractors = nn.ModuleDict({
            t: nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_size, hidden_size)
            ) for t in available_tasks
        })
        
        self.discriminator = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size//2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x, compute_losses=False):
        g, g_seq = self.gru_agent(x)
        l, l_seq = self.lstm_agent(x)
        t, t_seq = self.transformer_agent(x)
        
        agent_features = {'gru': g, 'lstm': l, 'transformer': t}
        
        distilled_features = self.distillation(g, l, t)
        
        fused, weights = self.adaptive_fusion([g, l, t])
        
        meta = self.meta_learning(fused, weights)
        
        task_features = {task: self.task_feature_extractors[task](meta) 
                        for task in self.available_tasks}
        
        cross_supervised_features, consistency_scores, cross_attention = self.cross_supervision(task_features)
        
        tasks = {}
        lengths = {}
        for i, task in enumerate(self.available_tasks):
            enhanced_feature = cross_supervised_features[:, i, :]
            tasks[task] = self.task_heads[task](enhanced_feature)
            lengths[f"{task}_length"] = self.length_heads[f"{task}_length"](enhanced_feature)
        
        disc = self.discriminator(meta)
        
        result = {
            "tasks": tasks, 
            "lengths": lengths, 
            "discriminator": disc,
            "fusion_weights": weights,
            "meta_features": meta,
            "agent_features": agent_features,
            "distilled_features": distilled_features,
            "task_features": task_features,
            "consistency_scores": consistency_scores,
            "cross_attention": cross_attention
        }
        
        return result
    
    def compute_distillation_losses(self, agent_features, distilled_features):
        distill_losses = {}
        
        distill_losses['gru_to_lstm'] = self.distillation.compute_distillation_loss(
            agent_features['gru'], distilled_features['gru_to_lstm']
        )
        
        distill_losses['lstm_to_transformer'] = self.distillation.compute_distillation_loss(
            agent_features['lstm'], distilled_features['lstm_to_transformer']
        )
        
        distill_losses['transformer_to_gru'] = self.distillation.compute_distillation_loss(
            agent_features['transformer'], distilled_features['transformer_to_gru']
        )
        
        total_distill_loss = sum(distill_losses.values())
        
        return total_distill_loss, distill_losses
    
    def compute_total_loss(self, outputs, targets):
        device = next(self.parameters()).device
        
        task_losses = {}
        length_losses = {}
        
        for task in self.available_tasks:
            if f'{task}_label' in targets:
                task_losses[task] = F.cross_entropy(
                    outputs['tasks'][task], 
                    targets[f'{task}_label']
                )
            
            if f'{task}_length' in targets:
                length_losses[f'{task}_length'] = F.mse_loss(
                    outputs['lengths'][f'{task}_length'].squeeze(),
                    targets[f'{task}_length']
                )
        
        main_task_loss = sum(task_losses.values()) + sum(length_losses.values())
        
        distill_loss, distill_details = self.compute_distillation_losses(
            outputs['agent_features'], 
            outputs['distilled_features']
        )
        
        cross_supervision_loss = self.cross_supervision.compute_consistency_loss(outputs['tasks'])
        
        total_loss = (main_task_loss + 
                     self.distillation_weight * distill_loss + 
                     self.cross_supervision_weight * cross_supervision_loss)
        
        loss_components = {
            'main_task_loss': main_task_loss.item(),
            'distillation_loss': distill_loss.item(),
            'cross_supervision_loss': cross_supervision_loss.item(),
            'total_loss': total_loss.item(),
            'task_losses': {k: v.item() for k, v in task_losses.items()},
            'length_losses': {k: v.item() for k, v in length_losses.items()},
            'distill_details': {k: v.item() for k, v in distill_details.items()}
        }
        
        return total_loss, loss_components

class GAN(nn.Module):
    def __init__(self, input_size=4, hidden_size=128, available_tasks=['cd1', 'cd2', 'cd3', 'va']):
        super().__init__()
        self.available_tasks = available_tasks
        
        self.generator = nn.Sequential(
            nn.Linear(input_size * 20, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size)
        )
        
        self.task_heads = nn.ModuleDict({
            task: nn.Sequential(
                nn.Linear(hidden_size, hidden_size//2),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_size//2, 2)
            ) for task in available_tasks
        })
        
        self.length_heads = nn.ModuleDict({
            f"{task}_length": nn.Sequential(
                nn.Linear(hidden_size, hidden_size//2),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_size//2, 1)
            ) for task in available_tasks
        })
        
        self.discriminator = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size//2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x_flat = x.view(x.size(0), -1)
        generated = self.generator(x_flat)
        
        tasks = {task: self.task_heads[task](generated) for task in self.available_tasks}
        lengths = {f"{task}_length": self.length_heads[f"{task}_length"](generated) 
                  for task in self.available_tasks}
        disc = self.discriminator(generated)
        
        return {"tasks": tasks, "lengths": lengths, "discriminator": disc}

class SimpleGRU(nn.Module):
    def __init__(self, input_size=4, hidden_size=128, available_tasks=['cd1', 'cd2', 'cd3', 'va']):
        super().__init__()
        self.available_tasks = available_tasks
        self.gru = nn.GRU(input_size, hidden_size, num_layers=2, batch_first=True, dropout=0.2)
        
        self.task_heads = nn.ModuleDict({
            task: nn.Sequential(
                nn.Linear(hidden_size, hidden_size//2),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_size//2, 2)
            ) for task in available_tasks
        })
        
        self.length_heads = nn.ModuleDict({
            f"{task}_length": nn.Sequential(
                nn.Linear(hidden_size, hidden_size//2),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_size//2, 1)
            ) for task in available_tasks
        })
    
    def forward(self, x):
        gru_out, _ = self.gru(x)
        final_out = gru_out[:, -1, :]
        
        tasks = {task: self.task_heads[task](final_out) for task in self.available_tasks}
        lengths = {f"{task}_length": self.length_heads[f"{task}_length"](final_out) 
                  for task in self.available_tasks}
        
        return {"tasks": tasks, "lengths": lengths}

class SimpleLSTM(nn.Module):
    def __init__(self, input_size=4, hidden_size=128, available_tasks=['cd1', 'cd2', 'cd3', 'va']):
        super().__init__()
        self.available_tasks = available_tasks
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True, dropout=0.2)
        
        self.task_heads = nn.ModuleDict({
            task: nn.Sequential(
                nn.Linear(hidden_size, hidden_size//2),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_size//2, 2)
            ) for task in available_tasks
        })
        
        self.length_heads = nn.ModuleDict({
            f"{task}_length": nn.Sequential(
                nn.Linear(hidden_size, hidden_size//2),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_size//2, 1)
            ) for task in available_tasks
        })
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        final_out = lstm_out[:, -1, :]
        
        tasks = {task: self.task_heads[task](final_out) for task in self.available_tasks}
        lengths = {f"{task}_length": self.length_heads[f"{task}_length"](final_out) 
                  for task in self.available_tasks}
        
        return {"tasks": tasks, "lengths": lengths}

class SimpleTransformer(nn.Module):
    def __init__(self, input_size=4, hidden_size=128, available_tasks=['cd1', 'cd2', 'cd3', 'va']):
        super().__init__()
        self.available_tasks = available_tasks
        
        self.input_projection = nn.Linear(input_size, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=4, dim_feedforward=hidden_size*2, 
            dropout=0.2, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        self.task_heads = nn.ModuleDict({
            task: nn.Sequential(
                nn.Linear(hidden_size, hidden_size//2),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_size//2, 2)
            ) for task in available_tasks
        })
        
        self.length_heads = nn.ModuleDict({
            f"{task}_length": nn.Sequential(
                nn.Linear(hidden_size, hidden_size//2),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_size//2, 1)
            ) for task in available_tasks
        })
    
    def forward(self, x):
        x_proj = self.input_projection(x)
        transformer_out = self.transformer(x_proj)
        final_out = transformer_out[:, -1, :]
        
        tasks = {task: self.task_heads[task](final_out) for task in self.available_tasks}
        lengths = {f"{task}_length": self.length_heads[f"{task}_length"](final_out) 
                  for task in self.available_tasks}
        
        return {"tasks": tasks, "lengths": lengths}

