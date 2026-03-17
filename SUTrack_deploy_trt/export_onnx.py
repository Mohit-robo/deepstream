import os
import sys
import argparse
import torch

env_path = os.path.join(os.path.dirname(__file__), '..')
if env_path not in sys.path:
    sys.path.append(env_path)

from lib.test.parameter.sutrack import parameters
from lib.models.sutrack import build_sutrack

def export_onnx(param_name, checkpoint, output_path):
    print(f"Loading parameters: {param_name}")
    params = parameters(param_name)
    
    # Optional overrides typically done in tracker
    if hasattr(params, 'checkpoint') and checkpoint:
        params.checkpoint = checkpoint
        
    cfg = params.cfg
    
    print("Building model...")
    model = build_sutrack(cfg)
    
    print(f"Loading weights from {params.checkpoint}")
    checkpoint_dict = torch.load(params.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint_dict['net'], strict=True)
    
    model.eval()
    
    # Create dummy inputs based on cfg
    template_sz = cfg.TEST.TEMPLATE_SIZE
    search_sz = cfg.TEST.SEARCH_SIZE
    
    print(f"Template size: {template_sz}, Search size: {search_sz}")
    
    # SUTRACK forward_encoder expects:
    # template_list, search_list, template_anno_list, text_src, task_index
    # The actual forward path when tracking expects:
    # template_list: list of Tensors (B, 3, H_z, W_z)
    # search_list: list of Tensors (B, 3, H_x, W_x)
    # template_anno_list: list of Tensors (B, 1, H_z, W_z)
    
    dummy_template = torch.randn(1, 6, template_sz, template_sz)
    dummy_search = torch.randn(1, 6, search_sz, search_sz)
    dummy_anno = torch.tensor([[[0.5, 0.5, 0.5, 0.5]]])  # (batch, num_templates, 4)  or (batch, 4) in the list
    dummy_anno = dummy_anno.view(1, 4)  # Each element in template_anno_list is expected to be (B, 4)
    
    # Replicate lists based on number of templates
    num_template = cfg.TEST.NUM_TEMPLATES
    print(f"Num templates: {num_template}")
    
    template_list = [dummy_template] * num_template
    search_list = [dummy_search]
    template_anno_list = [dummy_anno] * num_template
    
    # We will export the full forward pass (encoder + decoder + task_decoder)
    # But SUTRACK's forward() has a 'mode' argument. ONNX export doesn't work well 
    # with control flow based on string arguments unless we export a specific wrapper.
    
    class SUTrackWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            
        def forward(self, template, search, template_anno):
            # ONNX prefers tensor inputs, not lists. So we wrap it.
            # Assuming num_template = 1 for simple deployment
            template_list = [template]
            search_list = [search]
            template_anno_list = [template_anno]
            
            # Encoder
            enc_opt = self.model.forward_encoder(template_list, search_list, template_anno_list, None, None)
            
            # Decoder
            out_dict = self.model.forward_decoder(feature=enc_opt)
            return out_dict['pred_boxes'], out_dict['score_map'], out_dict.get('size_map', torch.empty(0)), out_dict.get('offset_map', torch.empty(0))

    wrapper = SUTrackWrapper(model)
    
    # --- MONKEY PATCH FOR ONNX EXPORT ---
    # aten::unfold is not properly supported in ONNX opset 11/16 for this specific tensor operation.
    # The unfold here simply downsamples the (112x112) original mask into a (7x7) mask that matches the token count.
    # We can bypass it by using avg_pool2d which is fully supported by ONNX.
    original_prepare = model.encoder.body.prepare_tokens_with_masks
    
    def prepare_tokens_with_masks_patched(self, template_list, search_list, template_anno_list, text_src, task_index):
        import torch.nn.functional as F
        B = search_list[0].size(0)
        num_template = len(template_list)
        num_search = len(search_list)

        z = torch.stack(template_list, dim=1).view(-1, *template_list[0].size()[1:])
        x = torch.stack(search_list, dim=1).view(-1, *search_list[0].size()[1:])
        z_anno = torch.stack(template_anno_list, dim=1).view(-1, 4)
        
        if self.token_type_indicate:
            z_indicate_mask = self.create_mask(z, z_anno) # (b, h, w)
            # Replace unfold with exact equivalent avg_pool2d
            z_indicate_mask = z_indicate_mask.unsqueeze(1) # (b, 1, h, w)
            z_indicate_mask = F.avg_pool2d(z_indicate_mask, kernel_size=self.patch_size, stride=self.patch_size)
            z_indicate_mask = z_indicate_mask.flatten(1) # (b, 49)

        if self.token_type_indicate:
            template_background_token = self.template_background_token.unsqueeze(0).unsqueeze(1).expand(z_indicate_mask.size(0), z_indicate_mask.size(1), self.embed_dim)
            template_foreground_token = self.template_foreground_token.unsqueeze(0).unsqueeze(1).expand(z_indicate_mask.size(0), z_indicate_mask.size(1), self.embed_dim)
            weighted_foreground = template_foreground_token * z_indicate_mask.unsqueeze(-1)
            weighted_background = template_background_token * (1 - z_indicate_mask.unsqueeze(-1))
            z_indicate = weighted_foreground + weighted_background

        z = self.patch_embed(z)
        x = self.patch_embed(x)
        
        if not self.convmlp and self.stop_grad_conv1:
            x = x.detach() * 0.9 + x * 0.1

        import torch.utils.checkpoint as checkpoint
        for blk in self.blocks[:-self.num_main_blocks]:
            z = checkpoint.checkpoint(blk, z) if self.grad_ckpt else blk(z)
            x = checkpoint.checkpoint(blk, x) if self.grad_ckpt else blk(x)

        x = x.flatten(2).transpose(1, 2)
        z = z.flatten(2).transpose(1, 2)

        if self.pos_embed is not None:
            x = x + self.pos_embed[:, :self.num_patches_search, :]
            z = z + self.pos_embed[:, self.num_patches_search:, :]

        if self.token_type_indicate:
            x_indicate = self.search_token.unsqueeze(0).unsqueeze(1).expand(x.size(0), x.size(1), self.embed_dim)
            x = x + x_indicate
            z = z + z_indicate

        z = z.view(-1, num_template, z.size(-2), z.size(-1)).reshape(z.size(0), -1, z.size(-1))
        x = x.view(-1, num_search, x.size(-2), x.size(-1)).reshape(x.size(0), -1, x.size(-1))

        if text_src is not None:
            xz = torch.cat([x, z, text_src], dim=1)
        else:
            xz = torch.cat([x, z], dim=1)

        if self.cls_token is not None:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            xz = torch.cat([cls_tokens, xz], dim=1)

        return xz

    # Apply monkey patch
    import types
    model.encoder.body.prepare_tokens_with_masks = types.MethodType(prepare_tokens_with_masks_patched, model.encoder.body)
    # ------------------------------------

    input_names = ['template', 'search', 'template_anno']
    output_names = ['pred_boxes', 'score_map', 'size_map', 'offset_map']
    
    print(f"Exporting to {output_path}...")
    torch.onnx.export(
        wrapper,
        (dummy_template, dummy_search, dummy_anno),
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={
            'template': {0: 'batch_size'},
            'search': {0: 'batch_size'},
            'template_anno': {0: 'batch_size'},
            'pred_boxes': {0: 'batch_size'},
            'score_map': {0: 'batch_size'}
        }
    )
    print("Export successful!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--param', type=str, default='sutrack_b224')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--output', type=str, default='sutrack.onnx')
    args = parser.parse_args()
    
    export_onnx(args.param, args.checkpoint, args.output)
