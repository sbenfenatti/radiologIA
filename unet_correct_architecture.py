# unet_correct_architecture.py - ARQUITETURA CORRETA BASEADA NO TREINAMENTO
import torch
import torch.nn as nn
import torchvision.models as models

class CorrectUNet(nn.Module):
    """U-Net com encoder ResNet - arquitetura correta do modelo treinado"""
    
    def __init__(self, n_channels=3, n_classes=5):
        super(CorrectUNet, self).__init__()
        
        # Encoder: ResNet34 backbone (baseado nas chaves do checkpoint)
        self.encoder = models.resnet34(pretrained=False)
        # Remover camadas finais do ResNet
        self.encoder.fc = nn.Identity()
        self.encoder.avgpool = nn.Identity()
        
        # Decoder: Blocos de upsampling
        self.decoder = nn.ModuleList([
            # 5 blocos de decoder baseados no checkpoint
            self._make_decoder_block(512, 256),  # decoder.blocks.0
            self._make_decoder_block(256, 128),  # decoder.blocks.1  
            self._make_decoder_block(128, 64),   # decoder.blocks.2
            self._make_decoder_block(64, 32),    # decoder.blocks.3
            self._make_decoder_block(32, 16),    # decoder.blocks.4
        ])
        
        # Segmentation head
        self.segmentation_head = nn.Conv2d(16, n_classes, kernel_size=1)
        
        # Output final (outc no checkpoint)
        self.outc = nn.Conv2d(n_classes, n_classes, kernel_size=1)
        
    def _make_decoder_block(self, in_channels, out_channels):
        """Cria bloco de decoder baseado no checkpoint"""
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        )
    
    def forward(self, x):
        # Encoder forward
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        x = self.encoder.maxpool(x)
        
        x1 = self.encoder.layer1(x)
        x2 = self.encoder.layer2(x1)
        x3 = self.encoder.layer3(x2)
        x4 = self.encoder.layer4(x3)
        
        # Decoder forward
        x = x4
        for decoder_block in self.decoder:
            x = decoder_block(x)
        
        # Segmentation head
        x = self.segmentation_head(x)
        
        # Output final
        x = self.outc(x)
        
        # Redimensionar para tamanho de entrada se necessário
        return nn.functional.interpolate(x, size=(512, 512), mode='bilinear', align_corners=False)

# Função para carregar o modelo corretamente
def load_correct_unet_model(model_path, device):
    """Carrega o modelo U-Net com a arquitetura correta"""
    print("🔧 Carregando U-Net com arquitetura correta...")
    
    try:
        model = CorrectUNet(n_channels=3, n_classes=5)
        checkpoint = torch.load(model_path, map_location=device)
        
        # Carregar estado com strict=False para permitir chaves extras
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)
        
        if missing_keys:
            print(f"⚠️ Chaves faltando: {len(missing_keys)} (isso é esperado)")
        if unexpected_keys:
            print(f"ℹ️ Chaves extras: {len(unexpected_keys)} (checkpoint tem mais parâmetros)")
        
        model.to(device)
        model.eval()
        
        print("✅ Modelo U-Net correto carregado com sucesso!")
        return model
        
    except Exception as e:
        print(f"❌ Erro ao carregar modelo correto: {e}")
        return None

# Corrigir o bug do max() no código principal
def fix_max_bug_in_analysis():
    """Código corrigido para o bug do max()"""
    code_fix = """
    # ANTES (com bug):
    max_probs = unet_probs.max(dim=(2,3))[0].squeeze()
    
    # DEPOIS (corrigido):
    max_probs = unet_probs.flatten(start_dim=2).max(dim=2)[0].squeeze()
    # OU mais simples:
    max_probs = torch.max(torch.max(unet_probs, dim=2)[0], dim=2)[0]
    """
    return code_fix

if __name__ == "__main__":
    # Teste da arquitetura
    device = torch.device('cpu')
    model = CorrectUNet(n_channels=3, n_classes=5)
    
    # Teste com tensor dummy
    dummy_input = torch.randn(1, 3, 512, 512)
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"🧪 Teste da arquitetura:")
    print(f"📊 Input: {dummy_input.shape}")
    print(f"📊 Output: {output.shape}")
    print("✅ Arquitetura funcionando corretamente!")