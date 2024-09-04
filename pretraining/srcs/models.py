import sys
import copy
from torch import Tensor
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from transformers.models.bert.modeling_bert import *
sys.path.append(os.getcwd())
from pretraining.srcs.functions import end_pad_to_divisible, repeat_interleave, trim_pad

import warnings

warnings.filterwarnings("ignore")


class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        a, _ = x  # the output of the GRU is the tuple, (hidden_states, final_hidden_state)
        return self.func(a)


# kombo_config.json
class TokenFusingModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.vocab_size = config.vocab_size
        self.d_model = config.d_model

        self.pad_token_id = config.pad_token_id
        self.unk_token_id = config.unk_token_id
        self.bos_token_id = config.bos_token_id
        self.eos_token_id = config.eos_token_id
        self.cls_token_id = config.cls_token_id
        self.sep_token_id = config.sep_token_id
        self.mask_token_id = config.mask_token_id

        self.type_scores = None

        if ('var' in config.tok_type) or ('distinct' in config.tok_type):
            self.space_symbol_id = config.space_symbol_id
            self.empty_jamo_id = config.empty_jamo_id
            # print(f"space_symbol_id: {self.space_symbol_id}")
            # print(f"empty_jamo_id: {self.empty_jamo_id}")

            if config.tok_type == 'stroke_var':
                self.cho_len = 4
                self.joong_len = 1
                self.jong_len = 4
                self.max_jamo_len = max(self.cho_len, self.joong_len, self.jong_len)
                self.char_len = self.cho_len + self.joong_len + self.jong_len
            elif config.tok_type == 'cji_var':
                self.cho_len = 1
                self.joong_len = 5
                self.jong_len = 1
                self.max_jamo_len = max(self.cho_len, self.joong_len, self.jong_len)
                self.char_len = self.cho_len + self.joong_len + self.jong_len
            elif config.tok_type == 'bts_var':
                self.cho_len = 4
                self.joong_len = 5
                self.jong_len = 4
                self.max_jamo_len = max(self.cho_len, self.joong_len, self.jong_len)
                self.char_len = self.cho_len + self.joong_len + self.jong_len
            elif config.tok_type == 'jamo_distinct':
                self.cho_len = 1
                self.joong_len = 1
                self.jong_len = 1
                self.max_jamo_len = max(self.cho_len, self.joong_len, self.jong_len)
                self.char_len = self.cho_len + self.joong_len + self.jong_len
            else:
                raise NotImplementedError

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)

        if self.config.only_contextualization:
            jamo_encode_layer = nn.TransformerEncoderLayer(
                d_model=config.d_model,
                nhead=config.num_attention_heads,
                dim_feedforward=config.intermediate_size,
                activation="gelu",
                batch_first=True,
                norm_first=False,
            )
            self.jamo_gru = nn.Sequential(
                nn.TransformerEncoder(jamo_encode_layer, num_layers=config.jamo_trans_layer),
                nn.GRU(
                    input_size=config.d_model,
                    hidden_size=config.d_model,
                    num_layers=1,
                    batch_first=True,
                ),
            )

        self.jamo_fusion = config.jamo_fusion
        self.jamo_residual = config.jamo_residual
        if self.jamo_fusion == 'conv1':
            """
            without contextualization
            """
            self.jamo_conv = nn.Sequential(
                Rearrange('b l n d -> b d l n'),
                nn.Conv2d(
                    in_channels=config.d_model,
                    out_channels=config.d_model,
                    kernel_size=(2, 1),
                    stride=1,
                    padding='same',
                    padding_mode='zeros',
                    groups=config.d_model
                ),
                nn.AvgPool2d(
                    kernel_size=(2, 1),
                    stride=1,
                    count_include_pad=True,
                ),
                Rearrange('b d l n -> b l n d')
            )
        elif self.jamo_fusion == 'trans_gru_conv':
            """
            Kernel size of (2 x 2)
            """
            jamo_encode_layer = nn.TransformerEncoderLayer(
                d_model=config.d_model,
                nhead=config.num_attention_heads,
                dim_feedforward=config.intermediate_size,
                activation="gelu",
                batch_first=True,
                norm_first=False,
            )
            self.jamo_gru = nn.Sequential(
                nn.TransformerEncoder(jamo_encode_layer, num_layers=config.jamo_trans_layer),
                nn.GRU(
                    input_size=config.d_model,
                    hidden_size=config.d_model,
                    num_layers=1,
                    batch_first=True,
                ),
            )
            self.jamo_conv = nn.Sequential(
                Rearrange('b l n d -> b d l n'),
                nn.Conv2d(
                    in_channels=config.d_model,
                    out_channels=config.d_model,
                    kernel_size=(2, 2),
                    stride=1,
                    padding='same',
                    padding_mode='zeros',
                    groups=config.d_model
                ),
                nn.AvgPool2d(
                    kernel_size=(2, 1),
                    stride=1,
                    count_include_pad=True,
                ),
                Rearrange('b d l n -> b l n d')
            )
        elif self.jamo_fusion == 'trans_gru_conv1':
            """
            (2x1) kernel
            """
            jamo_encode_layer = nn.TransformerEncoderLayer(
                d_model=config.d_model,
                nhead=config.num_attention_heads,
                dim_feedforward=config.intermediate_size,
                activation="gelu",
                batch_first=True,
                norm_first=False,
            )
            self.jamo_gru = nn.Sequential(
                nn.TransformerEncoder(jamo_encode_layer, num_layers=config.jamo_trans_layer),
                nn.GRU(
                    input_size=config.d_model,
                    hidden_size=config.d_model,
                    num_layers=1,
                    batch_first=True,
                ),
            )
            self.jamo_conv = nn.Sequential(
                Rearrange('b l n d -> b d l n'),
                nn.Conv2d(
                    in_channels=config.d_model,
                    out_channels=config.d_model,
                    kernel_size=(2, 1),
                    stride=1,
                    padding='same',
                    padding_mode='zeros',
                    groups=config.d_model
                ),
                nn.AvgPool2d(
                    kernel_size=(2, 1),
                    stride=1,
                    count_include_pad=True,
                ),
                Rearrange('b d l n -> b l n d')
            )
        elif self.jamo_fusion == 'trans_gru_conv2':
            """
            (2x1)+(2x2) kernel
            """
            jamo_encode_layer = nn.TransformerEncoderLayer(
                d_model=config.d_model,
                nhead=config.num_attention_heads,
                dim_feedforward=config.intermediate_size,
                activation="gelu",
                batch_first=True,
                norm_first=False,
            )
            self.jamo_gru = nn.Sequential(
                nn.TransformerEncoder(jamo_encode_layer, num_layers=config.jamo_trans_layer),
                nn.GRU(
                    input_size=config.d_model,
                    hidden_size=config.d_model,
                    num_layers=1,
                    batch_first=True,
                ),
            )
            self.jamo_conv = nn.ModuleList([
                Rearrange('b l n d -> b d l n'),
                nn.Conv2d(
                    in_channels=config.d_model,
                    out_channels=config.d_model,
                    kernel_size=(2, 1),
                    stride=1,
                    padding='same',
                    padding_mode='zeros',
                    groups=config.d_model
                ),
                nn.Conv2d(
                    in_channels=config.d_model,
                    out_channels=config.d_model,
                    kernel_size=(2, 2),
                    stride=1,
                    padding='same',
                    padding_mode='zeros',
                    groups=config.d_model
                ),
                nn.AvgPool2d(
                    kernel_size=(2, 1),
                    stride=1,
                    count_include_pad=True,
                ),
                Rearrange('b d l n -> b l n d')
            ])
        elif self.jamo_fusion == 'trans_gru_conv3':
            """
            (2x3) kernel
            """
            jamo_encode_layer = nn.TransformerEncoderLayer(
                d_model=config.d_model,
                nhead=config.num_attention_heads,
                dim_feedforward=config.intermediate_size,
                activation="gelu",
                batch_first=True,
                norm_first=False,
            )
            self.jamo_gru = nn.Sequential(
                nn.TransformerEncoder(jamo_encode_layer, num_layers=config.jamo_trans_layer),
                nn.GRU(
                    input_size=config.d_model,
                    hidden_size=config.d_model,
                    num_layers=1,
                    batch_first=True,
                ),
            )
            self.jamo_conv = nn.Sequential(
                Rearrange('b l n d -> b d l n'),
                nn.Conv2d(
                    in_channels=config.d_model,
                    out_channels=config.d_model,
                    kernel_size=(2, 3),
                    stride=1,
                    padding='same',
                    padding_mode='zeros',
                    groups=config.d_model
                ),
                nn.AvgPool2d(
                    kernel_size=(2, 1),
                    stride=1,
                    count_include_pad=True,
                ),
                Rearrange('b d l n -> b l n d')
            )
        elif self.jamo_fusion == 'trans_gru_sum':
            """
            Sum
            """
            jamo_encode_layer = nn.TransformerEncoderLayer(
                d_model=config.d_model,
                nhead=config.num_attention_heads,
                dim_feedforward=config.intermediate_size,
                activation="gelu",
                batch_first=True,
                norm_first=False,
            )
            self.jamo_trans = nn.Sequential(
                nn.TransformerEncoder(jamo_encode_layer, num_layers=config.jamo_trans_layer),
                nn.GRU(
                    input_size=config.d_model,
                    hidden_size=config.d_model,
                    num_layers=1,
                    batch_first=True,
                ),
            )
        elif self.jamo_fusion == 'trans_gru_avgpool':
            """
            Average Pooling
            """
            jamo_encode_layer = nn.TransformerEncoderLayer(
                d_model=config.d_model,
                nhead=config.num_attention_heads,
                dim_feedforward=config.intermediate_size,
                activation="gelu",
                batch_first=True,
                norm_first=False,
            )
            self.jamo_trans = nn.Sequential(
                nn.TransformerEncoder(jamo_encode_layer, num_layers=config.jamo_trans_layer),
                nn.GRU(
                    input_size=config.d_model,
                    hidden_size=config.d_model,
                    num_layers=1,
                    batch_first=True,
                ),
            )
            self.jamo_pool = nn.Sequential(
                Rearrange('b l n d -> b d l n'),
                nn.AvgPool2d(
                    kernel_size=(3, 1),
                    stride=1,
                    count_include_pad=True,
                ),
                Rearrange('b d l n -> b l n d')
            )
        elif self.jamo_fusion == 'trans_gru_maxpool':
            """
            Max Pooling
            """
            jamo_encode_layer = nn.TransformerEncoderLayer(
                d_model=config.d_model,
                nhead=config.num_attention_heads,
                dim_feedforward=config.intermediate_size,
                activation="gelu",
                batch_first=True,
                norm_first=False,
            )
            self.jamo_trans = nn.Sequential(
                nn.TransformerEncoder(jamo_encode_layer, num_layers=config.jamo_trans_layer),
                nn.GRU(
                    input_size=config.d_model,
                    hidden_size=config.d_model,
                    num_layers=1,
                    batch_first=True,
                ),
            )
            self.jamo_pool = nn.Sequential(
                Rearrange('b l n d -> b d l n'),
                nn.MaxPool2d(
                    kernel_size=(3, 1),
                    stride=1,
                ),
                Rearrange('b d l n -> b l n d')
            )
        elif self.jamo_fusion == 'trans_gru_linear':
            """
            Linear & Sum
            """
            jamo_encode_layer = nn.TransformerEncoderLayer(
                d_model=config.d_model,
                nhead=config.num_attention_heads,
                dim_feedforward=config.intermediate_size,
                activation="gelu",
                batch_first=True,
                norm_first=False,
            )
            self.jamo_trans = nn.Sequential(
                nn.TransformerEncoder(jamo_encode_layer, num_layers=config.jamo_trans_layer),
                nn.GRU(
                    input_size=config.d_model,
                    hidden_size=config.d_model,
                    num_layers=1,
                    batch_first=True,
                )
            )
            self.jamo_linear = nn.Linear(config.d_model, config.d_model, bias=False)
        elif self.jamo_fusion == 'trans_linear_pool':
            """
            For Hourglass Transformer
            """
            jamo_encode_layer = nn.TransformerEncoderLayer(
                d_model=config.d_model,
                nhead=config.num_attention_heads,
                dim_feedforward=config.intermediate_size,
                activation="gelu",
                batch_first=True,
                norm_first=False,
            )
            self.jamo_trans = nn.Sequential(
                nn.TransformerEncoder(jamo_encode_layer, num_layers=config.jamo_trans_layer),
            )
            self.linear_pool = nn.Sequential(
                Rearrange('b n k d -> b n (k d)', k=self.char_len),
                nn.Linear(config.d_model * self.char_len, config.d_model),
            )
        elif self.jamo_fusion == 'trans_attention_pool':
            """
            For Funnel Transformer
            """
            jamo_encode_layer = nn.TransformerEncoderLayer(
                d_model=config.d_model,
                nhead=config.num_attention_heads,
                dim_feedforward=config.intermediate_size,
                activation="gelu",
                batch_first=True,
                norm_first=False,
            )
            self.jamo_trans = nn.Sequential(
                nn.TransformerEncoder(jamo_encode_layer, num_layers=config.jamo_trans_layer),
            )
            self.avg_pool = nn.Sequential(
                Rearrange('b n k d -> b n (k d)', k=self.char_len),
                nn.AvgPool1d(kernel_size=self.char_len,
                             stride=self.char_len,
                             count_include_pad=True
                             ),
            )
            s_attn = CustomTransformerEncoderLayer(
                d_model=config.d_model,
                nhead=config.num_attention_heads,
                dim_feedforward=config.intermediate_size,
                activation="gelu",
                batch_first=True,
                norm_first=False,
            )
            self.jamo_s_attn = CustomTransformerEncoder(s_attn, num_layers=1)

        if self.jamo_residual:
            self.jamo_lnorm = nn.LayerNorm(config.d_model)

        """
        Upsampling
        * 'upsampling_residual' is the context_inputs of each token units.
        * 'upsampling_repeat_num' is the number of repeats for upsampling using repeat_interleave of each token units.
           i.e., It needs when the model make the small size of the context_inputs to large size of the context_inputs.
                 It's used with the option 'repeat_cascade' in 'upsampling' argument.
        """
        if self.config.upsampling_residual:
            self.upsampling_residual = {'bts': torch.tensor([]),
                                        'jamo': torch.tensor([]),
                                        'char': torch.tensor([]),
                                        'subword': torch.tensor([]),
                                        }
        self.upsampling_repeat_num = {'bts': torch.tensor([]),
                                      'jamo': torch.tensor([]),
                                      'char': torch.tensor([]),
                                      'subword': torch.tensor([])
                                      }

        """
        Initialize the weights and biases using in post_init() method.
        """
        self.weight = self.token_embedding.weight

        if self.config.only_contextualization:
            self.weight_jg_1 = self.jamo_gru[1].weight_hh_l0
            self.weight_jg_2 = self.jamo_gru[1].weight_ih_l0
            self.weight_jg_3 = self.jamo_gru[1].bias_hh_l0
            self.weight_jg_4 = self.jamo_gru[1].bias_ih_l0

        if self.jamo_fusion == 'conv1':
            self.weight_jc_1 = self.jamo_conv[1].weight
            self.weight_jc_2 = self.jamo_conv[1].bias
        elif self.jamo_fusion == 'trans_linear_pool':
            self.weight_jl_1 = self.linear_pool[1].weight
            self.weight_jl_2 = self.linear_pool[1].bias
        elif self.jamo_fusion == 'trans_attention_pool':
            pass
        elif self.jamo_fusion in ['trans_gru_conv', 'trans_gru_conv1', 'trans_gru_conv3']:
            self.weight_jg_1 = self.jamo_gru[1].weight_hh_l0
            self.weight_jg_2 = self.jamo_gru[1].weight_ih_l0
            self.weight_jg_3 = self.jamo_gru[1].bias_hh_l0
            self.weight_jg_4 = self.jamo_gru[1].bias_ih_l0
            self.weight_jc_1 = self.jamo_conv[1].weight
            self.weight_jc_2 = self.jamo_conv[1].bias
        elif self.jamo_fusion == 'trans_gru_conv2':
            self.weight_jg_1 = self.jamo_gru[1].weight_hh_l0
            self.weight_jg_2 = self.jamo_gru[1].weight_ih_l0
            self.weight_jg_3 = self.jamo_gru[1].bias_hh_l0
            self.weight_jg_4 = self.jamo_gru[1].bias_ih_l0
            self.weight_jc_1 = self.jamo_conv[1].weight
            self.weight_jc_2 = self.jamo_conv[1].bias
            self.weight_jc_3 = self.jamo_conv[2].weight
            self.weight_jc_4 = self.jamo_conv[2].bias
        elif self.jamo_fusion in ['trans_gru_sum', 'trans_gru_linear', 'trans_gru_avgpool', 'trans_gru_maxpool']:
            self.weight_jg_1 = self.jamo_trans[1].weight_hh_l0
            self.weight_jg_2 = self.jamo_trans[1].weight_ih_l0
            self.weight_jg_3 = self.jamo_trans[1].bias_hh_l0
            self.weight_jg_4 = self.jamo_trans[1].bias_ih_l0
            if self.jamo_fusion == 'trans_gru_linear':
                self.weight_jl_1 = self.jamo_linear.weight


    def forward(self, input_ids: torch.Tensor):
        f"""
        [CLS] token is separated in advance, so we don't downsample the [CLS] token. 
        [MASK] is the character-level token (not span).
        """
        inputs = self.token_embedding(input_ids)
        batch_size = inputs.shape[0]
        # print("")
        # print("########## Initial input ##########")
        # print(f"input_ids(B, 1+N_init): {input_ids.shape}")
        # print(f"inputs(B, 1+N_init, D): {inputs.shape}")

        cls_input_ids = input_ids[:, 0: 1]
        context_input_ids = input_ids[:, 1:]

        cls_input = inputs[:, 0: 1, :]
        context_inputs = inputs[:, 1:, :]
        # print(f"context_input_ids(B, N_init): {context_input_ids.shape}")
        # print(f"context_inputs(B, N_init, D): {context_inputs.shape}")

        context_input_ids, context_inputs = trim_pad(context_input_ids, context_inputs, pad_value=self.pad_token_id)
        # print(f"trimmed context_input_ids(B, N_init): {context_input_ids.shape}")
        # print(f"trimmed context_inputs(B, N_init, D): {context_inputs.shape}")

        if self.config.only_contextualization:
            context_inputs = self.jamo_gru(context_inputs)[0]

        if self.jamo_fusion:
            """
            In this step, we apply the 'jamo_fusion' layer which fuse the Jamo-level representation into Syllable-level 
            representation. The segment boundary of the Character-level representation is the end_char token which comes 
            third after two end_jamo tokens. Exceptionally, the [SEP] token and the space_symbol are considered as a 
            single independent character, so these tokens should be duplicated to match a total of three like any other 
            normal Hanguel Jamo tokens which consists of two end_jamo token and a end_char token. In order to map into 
            two layered sequences considering the position of the chosung, joongsung, and jongsung, at first, we divide 
            the sequence with every three tokens and add a [PAD] token to each segment to ensure four tokens as a segment. 
            Then we consider each segment as a syllable. Finally we reshape the sequence to have 2 layers as below.
    
            e.g.,
            From text
            감 사 합 니 다.
            
            To Jamo-level tokens
            ['ㄱ', 'ㅏ', 'ㅁ', 'ㅅ', 'ㅏ', '▃', 'ㅎ', 'ㅏ', 'ㅂ', 'ㄴ', 'ㅣ', '▃', 'ㄷ', 'ㅏ', '▃', '.', '▄', '▄', '[SEP]']
            
            To Intermediate-level tokens
            가   사   하   니   다   .
            ㅁ   🆇   ㅂ   🆇   🆇   🆇
            
            To Syllable-level tokens
            ['감', '사', '합', '니', '다', '.', '[SEP]']
            """
            # print("")
            # print("########## Jamo fusion ##########")
            d_jamo = self.d_model

            if self.config.upsampling_residual:
                self.upsampling_residual['jamo'] = context_inputs

            if self.config.ignore_structure:
                pass
            else:
                sep_token_idx = (context_input_ids == self.sep_token_id)
                repeat_num = torch.ones_like(context_input_ids).to(self.device)

                repeat_num[sep_token_idx] = self.char_len
                # print(f"repeat_num: \n{repeat_num}")

                # Calculate the context_input_ids too.
                context_input_ids = repeat_interleave(context_input_ids, repeats=repeat_num.detach().cpu(), dim=1)
                context_inputs = repeat_interleave(context_inputs, repeats=repeat_num.detach().cpu(), dim=1)
                # print(f"(After repeat)context_input_ids(B, N_jamo + two SEP pad * char_len): {context_input_ids.shape}")
                # print(f"After repeat)context_inputs(B, N_jamo + two SEP pad, D_jamo): {context_inputs.shape}")

            try:
                assert context_input_ids.shape[1] % self.char_len == 0
                assert context_input_ids.shape[1] == context_inputs.shape[1]
            except AssertionError:
                # print("[WARNING] seq_len error!!!!")
                before_seq_len = context_input_ids.shape[1]
                context_input_ids = end_pad_to_divisible(context_input_ids, [self.char_len], pad_value=0)
                context_inputs = end_pad_to_divisible(context_inputs, [self.char_len], pad_value=0)
                after_seq_len = context_input_ids.shape[1]
                pad_len = abs(after_seq_len - before_seq_len)
                if self.config.upsampling_residual:
                    self.upsampling_residual['jamo'] = torch.concat([self.upsampling_residual['jamo'],
                                                                     torch.zeros(batch_size, pad_len, d_jamo).to(self.device)],
                                                                    dim=1)
            # print(f"context_input_ids(B, N_jamo): {context_input_ids.shape}")
            # print(f"context_inputs(B, N_jamo, D_jamo): {context_inputs.shape}")

            # select the first token per three tokens as a representative token of character
            context_input_ids = context_input_ids[:, ::self.char_len]
            # print(f"(After Pooling) context_input_ids(B, N_char): {context_input_ids.shape}")

            if self.jamo_fusion in ['trans_gru_conv', 'trans_gru_conv1', 'trans_gru_conv2', 'trans_gru_conv3']:
                context_inputs = self.jamo_gru(context_inputs)[0]
            elif self.jamo_fusion in ['trans_gru_sum', 'trans_gru_linear', 'trans_gru_maxpool', 'trans_gru_avgpool']:
                context_inputs = self.jamo_trans(context_inputs)[0]
            elif self.jamo_fusion in ['trans_linear_pool', 'trans_attention_pool']:
                context_inputs = self.jamo_trans(context_inputs)
            else:
                pass

            context_inputs = context_inputs.reshape(batch_size, -1, self.char_len, d_jamo)
            # print(f"context_inputs(B, N_char, 3, D_jamo): {context_inputs.shape}")

            jamo_residual = context_inputs
            # print(f"jamo_residual(B, N_char, 3, D_jamo): {jamo_residual.shape}")

            if self.jamo_fusion == 'conv1':
                cho_joong_inputs = torch.sum(context_inputs[:, :, :- self.jong_len], dim=2, keepdim=True)
                jong_inputs = torch.sum(context_inputs[:, :, -self.jong_len:], dim=2, keepdim=True)
                context_inputs = torch.concat([cho_joong_inputs, jong_inputs], dim=2)
                context_inputs = rearrange(context_inputs, 'b n l d -> b l n d')
                context_inputs = self.jamo_conv(context_inputs).squeeze()
            elif self.jamo_fusion == 'trans_gru_sum':
                context_inputs = torch.sum(context_inputs, dim=2)
            elif self.jamo_fusion == 'trans_gru_linear':
                context_inputs = self.jamo_linear(context_inputs)
                context_inputs = torch.sum(context_inputs, dim=2)
            elif self.jamo_fusion in ['trans_gru_maxpool', 'trans_gru_avgpool']:
                context_inputs = rearrange(context_inputs, 'b n l d -> b l n d')
                context_inputs = self.jamo_pool(context_inputs).squeeze()
            elif self.jamo_fusion == 'trans_linear_pool':
                context_inputs = self.linear_pool(context_inputs)
            elif self.jamo_fusion == 'trans_attention_pool':
                s_attn_res = rearrange(context_inputs, 'b n l d -> b (n l) d')
                context_inputs = self.avg_pool(context_inputs)
                context_inputs = self.jamo_s_attn(context_inputs, s_attn_res, s_attn_res)
            else:
                """
                Chosung + Joongsung
                """
                cho_joong_inputs = torch.sum(context_inputs[:, :, :- self.jong_len], dim=2, keepdim=True)
                jong_inputs = torch.sum(context_inputs[:, :, -self.jong_len:], dim=2, keepdim=True)
                context_inputs = torch.concat([cho_joong_inputs, jong_inputs], dim=2)
                # context_inputs = context_inputs[:, :, 1:]
                # print(f"context_inputs(B, N_char, 2, D_jamo): {context_inputs.shape}")
                """
                Merge Jongsung (Rearrange & Conv)
                """
                context_inputs = rearrange(context_inputs, 'b n l d -> b l n d')
                # print(f"context_inputs(B, 2, N_char, D_jamo): {context_inputs.shape}")
                if self.jamo_fusion == 'trans_gru_conv2':
                    context_inputs = self.jamo_conv[0](context_inputs)  # Rearrange
                    context_inputs1 = self.jamo_conv[1](context_inputs)  # (2x1) kernel
                    context_inputs2 = self.jamo_conv[2](context_inputs)  # (2x2) kernel
                    context_inputs = context_inputs1 + context_inputs2
                    context_inputs = self.jamo_conv[3](context_inputs)  # Avg_Pooling
                    context_inputs = self.jamo_conv[4](context_inputs).squeeze()  # Rearrange
                else:
                    context_inputs = self.jamo_conv(context_inputs).squeeze()

            if context_inputs.ndim == 2:  # it runs when the batch size is 1.
                context_inputs = context_inputs.unsqueeze(0)
            # print(f"context_inputs(B, N_char, D_char): {context_inputs.shape}")

            if self.jamo_residual:
                context_inputs = torch.sum(torch.concat([jamo_residual, context_inputs.unsqueeze(2)], dim=2), dim=2)
                context_inputs = self.jamo_lnorm(context_inputs)
            # print(f"context_inputs(B, N_char, D_jamo): {context_inputs.shape}")

            char_repeat_num = torch.ones_like(context_input_ids) * self.char_len
            if self.config.ignore_structure:
                pass
            else:
                sep_repeat_num = (context_input_ids == self.sep_token_id).to(torch.long) * -(self.char_len - 1)
                char_repeat_num += sep_repeat_num
            # print(f"char_repeat_num[0]: {char_repeat_num[0]}")

            self.upsampling_repeat_num["char"] = char_repeat_num
            if self.config.upsampling_residual:
                self.upsampling_residual['char'] = context_inputs
            # print(f"After jamo_fusion - context_inputs(B, N_char, D_char): {context_inputs.shape}")
            # print(f"After jamo_fusion - context_input_ids(B, N_char): {context_input_ids.shape}")

        outputs = torch.concat([cls_input, context_inputs], dim=1)
        # print(f"outputs(B, 1+N_char(downsampled), D_char): {outputs.shape}")

        # print(f"cls_input_ids.shape: {cls_input_ids.shape}")
        # print(f"context_input_ids.shape: {context_input_ids.shape}")
        output_ids = torch.concat([cls_input_ids, context_input_ids], dim=1)
        # print(f"output_ids(B, 1+N_char(downsampled)): {output_ids.shape}")
        # print("\n")

        # Create attention mask too!
        attention_mask = ((output_ids != self.pad_token_id) * 1).to(self.device)
        # print(f"attention_mask(B, 1+N_char(downsampled)): {attention_mask.shape}")
        return outputs, attention_mask


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class CustomTransformerEncoder(nn.Module):
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(CustomTransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        output = q
        k = k
        v = v
        for mod in self.layers:
            output = mod(output, k, v, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
            break       # We only use a iteration of encoder layer.

        if self.norm is not None:
            output = self.norm(output)

        return output


class CustomTransformerEncoderLayer(nn.Module):
    """
    This is for Funnel Transformer downsampling (shortening).
    We separate the inputs of self-attention in the encoder layer as query, key, and value.
    """
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=F.relu,
                 layer_norm_eps=1e-5, batch_first=False, norm_first=False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(CustomTransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                            **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(CustomTransformerEncoderLayer, self).__setstate__(state)

    def forward(self, q: Tensor, k: Tensor, v: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """

        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        q = q
        k = k
        v = v
        if self.norm_first:
            x = q + self._sa_block(self.norm1(q), self.norm1(k), self.norm1(v), src_mask, src_key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(q + self._sa_block(q, k, v, src_mask, src_key_padding_mask))
            x = self.norm2(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(self, q: Tensor, k: Tensor, v: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.self_attn(q, k, v,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)



class CustomBertModel(BertPreTrainedModel):
    """
    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in [Attention is
    all you need](https://arxiv.org/abs/1706.03762) by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as a decoder the model needs to be initialized with the `is_decoder` argument of the configuration set
    to `True`. To be used in a Seq2Seq model, the model needs to initialized with both `is_decoder` argument and
    `add_cross_attention` set to `True`; an `encoder_hidden_states` is then expected as an input to the forward pass.
    """

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config
        if config.embedding_type == 'TokenFusingModule':
            self.embeddings = TokenFusingModule(config)
        else:
            raise NotImplementedError

        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

        elif isinstance(module, TokenFusingModule):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if ('var' in self.config.tok_type) or ('distinct' in self.config.tok_type) or (
                    'position' in self.config.tok_type):
                module.weight.data[self.config.empty_jamo_id].zero_()

            if hasattr(module, "lm_head") and not self.config.tie_word_embeddings:
                module.lm_head.weight.data.normal_(mean=0.0, std=self.config.initializer_range)

            if module.config.only_contextualization:
                module.weight_jg_1.data.normal_(mean=0.0, std=self.config.initializer_range)
                module.weight_jg_2.data.normal_(mean=0.0, std=self.config.initializer_range)
                module.weight_jg_3.data.zero_()
                module.weight_jg_4.data.zero_()

            if module.jamo_fusion == 'conv1':
                module.weight_jc_1.data.normal_(mean=0.0, std=self.config.initializer_range)
                module.weight_jc_2.data.zero_()
            elif module.jamo_fusion == 'trans_linear_pool':
                module.weight_jl_1.data.normal_(mean=0.0, std=self.config.initializer_range)
                module.weight_jl_2.data.zero_()
            elif module.jamo_fusion == 'trans_attention_pool':
                pass
            elif module.jamo_fusion in ['trans_gru_conv', 'trans_gru_conv1', 'trans_gru_conv3']:
                module.weight_jg_1.data.normal_(mean=0.0, std=self.config.initializer_range)
                module.weight_jg_2.data.normal_(mean=0.0, std=self.config.initializer_range)
                module.weight_jg_3.data.zero_()
                module.weight_jg_4.data.zero_()
                module.weight_jc_1.data.normal_(mean=0.0, std=self.config.initializer_range)
                module.weight_jc_2.data.zero_()
            elif module.jamo_fusion in ['trans_gru_sum', 'trans_gru_linear', 'trans_gru_avgpool', 'trans_gru_maxpool']:
                module.weight_jg_1.data.normal_(mean=0.0, std=self.config.initializer_range)
                module.weight_jg_2.data.normal_(mean=0.0, std=self.config.initializer_range)
                module.weight_jg_3.data.zero_()
                module.weight_jg_4.data.zero_()
                if module.jamo_fusion == 'trans_gru_linear':
                    module.weight_jl_1.data.normal_(mean=0.0, std=self.config.initializer_range)
            elif module.jamo_fusion == 'trans_gru_conv2':
                module.weight_jg_1.data.normal_(mean=0.0, std=self.config.initializer_range)
                module.weight_jg_2.data.normal_(mean=0.0, std=self.config.initializer_range)
                module.weight_jg_3.data.zero_()
                module.weight_jg_4.data.zero_()
                module.weight_jc_1.data.normal_(mean=0.0, std=self.config.initializer_range)
                module.weight_jc_2.data.zero_()
                module.weight_jc_3.data.normal_(mean=0.0, std=self.config.initializer_range)
                module.weight_jc_4.data.zero_()

    def get_input_embeddings(self):
        return self.embeddings.token_embedding

    def set_input_embeddings(self, value):
        self.embeddings.token_embedding = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # @add_code_sample_docstrings(
    #     processor_class=_TOKENIZER_FOR_DOC,
    #     checkpoint=_CHECKPOINT_FOR_DOC,
    #     output_type=BaseModelOutputWithPoolingAndCrossAttentions,
    #     config_class=_CONFIG_FOR_DOC
    # )
    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        r"""
        encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        """

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        # # past_key_values_length
        # past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        embedding_output, attention_mask = self.embeddings(
            input_ids
        )

        if input_ids is not None and inputs_embeds is not None and embedding_output is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif embedding_output is not None:
            input_shape = embedding_output.size()[:2]  # (batch_size, seq_length)
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        # batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # if attention_mask is None:
        #     attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        # if token_type_ids is None:
        #     if hasattr(self.embeddings, "token_type_ids"):
        #         buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
        #         buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
        #         token_type_ids = buffered_token_type_ids_expanded
        #     else:
        #         token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )

class CustomBertForPreTraining(BertPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias", r"cls.predictions.decoder.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.bert = CustomBertModel(config)

        if config.upsampling == 'linear':
            self.linear = nn.Linear(config.hidden_size, config.hidden_size)
        elif config.upsampling == 'repeat_linear':
            self.cascade = nn.ModuleList([
                nn.Linear(config.hidden_size, config.hidden_size)
                for _ in range(len(self.bert.embeddings.upsampling_repeat_num))
            ])
            for modules in self.cascade:
                linear_module = modules
                linear_module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
                linear_module.bias.data.zero_()
        elif config.upsampling == 'gru':
            self.gru = nn.GRU(
                        input_size=config.d_model,
                        hidden_size=config.d_model,
                        num_layers=1,
                        batch_first=True
            )
            self.gru.weight_hh_l0.data.normal_(mean=0.0, std=self.config.initializer_range)
            self.gru.weight_ih_l0.data.normal_(mean=0.0, std=self.config.initializer_range)
            self.gru.bias_hh_l0.data.zero_()
            self.gru.bias_ih_l0.data.zero_()
        elif config.upsampling == 'repeat_gru':
            self.cascade = nn.ModuleList([
                nn.Sequential(
                    nn.GRU(
                        input_size=config.d_model,
                        hidden_size=config.d_model,
                        num_layers=1,
                        batch_first=True,
                    ),
                )
                for _ in range(len(self.bert.embeddings.upsampling_repeat_num))
            ])
            for modules in self.cascade:
                gru_module = modules[0]
                gru_module.weight_hh_l0.data.normal_(mean=0.0, std=self.config.initializer_range)
                gru_module.weight_ih_l0.data.normal_(mean=0.0, std=self.config.initializer_range)
                gru_module.bias_hh_l0.data.zero_()
                gru_module.bias_ih_l0.data.zero_()

        if config.upsampling_residual:
            self.lnorm = nn.ModuleList([
                nn.LayerNorm(config.d_model) for _ in range(len(self.bert.embeddings.upsampling_residual))
            ])

        self.cls = BertPreTrainingHeads(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # @replace_return_docstrings(output_type=BertForPreTrainingOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            next_sentence_label: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BertForPreTrainingOutput]:
        r"""
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
                config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked),
                the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
            next_sentence_label (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
                Labels for computing the next sequence prediction (classification) loss. Input should be a sequence
                pair (see `input_ids` docstring) Indices should be in `[0, 1]`:

                - 0 indicates sequence B is a continuation of sequence A,
                - 1 indicates sequence B is a random sequence.
            kwargs (`Dict[str, any]`, optional, defaults to *{}*):
                Used to hide legacy arguments that have been deprecated.

        Returns:

        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids=input_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output, pooled_output = outputs[:2]  # sequence_output = (batch_size, seq_len, d_model)
        # (CLS) pooled_output = (batch_size, d_model)

        if self.config.upsampling:
            """
            Upsampling
            """
            # print(f"context_input_ids: {input_ids.shape}")

            cls_output = sequence_output[:, 0: 1, :]  # (batch_size, 1, d_model)

            context_output = sequence_output[:, 1:, :]  # (batch_size, seq_len-1, d_model)
            # print(f"context_output(B, N_char, D): {context_output.shape}\n")

            batch_size, context_seq_len, d_model = list(context_output.shape)

            if self.config.upsampling == 'linear':
                for i, type_ in enumerate(list(self.bert.embeddings.upsampling_repeat_num)[::-1]):
                    repeat_num = self.bert.embeddings.upsampling_repeat_num[type_]
                    # print(f"repeat_num: {self z.bert.embeddings.upsampling_repeat_num[type_]}")
                    if repeat_num.__len__() == 0:
                        # print("continue\n")
                        continue
                    context_output = repeat_interleave(context_output, repeats=repeat_num.detach().cpu(), dim=1)

                context_output = self.linear(context_output)
            elif self.config.upsampling == 'repeat_linear':
                for i, type_ in enumerate(list(self.bert.embeddings.upsampling_repeat_num)[::-1]):
                    repeat_num = self.bert.embeddings.upsampling_repeat_num[type_]
                    # print(f"repeat_num: {self z.bert.embeddings.upsampling_repeat_num[type_]}")
                    if repeat_num.__len__() == 0:
                        # print("continue\n")
                        continue

                    # print(f"before_repeat context_output: {context_output.shape}")
                    context_output = repeat_interleave(context_output, repeats=repeat_num.detach().cpu(), dim=1)
                    # print(f"after_repeat context_output: {context_output.shape}")

                    if self.config.upsampling_residual:
                        if type_ == 'jamo':
                            up_type = 'bts'
                        elif type_ == 'char':
                            up_type = 'jamo'
                        elif type_ == 'subword':
                            up_type = 'char'
                        else:
                            raise NotImplementedError
                        # print(f"up_type: {up_type}")
                        # print(f"self.bert.embeddings.upsampling_residual[up_type].shape: {self.bert.embeddings.upsampling_residual[up_type].shape}")
                        context_output = context_output + self.bert.embeddings.upsampling_residual[up_type]
                        context_output = self.lnorm[i](context_output)

                    context_output = self.cascade[i](context_output)
                    # print(f"context_output: {context_output.shape}\n")
            elif self.config.upsampling == 'gru':
                for i, type_ in enumerate(list(self.bert.embeddings.upsampling_repeat_num)[::-1]):
                    repeat_num = self.bert.embeddings.upsampling_repeat_num[type_]
                    # print(f"repeat_num: {self.bert.embeddings.upsampling_repeat_num[type_]}")
                    if repeat_num.__len__() == 0:
                        # print(f"{type_} type: continue\n")
                        continue
                    # print(f"Before_repeat: {context_output.shape}")
                    context_output = repeat_interleave(context_output, repeats=repeat_num.detach().cpu(), dim=1)
                    # print(f"After_repeat: {context_output.shape}\n")
                context_output = self.gru(context_output)[0]
            elif self.config.upsampling == 'repeat_gru':
                for i, type_ in enumerate(list(self.bert.embeddings.upsampling_repeat_num)[::-1]):
                    repeat_num = self.bert.embeddings.upsampling_repeat_num[type_]
                    # print(f"repeat_num: {self z.bert.embeddings.upsampling_repeat_num[type_]}")
                    if repeat_num.__len__() == 0:
                        # print("continue\n")
                        continue

                    # print(f"before_repeat context_output: {context_output.shape}")
                    context_output = repeat_interleave(context_output, repeats=repeat_num.detach().cpu(), dim=1)
                    # print(f"after_repeat context_output: {context_output.shape}")

                    if self.config.upsampling_residual:
                        if type_ == 'jamo':
                            up_type = 'bts'
                        elif type_ == 'char':
                            up_type = 'jamo'
                        elif type_ == 'subword':
                            up_type = 'char'
                        else:
                            raise NotImplementedError
                        # print(f"up_type: {up_type}")
                        # print(f"self.bert.embeddings.upsampling_residual[up_type].shape: {self.bert.embeddings.upsampling_residual[up_type].shape}")
                        context_output = context_output + self.bert.embeddings.upsampling_residual[up_type]
                        context_output = self.lnorm[i](context_output)

                    context_output = self.cascade[i](context_output)[0]
                    # print(f"context_output: {context_output.shape}\n")
            else:
                raise NotImplementedError

            extended_sequence_output = torch.concat([cls_output, context_output], dim=1)  # (batch_size, seq_len, d_model)
            try:
                assert extended_sequence_output.shape[1] == input_ids.shape[1]
            except AssertionError:
                # print(f"extended_sequence_output: {extended_sequence_output.shape}")
                # print(f"input_ids: {input_ids.shape}")
                extended_sequence_output = extended_sequence_output[:, :input_ids.shape[1], :]

        else:
            extended_sequence_output = sequence_output

        prediction_scores, seq_relationship_score = self.cls(extended_sequence_output, pooled_output)

        total_loss = None
        if labels is not None and next_sentence_label is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            total_loss = masked_lm_loss + next_sentence_loss

        if not return_dict:
            output = (prediction_scores, seq_relationship_score) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return BertForPreTrainingOutput(
            loss=total_loss,
            prediction_logits=prediction_scores,
            seq_relationship_logits=seq_relationship_score,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
