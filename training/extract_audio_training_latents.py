import logging
import os
from argparse import ArgumentParser
from datetime import timedelta
from pathlib import Path

import pandas as pd
import tensordict as td
import torch
import torch.distributed as distributed
import torch.nn.functional as F
from open_clip import create_model_from_pretrained
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from mmaudio.data.data_setup import error_avoidance_collate
from mmaudio.data.extraction.wav_dataset import WavTextClipsDataset
from mmaudio.ext.autoencoder import AutoEncoderModule
from mmaudio.ext.mel_converter import get_mel_converter
from mmaudio.utils.dist_utils import local_rank, world_size

log = logging.getLogger()

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# 16k
SAMPLE_RATE = 16_000
NUM_SAMPLES = 16_000 * 8
tod_vae_ckpt = './ext_weights/v1-16.pth'
bigvgan_vocoder_ckpt = './ext_weights/best_netG.pt'
mode = '16k'

# 44k
"""
NOTE: 352800 (8*44100) is not divisible by (STFT hop size * VAE downsampling ratio) which is 1024.
353280 is the next integer divisible by 1024.
"""

# SAMPLE_RATE = 44100
# NUM_SAMPLES = 353280
# tod_vae_ckpt = './ext_weights/v1-44.pth'
# bigvgan_vocoder_ckpt = None
# mode = '44k'


def distributed_setup():
    distributed.init_process_group(backend="nccl", timeout=timedelta(hours=1))
    log.info(f'Initialized: local_rank={local_rank}, world_size={world_size}')
    return local_rank, world_size


@torch.inference_mode()
def main():
    distributed_setup()

    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=Path, default='./training/example_audios/')
    parser.add_argument('--captions_tsv', type=Path, default='./training/example_audio.tsv')
    parser.add_argument('--clips_tsv', type=Path, default='./training/example_output/clips.tsv')
    parser.add_argument('--latent_dir',
                        type=Path,
                        default='./training/example_output/audio-latents')
    parser.add_argument('--output_dir',
                        type=Path,
                        default='./training/example_output/memmap/audio-example')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=8)
    args = parser.parse_args()

    data_dir = args.data_dir
    captions_tsv = args.captions_tsv
    clips_tsv = args.clips_tsv
    latent_dir = args.latent_dir
    output_dir = args.output_dir
    batch_size = args.batch_size
    num_workers = args.num_workers

    # cuda setup
    torch.cuda.set_device(local_rank)
    clip_model = create_model_from_pretrained('hf-hub:apple/DFN5B-CLIP-ViT-H-14-384',
                                              return_transform=False).eval().cuda()

    # a hack to make it output last hidden states
    def new_encode_text(self, text, normalize: bool = False):
        cast_dtype = self.transformer.get_cast_dtype()

        x = self.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.to(cast_dtype)
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = self.ln_final(x)  # [batch_size, n_ctx, transformer.width]
        return F.normalize(x, dim=-1) if normalize else x

    clip_model.encode_text = new_encode_text.__get__(clip_model)

    tod = AutoEncoderModule(vae_ckpt_path=tod_vae_ckpt,
                            vocoder_ckpt_path=bigvgan_vocoder_ckpt,
                            mode=mode).eval().cuda()
    mel_converter = get_mel_converter(mode).eval().cuda()

    dataset = WavTextClipsDataset(data_dir,
                                  captions_tsv=captions_tsv,
                                  clips_tsv=clips_tsv,
                                  sample_rate=SAMPLE_RATE,
                                  num_samples=NUM_SAMPLES,
                                  normalize_audio=True,
                                  reject_silent=True
    )
    sampler = DistributedSampler(dataset, rank=local_rank, shuffle=False)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            sampler=sampler,
                            drop_last=False,
                            collate_fn=error_avoidance_collate)
    latent_dir.mkdir(exist_ok=True, parents=True)

    # extraction
    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        ids = batch['id']
        waveforms = batch['waveform'].cuda()
        tokens = batch['tokens'].cuda()

        text_features = clip_model.encode_text(tokens, normalize=True)
        mel = mel_converter(waveforms)
        dist = tod.encode(mel)

        a_mean = dist.mean.detach().cpu().transpose(1, 2)
        a_std = dist.std.detach().cpu().transpose(1, 2)
        text_features = text_features.detach().cpu()

        ids = [id for id in ids]
        captions = [caption for caption in batch['caption']]

        data = {
            'id': ids,
            'caption': captions,
            'mean': a_mean,
            'std': a_std,
            'text_features': text_features,
        }

        torch.save(data, latent_dir / f'r{local_rank}_{i:05d}.pth')

    distributed.barrier()
    # combine the results
    if local_rank == 0:
        print('Extraction done. Combining the results.')

        list_of_ids_and_labels = []
        output_data = {
            'mean': [],
            'std': [],
            'text_features': [],
        }

        latents = sorted(os.listdir(latent_dir))
        latents = [l for l in latents if l.endswith('.pth')]
        for t in tqdm(latents):
            data = torch.load(latent_dir / t, weights_only=True)
            bs = len(data['id'])

            for bi in range(bs):
                this_id = data['id'][bi]
                this_caption = data['caption'][bi]

                list_of_ids_and_labels.append({'id': this_id, 'caption': this_caption})
                output_data['mean'].append(data['mean'][bi])
                output_data['std'].append(data['std'][bi])
                output_data['text_features'].append(data['text_features'][bi])

        output_df = pd.DataFrame(list_of_ids_and_labels)
        output_dir.mkdir(exist_ok=True, parents=True)
        output_name = output_dir.stem
        output_df.to_csv(output_dir.parent / f'{output_name}.tsv', sep='\t', index=False)

        print(f'Output: {len(output_df)}')

        output_data = {k: torch.stack(v) for k, v in output_data.items()}
        td.TensorDict(output_data).memmap_(output_dir)


if __name__ == '__main__':
    main()
    distributed.destroy_process_group()
