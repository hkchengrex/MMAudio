import logging
import os
from pathlib import Path

import hydra
import torch
import torch.distributed as distributed
import torchaudio
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from tqdm import tqdm

from mmaudio.data.data_setup import setup_eval_dataset
from mmaudio.eval_utils import ModelConfig, all_model_cfg, generate
from mmaudio.model.flow_matching import FlowMatching
from mmaudio.model.networks import MMAudio, get_my_mmaudio
from mmaudio.model.utils.features_utils import FeaturesUtils

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

local_rank = int(os.environ['LOCAL_RANK'])
world_size = int(os.environ['WORLD_SIZE'])
log = logging.getLogger()


@torch.inference_mode()
@hydra.main(version_base='1.3.2', config_path='config', config_name='eval_config.yaml')
def main(cfg: DictConfig):
    device = 'cuda'
    torch.cuda.set_device(local_rank)

    if cfg.model not in all_model_cfg:
        raise ValueError(f'Unknown model variant: {cfg.model}')
    model: ModelConfig = all_model_cfg[cfg.model]
    model.download_if_needed()
    seq_cfg = model.seq_cfg

    run_dir = Path(HydraConfig.get().run.dir)
    if cfg.output_name is None:
        output_dir = run_dir / cfg.dataset
    else:
        output_dir = run_dir / f'{cfg.dataset}-{cfg.output_name}'
    output_dir.mkdir(parents=True, exist_ok=True)

    # load a pretrained model
    seq_cfg.duration = cfg.duration_s
    net: MMAudio = get_my_mmaudio(cfg.model).to(device).eval()
    net.load_weights(torch.load(model.model_path, map_location=device, weights_only=True))
    log.info(f'Loaded weights from {model.model_path}')
    net.update_seq_lengths(seq_cfg.latent_seq_len, seq_cfg.clip_seq_len, seq_cfg.sync_seq_len)
    log.info(f'Latent seq len: {seq_cfg.latent_seq_len}')
    log.info(f'Clip seq len: {seq_cfg.clip_seq_len}')
    log.info(f'Sync seq len: {seq_cfg.sync_seq_len}')

    # misc setup
    rng = torch.Generator(device=device)
    rng.manual_seed(cfg.seed)
    fm = FlowMatching(cfg.sampling.min_sigma,
                      inference_mode=cfg.sampling.method,
                      num_steps=cfg.sampling.num_steps)

    feature_utils = FeaturesUtils(tod_vae_ckpt=model.vae_path,
                                  synchformer_ckpt=model.synchformer_ckpt,
                                  enable_conditions=True,
                                  mode=model.mode,
                                  bigvgan_vocoder_ckpt=model.bigvgan_16k_path,
                                  need_vae_encoder=False)
    feature_utils = feature_utils.to(device).eval()

    if cfg.compile:
        net.preprocess_conditions = torch.compile(net.preprocess_conditions)
        net.predict_flow = torch.compile(net.predict_flow)
        feature_utils.compile()

    dataset, loader = setup_eval_dataset(cfg.dataset, cfg)

    with torch.amp.autocast(enabled=cfg.amp, dtype=torch.bfloat16, device_type=device):
        for batch in tqdm(loader):
            audios = generate(batch.get('clip_video', None),
                              batch.get('sync_video', None),
                              batch.get('caption', None),
                              feature_utils=feature_utils,
                              net=net,
                              fm=fm,
                              rng=rng,
                              cfg_strength=cfg.cfg_strength,
                              clip_batch_size_multiplier=64,
                              sync_batch_size_multiplier=64)
            audios = audios.float().cpu()
            names = batch['name']
            for audio, name in zip(audios, names):
                torchaudio.save(output_dir / f'{name}.flac', audio, seq_cfg.sampling_rate)


def distributed_setup():
    distributed.init_process_group(backend="nccl")
    local_rank = distributed.get_rank()
    world_size = distributed.get_world_size()
    log.info(f'Initialized: local_rank={local_rank}, world_size={world_size}')
    return local_rank, world_size


if __name__ == '__main__':
    distributed_setup()

    main()

    # clean-up
    distributed.destroy_process_group()
