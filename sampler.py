"""SAMPLING ONLY."""

import torch

from dpm_solver_v3 import NoiseScheduleVP, model_wrapper, DPM_Solver_v3
from free_lunch_utils import register_free_upblock2d, register_free_crossattn_upblock2d


class DPMSolverv3Sampler:
    def __init__(self, stats_dir, pipe, steps, guidance_scale, **kwargs):
        super().__init__()
        self.model = pipe
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(pipe.device)
        DTYPE = torch.float32  # torch.float16 works as well, but pictures seem to be a bit worse
        device = "cuda" 
        noise_scheduler = pipe.scheduler
        alpha_schedule = noise_scheduler.alphas_cumprod.to(device=device, dtype=DTYPE)
        self.alphas_cumprod = alpha_schedule #to_torch(model.alphas_cumprod)
        self.device = device
        self.guidance_scale = guidance_scale

        self.ns = NoiseScheduleVP("discrete", alphas_cumprod=self.alphas_cumprod)

        assert stats_dir is not None, f"No statistics file found in {stats_dir}."
        print("Use statistics", stats_dir)
        self.dpm_solver_v3 = DPM_Solver_v3(
            statistics_dir=stats_dir,
            noise_schedule=self.ns,
            steps=steps,
            t_start=None,
            t_end=None,
            skip_type="customed_time_karras",
            degenerated=False,
            device=self.device,
        )
        self.steps = steps

    @torch.no_grad()
    def apply_free_unet(self):
        register_free_upblock2d(self.model, b1=1.1, b2=1.1, s1=0.9, s2=0.2)
        register_free_crossattn_upblock2d(self.model, b1=1.1, b2=1.1, s1=0.9, s2=0.2)

    @torch.no_grad()
    def stop_free_unet(self):
        register_free_upblock2d(self.model, b1=1.0, b2=1.0, s1=1.0, s2=1.0)
        register_free_crossattn_upblock2d(self.model, b1=1.0, b2=1.0, s1=1.0, s2=1.0)
    
    @torch.no_grad()
    def sample(
        self,
        batch_size,
        shape,
        conditioning=None,
        x_T=None,
        unconditional_conditioning=None,
        use_corrector=False,
        half=False,
        start_free_u_step=None,
        # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
        **kwargs,
    ):
        if conditioning is not None:
            cond_in = torch.cat([unconditional_conditioning, conditioning])
            # extra_args = {'cond': conditioning, 'uncond': unconditional_conditioning, 'cond_scale': self.guidance_scale}
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)

        if x_T is None:
            img = torch.randn(size, device=self.device)
        else:
            img = x_T

        if conditioning is None:
            model_fn = model_wrapper(
                lambda x, t, c: self.model.unet(x, t, encoder_hidden_states=c).sample,
                self.ns,
                model_type="noise",
                guidance_type="uncond",
            )
            ORDER = 3
        else:
            model_fn = model_wrapper(
                lambda x, t, c: self.model.unet(x, t, encoder_hidden_states=c).sample,
                self.ns,
                model_type="noise",
                guidance_type="classifier-free",
                condition=conditioning,
                unconditional_condition=unconditional_conditioning,
                guidance_scale=self.guidance_scale,
            )
            if self.steps == 8:
                ORDER = 2
            else:
                ORDER = 1

        x = self.dpm_solver_v3.sample(
            img,
            model_fn,
            order=ORDER,
            p_pseudo=False,
            c_pseudo=True,
            lower_order_final=True,
            use_corrector=use_corrector,
            start_free_u_step=start_free_u_step,
            free_u_apply_callback=self.apply_free_unet if start_free_u_step is not None else None,
            free_u_stop_callback=self.stop_free_unet if start_free_u_step is not None else None,
            half=half,
        )

        return x.to(self.device), None
