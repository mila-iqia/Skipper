import os, sys

sys.path.insert(1, os.getcwd())
import torch, numpy as np
from utils import init_weights, HistogramConverter, cyclical_schedule  # , unique_rows
from modules import ResidualBlock, Block_UNET, TopKMultiheadAttention


class CVAE_MiniGrid_Separate2(torch.nn.Module):
    def __init__(
        self,
        layout_extractor,
        decoder,
        sample_input,
        num_categoricals=8,
        num_categories=8,
        beta=0.0005,
        KL_balance=False,
        maximize_entropy=True,
        alpha_KL_balance=0.8,
        activation=torch.nn.ReLU,
        interval_beta=2500,
        batchnorm=False,
        argmax_latents=True,
        argmax_reconstruction=True,
        **kwargs,
    ):
        super(CVAE_MiniGrid_Separate2, self).__init__(**kwargs)
        self.argmax_latents = bool(argmax_latents)
        self.argmax_reconstruction = bool(argmax_reconstruction)

        self.num_categoricals, self.num_categories = num_categoricals, num_categories
        self.len_code = num_categoricals * num_categories
        self.decoder = decoder
        self.layout_extractor = layout_extractor

        self.KL_balance, self.maximize_entropy = KL_balance, maximize_entropy
        self.beta, self.alpha_KL_balance = beta, alpha_KL_balance
        from minigrid import OBJECT_TO_IDX

        self.object_to_idx = OBJECT_TO_IDX
        self.interval_beta = interval_beta
        self.steps_trained = 0
        self.size_input = sample_input.shape[-2]

        self.encoder_context = Embedder_MiniGrid_BOW(
            dim_embed=32, width=sample_input.shape[-3], height=sample_input.shape[-2], channels_obs=sample_input.shape[-1], ebd_pos=False
        )
        self.encoder_obs = Embedder_MiniGrid_BOW(
            dim_embed=32, width=sample_input.shape[-3], height=sample_input.shape[-2], channels_obs=sample_input.shape[-1], ebd_pos=False
        )

        self.compressor = torch.nn.Sequential(
            ResidualBlock(len_in=32, depth=2, kernel_size=3, stride=1, padding=1, activation=activation),
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),
            ResidualBlock(len_in=64, depth=2, kernel_size=3, stride=1, padding=1, activation=activation),
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0),
            ResidualBlock(len_in=128, depth=2, kernel_size=3, stride=1, padding=1, activation=activation),
            torch.nn.AdaptiveMaxPool2d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(128, self.len_code),
        )

        self.decompressor = torch.nn.Sequential(
            torch.nn.Linear(self.len_code, 128),
            activation(True),
            torch.nn.Unflatten(1, (128, 1, 1)),
            torch.nn.ConvTranspose2d(128, 32, kernel_size=self.size_input, stride=1, padding=0),
        )

        self.fuser = torch.nn.Sequential(
            ResidualBlock(len_in=32 + 32, depth=2, kernel_size=3, stride=1, padding=1, activation=activation),
            torch.nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0),
        )

        tensors_to_mesh = []
        for idx_category in range(self.num_categoricals):
            tensors_to_mesh.append(torch.arange(self.num_categories))
        indices_mesh = torch.meshgrid(*tensors_to_mesh)
        indices_mesh = torch.concatenate([indices.reshape(1, -1) for indices in indices_mesh], 0).permute(1, 0).contiguous()
        self.samples_uniform = torch.nn.functional.one_hot(indices_mesh, num_classes=self.num_categories).float()
        self.minus_log_uniform = float(np.log(self.num_categories))

    def compress_from_obs(self, obs):
        obs_encoded = self.encoder_obs(obs)
        logits_code = self.compressor(obs_encoded).reshape(-1, self.num_categoricals, self.num_categories)
        return logits_code

    # @profile
    def fuse_samples_with_context(self, samples, context):
        size_batch = samples.shape[0]
        assert context.shape[0] == size_batch
        samples_decompressed = self.decompressor(samples.reshape(size_batch, -1))
        logits_mask_agent = self.fuser(torch.cat([samples_decompressed, context], 1))
        return logits_mask_agent

    def to(self, device):
        super().to(device)
        self.encoder_context.to(device)
        self.encoder_obs.to(device)
        self.compressor.to(device)
        self.decompressor.to(device)
        self.fuser.to(device)

    def parameters(self):
        parameters = []
        parameters += list(self.encoder_context.parameters())
        parameters += list(self.encoder_obs.parameters())
        parameters += list(self.compressor.parameters())
        parameters += list(self.decompressor.parameters())
        parameters += list(self.fuser.parameters())
        return parameters

    @torch.no_grad()
    # @profile
    def mask_from_logits(self, logits_mask_agent, noargmax=False):
        size_batch = logits_mask_agent.shape[0]
        logits_mask_agent = logits_mask_agent.reshape(size_batch, -1)
        assert logits_mask_agent.shape[-1] == self.size_input**2
        if self.argmax_reconstruction and not noargmax:
            mask_agent_pred = torch.nn.functional.one_hot(logits_mask_agent.argmax(-1), num_classes=self.size_input**2)
        else:
            mask_agent_pred = torch.distributions.OneHotCategorical(logits=logits_mask_agent).sample()
        mask_agent_pred = mask_agent_pred.reshape(-1, self.size_input, self.size_input)
        return mask_agent_pred

    @torch.no_grad()
    # @profile
    def sample_from_uniform_prior(self, obs_curr, num_samples=None, code2exclude=None):
        if self.samples_uniform.device != obs_curr.device:
            self.samples_uniform = self.samples_uniform.to(obs_curr.device)
        samples = self.samples_uniform
        assert samples.shape[0] == self.num_categories**self.num_categoricals
        if code2exclude is not None:
            assert isinstance(code2exclude, torch.Tensor) and len(code2exclude.shape) == 2
            to_remove = torch.zeros(samples.shape[0], dtype=torch.bool, device=samples.device)
            all_codes = samples.reshape(-1, self.num_categories * self.num_categoricals).bool()
            for idx_row in range(code2exclude.shape[0]):
                to_exclude = code2exclude[idx_row, :].reshape(1, -1).bool()
                coincidence = (to_exclude == all_codes).all(-1)
                to_remove |= coincidence
            samples = samples[~to_remove]
        if num_samples is not None and num_samples < samples.shape[0]:
            indices = torch.randperm(samples.shape[0])[:num_samples]
            samples = samples[indices]
        if obs_curr.shape[0] == 1:
            obs_curr = torch.repeat_interleave(obs_curr, samples.shape[0], 0)
        else:
            assert obs_curr.shape[0] == samples.shape[0]
        samples = samples.float()
        return samples, self.forward(obs_curr, samples=samples, train=False)

    @torch.no_grad()
    # @profile
    def generate_from_obs(self, obs, num_samples=None):
        assert num_samples is not None
        size_batch = obs.shape[0]
        if size_batch > 1:
            assert size_batch == num_samples
        layout, _ = self.layout_extractor(obs)
        layout = layout.float().detach()
        code, mask_agent_pred = self.sample_from_uniform_prior(obs, num_samples=num_samples)
        mask_agent_pred = mask_agent_pred.reshape(num_samples, self.size_input, self.size_input)
        obses_pred = self.decoder(layout, mask_agent_pred)
        return code, obses_pred

    @torch.no_grad()
    # @profile
    def imagine_batch_from_obs(self, obs):
        layout, _ = self.layout_extractor(obs)
        layout = layout.float().detach()
        context = self.encoder_context(obs)
        if True:
            size_batch = obs.shape[0]
            samples = (
                torch.distributions.OneHotCategorical(
                    probs=torch.ones(size_batch, self.num_categoricals, self.num_categories, dtype=torch.float32, device=obs.device)
                )
                .sample()
                .reshape(size_batch, -1)
            )
        else:
            samples = self.encode_from_obs(obs, no_argmax=True)
        logits_mask_agent = self.fuse_samples_with_context(samples.float(), context)
        mask_agent_pred = self.mask_from_logits(logits_mask_agent, noargmax=True)
        obses_pred = self.decoder(layout, mask_agent_pred)
        return obses_pred

    @torch.no_grad()
    # @profile
    def encode_from_obs(self, obs, no_argmax=False):
        logits_samples = self.compress_from_obs(obs)
        if self.argmax_latents and not no_argmax:
            samples = torch.nn.functional.one_hot(logits_samples.argmax(-1), num_classes=self.num_categories)
        else:
            dist = torch.distributions.OneHotCategorical(logits=logits_samples)
            samples = dist.sample()
        return samples

    @torch.no_grad()
    # @profile
    def decode_to_obs(self, samples, obs):
        size_batch = obs.shape[0]
        assert samples.shape[0] == size_batch
        layout, mask_agent = self.layout_extractor(obs)
        layout = layout.float().detach()
        context = self.encoder_context(obs)
        samples = samples.reshape(size_batch, -1)
        logits_mask_agent = self.fuse_samples_with_context(samples, context)
        mask_agent_pred = self.mask_from_logits(logits_mask_agent)
        obses_pred = self.decoder(layout, mask_agent_pred)
        return obses_pred

    # @profile
    def forward(self, obs_curr, obs_targ=None, samples=None, train=False):
        size_batch = obs_curr.shape[0]
        context = self.encoder_context(obs_curr)
        if samples is None:
            assert obs_targ is not None
            logits_samples = self.compress_from_obs(obs_targ)
            if self.argmax_latents:
                argmax_samples = logits_samples.argmax(-1)
                samples = torch.nn.functional.one_hot(argmax_samples, num_classes=self.num_categories)
                probs_samples = logits_samples.softmax(-1)
                samples = probs_samples + (samples - probs_samples).detach()
            else:
                samples = torch.distributions.OneHotCategoricalStraightThrough(logits=logits_samples).rsample()
        samples = samples.reshape(size_batch, -1)
        logits_mask_agent = self.fuse_samples_with_context(samples, context)
        logits_mask_agent = logits_mask_agent.reshape(size_batch, -1)
        if train:
            return logits_samples, logits_mask_agent
        else:
            with torch.no_grad():
                mask_agent_pred = self.mask_from_logits(logits_mask_agent)
            return mask_agent_pred.bool()

    # @profile
    def compute_loss(self, batch_processed, debug=False):
        batch_obs_curr, batch_action, batch_reward, batch_obs_next, batch_done, batch_obs_targ, weights, batch_idxes = batch_processed
        with torch.no_grad():
            obses_context, obses_chosen = batch_obs_targ, batch_obs_curr
        size_batch = obses_chosen.shape[0]
        obses_chosen_train = obses_chosen
        layouts_train, masks_agent_train = self.layout_extractor(obses_chosen_train)
        logits_samples, logits_mask_agent_train = self.forward(obs_curr=obses_context, obs_targ=obses_chosen, train=True)
        logsoftmax_mask_agent = logits_mask_agent_train.log_softmax(-1)
        loss_recon = torch.nn.functional.kl_div(
            input=logsoftmax_mask_agent, target=masks_agent_train.float().reshape(size_batch, -1), log_target=False, reduction="none"
        ).sum(-1)

        loss_align = None

        # maximize the kl-loss
        eps = 1e-7
        probs_samples = logits_samples.softmax(-1)
        h1_minus_h2 = probs_samples * ((probs_samples + eps).log() + self.minus_log_uniform)
        loss_entropy = h1_minus_h2.reshape(size_batch, -1).sum(-1)
        # pull prior towards posterior
        loss_conditional_prior = None  # torch.nn.functional.kl_div(input=logsoftmax_samples.detach(), target=self.get_prior(state_cond), log_target=False, reduction='none').reshape(size_batch, -1).sum(-1)

        coeff_schedule = cyclical_schedule(step=self.steps_trained, interval=self.interval_beta)
        loss_overall = self.beta * coeff_schedule * loss_entropy + loss_recon
        if loss_conditional_prior is not None:
            loss_overall += self.beta * loss_conditional_prior
        if loss_align is not None:
            loss_overall += self.beta * loss_align

        self.steps_trained += 1

        if not debug:
            return loss_overall, loss_recon, loss_entropy, loss_conditional_prior, loss_align, None, None, None, None, None, None, None
        else:
            with torch.no_grad():
                masks_agent_pred = (
                    torch.nn.functional.one_hot(logits_mask_agent_train.argmax(-1), obses_chosen.shape[-3] * obses_chosen.shape[-2])
                    .bool()
                    .reshape(size_batch, obses_chosen.shape[-3], obses_chosen.shape[-2])
                )
                obses_pred = self.decoder(layouts_train, masks_agent_pred)

                code_chosen = torch.eye(logits_samples.shape[-1], device=logits_samples.device, dtype=torch.long)[logits_samples.argmax(-1)]
                code_pred = self.encode_from_obs(obses_pred)
                ratio_aligned = (code_chosen == code_pred).reshape(size_batch, -1).all(-1).float().mean()

                dist_L1 = torch.abs(obses_pred.float() - obses_chosen.float())
                mask_perfect_recon = dist_L1.reshape(dist_L1.shape[0], -1).sum(-1) == 0
                ratio_perfect_recon = mask_perfect_recon.sum() / mask_perfect_recon.shape[0]
                mask_agent = obses_chosen[:, :, :, 0] == self.object_to_idx["agent"]
                dist_L1_mean = dist_L1.mean()
                dist_L1_nontrivial = dist_L1[mask_agent].mean()
                dist_L1_trivial = dist_L1[~mask_agent].mean()
                uniformity = (probs_samples.mean(0) - 1.0 / self.num_categories).abs_().mean()
                entropy_prior = None
            return (
                loss_overall,
                loss_recon,
                loss_entropy,
                loss_conditional_prior,
                loss_align,
                dist_L1_mean,
                dist_L1_nontrivial,
                dist_L1_trivial,
                uniformity,
                entropy_prior,
                ratio_perfect_recon,
                ratio_aligned,
            )


class CVAE_MiniGrid_UNET(torch.nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        sample_input,
        depth=3,
        width=512,
        num_categoricals=8,
        num_categories=8,
        atoms=4,
        beta=0.0005,
        KL_balance=True,
        maximize_entropy=False,
        alpha_KL_balance=0.8,
        activation=torch.nn.ReLU,
        interval_beta=2500,
        batchnorm=True,
        fn_output=lambda x: x,
        **kwargs,
    ):
        super(CVAE_MiniGrid_UNET, self).__init__(**kwargs)
        self.atoms = atoms
        self.num_categoricals, self.num_categories = num_categoricals, num_categories
        self.len_code = num_categoricals * num_categories
        self.encoder, self.decoder = (
            encoder,
            decoder,
        )  # we want future compatibility, use torch.nn.Identity for decoder!
        self.KL_balance, self.maximize_entropy = KL_balance, maximize_entropy
        self.beta, self.alpha_KL_balance = beta, alpha_KL_balance
        from minigrid import OBJECT_TO_IDX

        self.object_to_idx = OBJECT_TO_IDX
        self.interval_beta = interval_beta
        self.steps_trained = 0
        num_channels = sample_input.shape[1]
        assert sample_input.shape[2] == sample_input.shape[3], "only square inputs supported"
        self.size_input = sample_input.shape[2]
        self.sizes_conv, self.sizes_conv_original = [], []
        size_conv = self.size_input
        size_conv_original = self.size_input
        for i in range(4):
            self.sizes_conv.append(size_conv)
            self.sizes_conv_original.append(size_conv_original)
            size_conv = size_conv // 2
            size_conv_original = size_conv_original // 2
            if size_conv > 1 and size_conv % 2 == 1:
                size_conv = size_conv + 1

        self.conv_level1_left = Block_UNET(channels_in=num_channels, channels_out=32, activation=activation, batchnorm=batchnorm)
        self.pool_level12 = torch.nn.MaxPool2d(2, stride=None, padding=0)  # 8x8 to 4x4
        self.conv_level2_left = Block_UNET(channels_in=32, channels_out=64, activation=activation, batchnorm=batchnorm)
        self.pool_level23 = torch.nn.MaxPool2d(2, stride=None, padding=0)  # 4x4 to 2x2
        self.conv_level3_left = Block_UNET(channels_in=64, channels_out=128, activation=activation, batchnorm=batchnorm)
        self.pool_level34 = torch.nn.MaxPool2d(2, stride=None, padding=0)  # 2x2 to 1x1

        size_bottleneck = max(self.len_code, 128)
        self.code_extractor = torch.nn.Sequential(
            torch.nn.Conv2d(
                128,
                size_bottleneck,
                kernel_size=self.sizes_conv[-1],
                stride=1,
                padding=0,
                bias=not batchnorm,
            ),
            torch.nn.Flatten(),
            activation(),
            torch.nn.Linear(size_bottleneck, self.len_code),
        )

        self.conv_bottom = Block_UNET(
            channels_in=self.len_code,  # 256 +
            channels_out=256,
            channels_mid=size_bottleneck,
            activation=activation,
            batchnorm=batchnorm,
        )

        self.unpool_level43 = torch.nn.Upsample(scale_factor=2, mode="nearest")
        # self.unpool_level43 = torch.nn.ConvTranspose2d(256, 128, kernel_size=self.sizes_conv_original[-2], stride=1, bias=not batchnorm)
        self.conv_level3_right = Block_UNET(channels_in=128 * 3, channels_out=128, channels_mid=256, activation=activation, batchnorm=batchnorm)
        self.unpool_level32 = torch.nn.Upsample(scale_factor=2, mode="nearest")
        # self.unpool_level32 = torch.nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, bias=not batchnorm)
        self.conv_level2_right = Block_UNET(channels_in=64 * 3, channels_out=64, channels_mid=128, activation=activation, batchnorm=batchnorm)
        self.unpool_level21 = torch.nn.Upsample(scale_factor=2, mode="nearest")
        # self.unpool_level21 = torch.nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2, bias=not batchnorm)
        self.conv_level1_right = Block_UNET(channels_in=32 * 3, channels_out=32, channels_mid=64, activation=activation, batchnorm=batchnorm)
        self.out = torch.nn.Conv2d(32, num_channels, kernel_size=1, stride=1, padding=0)

        self.fn_output = fn_output

    def to(self, device):
        super().to(device)
        self.conv_level1_left.to(device)
        self.pool_level12.to(device)
        self.conv_level2_left.to(device)
        self.pool_level23.to(device)
        self.conv_level3_left.to(device)
        self.pool_level34.to(device)
        self.conv_bottom.to(device)
        self.code_extractor.to(device)
        self.unpool_level43.to(device)
        self.conv_level3_right.to(device)
        self.unpool_level32.to(device)
        self.conv_level2_right.to(device)
        self.unpool_level21.to(device)
        self.conv_level1_right.to(device)
        self.out.to(device)

    def parameters(self):
        parameters = []
        parameters += list(self.conv_level1_left.parameters())
        parameters += list(self.conv_level2_left.parameters())
        parameters += list(self.conv_level3_left.parameters())
        parameters += list(self.conv_bottom.parameters())
        parameters += list(self.code_extractor.parameters())
        parameters += list(self.unpool_level43.parameters())
        parameters += list(self.conv_level3_right.parameters())
        parameters += list(self.unpool_level32.parameters())
        parameters += list(self.conv_level2_right.parameters())
        parameters += list(self.unpool_level21.parameters())
        parameters += list(self.conv_level1_right.parameters())
        parameters += list(self.out.parameters())
        return parameters

    def get_prior(self, input_cond):
        return self.get_logsoftmax_prior(input_cond).exp()

    def get_logsoftmax_prior(self, input_cond):
        size_batch = input_cond.shape[0]
        (
            after_conv_level1_left,
            after_conv_level2_left,
            after_conv_level3_left,
            before_conv_bottom,
        ) = self.forward_left(input_cond)
        logit_samples = self.code_extractor(before_conv_bottom).reshape(size_batch, self.num_categoricals, self.num_categories)
        return logit_samples.log_softmax(-1)

    @torch.no_grad()
    def sample_from_prior(self, input_cond, size_batch=64, unique=False, topk=0):
        probs_prior = self.get_prior(input_cond)
        dist = torch.distributions.Categorical(logits=probs_prior)
        samples = torch.eye(self.num_categories, device=input_cond.device)[dist.sample([size_batch])].reshape(
            size_batch, self.num_categoricals, self.num_categories
        )
        if unique:
            samples_unique, excluded = unique_rows(samples.reshape(samples.shape[0], -1))
            samples_unique = samples_unique.reshape(samples_unique.shape[0], self.num_categoricals, self.num_categories)
            if topk > 0:
                prob_samples = (samples_unique * probs_prior).sum(-1).prod(-1)
                idxs_select = torch.topk(prob_samples, topk)[1]
                samples_unique = samples_unique[idxs_select]
            if input_cond.shape[0] == 1:
                input_cond = torch.repeat_interleave(input_cond, samples_unique.shape[0], 0)
            else:
                assert input_cond.shape[0] == samples_unique.shape[0]
            return self.forward(input_cond, samples=samples_unique)
        else:
            if topk > 0:
                prob_samples = (samples * probs_prior).sum(-1).prod(-1)
                idxs_select = torch.topk(prob_samples, topk)[1]
                samples = samples[idxs_select]
            if input_cond.shape[0] == 1:
                input_cond = torch.repeat_interleave(input_cond, samples.shape[0], 0)
            else:
                assert input_cond.shape[0] == samples.shape[0]
            return self.forward(input_cond, samples=samples)

    @torch.no_grad()
    def sample_from_uniform_prior(self, input_cond):
        tensors_to_mesh = []
        for idx_category in range(self.num_categoricals):
            tensors_to_mesh.append(torch.arange(self.num_categories, device=input_cond.device))
        indices_mesh = torch.meshgrid(*tensors_to_mesh)
        indices_mesh = torch.concatenate([indices.reshape(1, -1) for indices in indices_mesh], 0).permute(1, 0).contiguous()
        samples = torch.nn.functional.one_hot(indices_mesh, num_classes=self.num_categories)
        assert samples.shape[0] == self.num_categories**self.num_categoricals
        if input_cond.shape[0] == 1:
            input_cond = torch.repeat_interleave(input_cond, samples.shape[0], 0)
        else:
            assert input_cond.shape[0] == samples.shape[0]
        return self.forward(input_cond, samples=samples)

    def forward_left(self, input_cond):
        input_cond = input_cond.float()

        before_conv_level1_left = input_cond

        after_conv_level1_left_original = self.conv_level1_left(before_conv_level1_left)
        if after_conv_level1_left_original.shape[-1] % 2 == 0:
            after_conv_level1_left = after_conv_level1_left_original
        else:
            after_conv_level1_left = torch.nn.functional.pad(after_conv_level1_left_original, (0, 1, 0, 1))

        before_conv_level2_left = self.pool_level12(after_conv_level1_left)
        after_conv_level2_left_original = self.conv_level2_left(before_conv_level2_left)
        if after_conv_level2_left_original.shape[-1] % 2 == 0:
            after_conv_level2_left = after_conv_level2_left_original
        else:
            after_conv_level2_left = torch.nn.functional.pad(after_conv_level2_left_original, (0, 1, 0, 1))
        before_conv_level3_left = self.pool_level23(after_conv_level2_left)
        after_conv_level3_left_original = self.conv_level3_left(before_conv_level3_left)
        if after_conv_level3_left_original.shape[-1] % 2 == 0:
            after_conv_level3_left = after_conv_level3_left_original
        else:
            after_conv_level3_left = torch.nn.functional.pad(after_conv_level3_left_original, (0, 1, 0, 1))
        before_conv_bottom = self.pool_level34(after_conv_level3_left)

        return (
            after_conv_level1_left_original,
            after_conv_level2_left_original,
            after_conv_level3_left_original,
            before_conv_bottom,
        )

    def forward_right(
        self,
        before_conv_bottom,
        after_conv_level3_left,
        after_conv_level2_left,
        after_conv_level1_left,
        samples,
    ):
        size_batch = samples.shape[0]
        samples_tiled = samples.reshape(size_batch, -1, 1, 1).repeat(1, 1, before_conv_bottom.shape[-2], before_conv_bottom.shape[-1])
        after_conv_bottom = self.conv_bottom(samples_tiled)
        # after_conv_bottom = self.conv_bottom(torch.cat([before_conv_bottom, samples_tiled], 1))

        unpooled_43 = self.unpool_level43(after_conv_bottom)
        if after_conv_level3_left.shape[-1] == unpooled_43.shape[-1] - 1:
            unpooled_43 = unpooled_43[:, :, :-1, :-1]
        before_conv_level3_right = torch.cat([after_conv_level3_left, unpooled_43], 1)
        after_conv_level3_right = self.conv_level3_right(before_conv_level3_right)

        unpooled_32 = self.unpool_level32(after_conv_level3_right)
        if after_conv_level2_left.shape[-1] == unpooled_32.shape[-1] - 1:
            unpooled_32 = unpooled_32[:, :, :-1, :-1]
        before_conv_level2_right = torch.cat([after_conv_level2_left, unpooled_32], 1)
        after_conv_level2_right = self.conv_level2_right(before_conv_level2_right)

        unpooled_21 = self.unpool_level21(after_conv_level2_right)
        if after_conv_level1_left.shape[-1] == unpooled_21.shape[-1] - 1:
            unpooled_21 = unpooled_21[:, :, :-1, :-1]
        before_conv_level1_right = torch.cat([after_conv_level1_left, unpooled_21], 1)
        after_conv_level1_right = self.conv_level1_right(before_conv_level1_right)

        output = self.out(after_conv_level1_right)
        return output

    def forward(self, input_cond, input_targ=None, samples=None, return_logits_and_probs=False):
        size_batch = input_cond.shape[0]
        input_cond = input_cond.float()
        if input_targ is None:
            input_targ = input_cond
        else:
            input_targ = input_targ.float()

        (
            after_conv_level1_left,
            after_conv_level2_left,
            after_conv_level3_left,
            before_conv_bottom,
        ) = self.forward_left(input_cond)
        if input_targ is None:
            before_conv_bottom_targ = before_conv_bottom
        else:
            _, _, _, before_conv_bottom_targ = self.forward_left(input_targ)

        if samples is None:
            logits_samples = self.code_extractor(before_conv_bottom_targ).reshape(size_batch, self.num_categoricals, self.num_categories)
            probs_samples = logits_samples.softmax(-1)
            # samples = torch.distributions.Categorical(probs=probs_samples).sample()
            # samples = torch.distributions.OneHotCategoricalStraightThrough(probs_samples).rsample()

            # #################################
            # Sample from Gumbel
            # eps = 1e-7
            # temp = 1.0
            # u = torch.rand_like(logits_samples)
            # g = -torch.log(-torch.log(u + eps) + eps)

            # # Gumbel-Softmax sample
            # probs_samples = torch.nn.functional.softmax((logits_samples + g) / temp, dim=-1)
            # # #################################
            # # samples = torch.distributions.OneHotCategoricalStraightThrough(probs_samples).rsample()
            # ###################################
            # samples = probs_samples
            from BACKUP.gumbel_rao import gumbel_rao

            samples = gumbel_rao(logits_samples, k=1024, temp=1.0, straight_through=False)  # , I=samples

        output = self.forward_right(
            before_conv_bottom,
            after_conv_level3_left,
            after_conv_level2_left,
            after_conv_level1_left,
            samples,
        )

        output = self.fn_output(output)
        if return_logits_and_probs:
            return output, logits_samples, probs_samples
        else:
            return output

    def compute_loss(
        self,
        batch_processed,
        debug=False,
    ):
        (
            batch_obs_curr,
            batch_action,
            batch_reward,
            batch_obs_next,
            batch_done,
            batch_obs_targ,
            weights,
            batch_idxes,
            info,
        ) = batch_processed

        size_batch = batch_obs_targ.shape[0]
        with torch.no_grad():
            obs_chosen = batch_obs_targ
            obs_cond = batch_obs_curr

        state_recon = self.encoder(obs_chosen)
        state_cond = self.encoder(obs_cond)

        state_pred, logit_samples, probs_samples = self.forward(state_cond, input_targ=state_recon, return_logits_and_probs=True)

        loss_recon = torch.nn.functional.mse_loss(state_pred, state_recon.float(), reduction="none").reshape(size_batch, -1).sum(-1)

        # maximize the kl-loss
        eps = 1e-7
        h1_minus_h2 = probs_samples * (probs_samples * self.num_categories + eps).log()

        loss_entropy = h1_minus_h2.reshape(size_batch, -1).sum(-1)
        # pull prior towards posterior
        loss_conditional_prior = torch.zeros_like(
            loss_entropy
        )  # torch.nn.functional.kl_div(input=logsoftmax_samples.detach(), target=self.get_prior(state_cond), log_target=False, reduction='none').reshape(size_batch, -1).sum(-1)

        coeff_schedule = cyclical_schedule(step=self.steps_trained, interval=self.interval_beta)
        loss_overall = self.beta * loss_conditional_prior + self.beta * coeff_schedule * loss_entropy + loss_recon

        self.steps_trained += 1

        if not debug:
            return (
                loss_overall,
                loss_recon,
                loss_entropy,
                loss_conditional_prior,
                None,
                None,
                None,
                None,
                None,
                None,
            )
        else:
            with torch.no_grad():
                obs_pred = self.decoder(state_pred, compact=True, learn=False)
                dist_L1 = torch.abs(obs_pred - obs_chosen).float()
                mask_perfect_recon = dist_L1.reshape(dist_L1.shape[0], -1).sum(-1) == 0
                ratio_perfect_recon = mask_perfect_recon.sum() / mask_perfect_recon.shape[0]
                mask_agent = obs_chosen[:, :, :, 0] == self.object_to_idx["agent"]
                dist_L1_mean = dist_L1.mean()
                dist_L1_nontrivial = dist_L1[mask_agent].mean()
                dist_L1_trivial = dist_L1[~mask_agent].mean()
                uniformity = (probs_samples.mean(0) - 1.0 / self.num_categories).abs_().mean()
                # dist = torch.distributions.Categorical(logits=logsoftmax_prior)
                # entropy_prior = dist.entropy().mean()
                entropy_prior = None
            return (
                loss_overall,
                loss_recon,
                loss_entropy,
                loss_conditional_prior,
                dist_L1_mean,
                dist_L1_nontrivial,
                dist_L1_trivial,
                uniformity,
                entropy_prior,
                ratio_perfect_recon,
            )


class Encoder_MiniGrid_Separate(torch.nn.Module):
    def __init__(self):
        super(Encoder_MiniGrid_Separate, self).__init__()
        from minigrid import OBJECT_TO_IDX, COLOR_TO_IDX

        self.object_to_idx = OBJECT_TO_IDX
        self.color_to_idx = COLOR_TO_IDX

    @torch.no_grad()
    def forward(self, obs):
        if len(obs.shape) == 3:
            obs = obs[None, :, :, :]
        size_batch = obs.shape[0]
        mask_agent = obs[:, :, :, 0] == self.object_to_idx["agent"]
        colors = obs[mask_agent][:, 1]
        mask_on_lava = colors == self.color_to_idx["yellow"]
        mask_on_goal = colors == self.color_to_idx["green"]
        mask_agent_on_empty = (~mask_on_lava & ~mask_on_goal).reshape(size_batch, 1, 1) & mask_agent
        mask_agent_on_lava = mask_on_lava.reshape(size_batch, 1, 1) & mask_agent
        mask_agent_on_goal = mask_on_goal.reshape(size_batch, 1, 1) & mask_agent
        layout = obs[:, :, :, [0]]
        layout[mask_agent_on_empty] = self.object_to_idx["empty"]
        layout[mask_agent_on_lava] = self.object_to_idx["lava"]
        layout[mask_agent_on_goal] = self.object_to_idx["goal"]
        return layout, mask_agent


class Decoder_MiniGrid_Separate(torch.nn.Module):
    def __init__(self):
        super(Decoder_MiniGrid_Separate, self).__init__()
        from minigrid import OBJECT_TO_IDX, COLOR_TO_IDX

        self.object_to_idx, self.color_to_idx = OBJECT_TO_IDX, COLOR_TO_IDX

    @torch.no_grad()
    def forward(self, layout, mask_agent):
        size_batch = mask_agent.shape[0]
        if layout.shape[0] == 1:
            layout = layout.repeat(size_batch, 1, 1, 1)
        mask_agent = mask_agent.bool()
        obs = torch.cat([layout, torch.zeros_like(layout)], dim=-1)
        colors = torch.full([size_batch], self.color_to_idx["red"], device=obs.device, dtype=obs.dtype)
        mask_lava = obs[:, :, :, 0] == self.object_to_idx["lava"]
        mask_on_lava = torch.logical_and(mask_agent.reshape(size_batch, -1), mask_lava.reshape(size_batch, -1)).any(-1)
        colors[mask_on_lava] = self.color_to_idx["yellow"]
        mask_goal = obs[:, :, :, 0] == self.object_to_idx["goal"]
        mask_on_goal = torch.logical_and(mask_agent.reshape(size_batch, -1), mask_goal.reshape(size_batch, -1)).any(-1)
        colors[mask_on_goal] = self.color_to_idx["green"]
        mask_agent = mask_agent.reshape(size_batch, layout.shape[-3], layout.shape[-2])

        obs[mask_agent] = torch.stack([torch.full([size_batch], self.object_to_idx["agent"], device=obs.device, dtype=obs.dtype), colors], dim=-1)
        return obs


class Pseudo_Encoder_MiniGrid(torch.nn.Module):
    def __init__(self, atoms=4, compact=False):
        super(Pseudo_Encoder_MiniGrid, self).__init__()
        self.atoms = atoms
        self.compact = compact

    def to(self, device):
        super().to(device)

    def parameters(self):
        return []

    @torch.no_grad()
    def forward(self, obs, from_compact=True):
        if self.compact:
            return (2.0 / (self.atoms - 1)) * (obs.permute(0, 3, 1, 2).contiguous().float()) - 1.0
        elif from_compact:
            return torch.nn.functional.one_hot(obs.long(), num_classes=self.atoms).reshape(*obs.shape[:-1], -1).permute(0, 3, 1, 2).contiguous().detach()
        else:
            return (2.0 / (self.atoms - 1)) * obs.reshape(*obs.shape[:-2], -1).permute(0, 3, 1, 2).contiguous().float() / (self.atoms - 1) - 1.0


class Pseudo_Decoder_MiniGrid(torch.nn.Module):
    def __init__(self, atoms=4, compact=False):
        super(Pseudo_Decoder_MiniGrid, self).__init__()
        self.atoms = atoms
        self.compact = compact

    def to(self, device):
        super().to(device)

    def parameters(self):
        return []

    def forward(self, state_pred, compact=False, learn=False):
        if self.compact:
            if learn:
                return (state_pred.permute(0, 2, 3, 1).contiguous() + 1) * ((self.atoms - 1) / 2.0)
            else:
                return (torch.round(((state_pred + 1) * ((self.atoms - 1) / 2.0)).clamp(0, self.atoms - 1))).long().permute(0, 2, 3, 1).contiguous()
        else:
            obs_pred = state_pred.reshape(state_pred.shape[0], -1, self.atoms, *state_pred.shape[2:])
        if compact:
            obs_pred = state_pred.reshape(state_pred.shape[0], -1, self.atoms, *state_pred.shape[2:])
            return obs_pred.argmax(2).permute(0, 2, 3, 1).contiguous()
        else:
            obs_pred = state_pred
            return obs_pred.permute(0, 2, 3, 1).contiguous()


class Embedder_MiniGrid_BOW(torch.nn.Module):  # adapted from BabyAI 1.1
    def __init__(self, max_value=32, dim_embed=8, channels_obs=2, height=8, width=8, ebd_pos=False):
        super().__init__()
        self.max_value = max_value
        self.dim_embed = dim_embed
        self.width, self.height = width, height
        self.ebd_pos = ebd_pos
        self.channels_obs = channels_obs + int(self.ebd_pos) * 2
        if self.ebd_pos:
            self.meshgrid_x, self.meshgrid_y = torch.meshgrid(torch.arange(self.width), torch.arange(self.height), indexing="ij")
            self.meshgrid_x = self.meshgrid_x.reshape(1, self.width, self.height, 1).contiguous().detach()
            self.meshgrid_y = self.meshgrid_y.reshape(1, self.width, self.height, 1).contiguous().detach()
            self.max_value = max(max_value, self.width, self.height)
        self.embedding = torch.nn.Embedding(self.channels_obs * self.max_value, dim_embed)  # NOTE(H): +2 for X and Y
        if self.channels_obs > 1:
            offset = []
            for index_channel in range(self.channels_obs):
                offset.append(index_channel * self.max_value)
            self.register_buffer("offsets", torch.Tensor(offset).long().reshape(1, 1, 1, -1).contiguous().to(self.embedding.weight.device))

    def to(self, device):
        super().to(device)
        self.embedding.to(device)
        if self.ebd_pos:
            self.meshgrid_x = self.meshgrid_x.to(device)
            self.meshgrid_y = self.meshgrid_y.to(device)
        if self.channels_obs > 1:
            self.offsets = self.offsets.to(device)

    def parameters(self):
        parameters = list(self.embedding.parameters())
        return parameters

    # @profile
    def forward(self, inputs):
        with torch.no_grad():
            if self.ebd_pos:
                inputs = torch.cat(
                    [
                        inputs,
                        self.meshgrid_x.expand(inputs.shape[0], self.width, self.height, 1).detach(),
                        self.meshgrid_y.expand(inputs.shape[0], self.width, self.height, 1).detach(),
                    ],
                    dim=-1,
                )
            else:
                inputs = inputs.long()
            if self.channels_obs > 1:
                inputs += self.offsets
        # permute so it can be fed to conv layers in the encoder
        return self.embedding(inputs.detach()).sum(-2).permute(0, 3, 1, 2).contiguous()


class Encoder_MiniGrid(torch.nn.Module):
    """
    minigrid observation encoder from the Conscious Planning paper
    inputs an observation from the environment and outputs a vector representation of the states
    """

    def __init__(self, dim_embed, sample_obs, norm=True, append_pos=False, activation=torch.nn.ReLU):
        super(Encoder_MiniGrid, self).__init__()
        self.norm = norm
        self.activation = activation
        self.embedder = Embedder_MiniGrid_BOW(
            dim_embed=dim_embed, width=sample_obs.shape[-3], height=sample_obs.shape[-2], channels_obs=sample_obs.shape[-1], ebd_pos=bool(append_pos)
        )
        self.layers = ResidualBlock(len_in=dim_embed, width=None, kernel_size=3, depth=2, stride=1, padding=1, activation=activation)

    def to(self, device):
        super().to(device)
        self.embedder.to(device)
        self.layers.to(device)

    def parameters(self):
        parameters = list(self.embedder.parameters())
        parameters += list(self.layers.parameters())
        return parameters

    # @profile
    def forward(self, obs_minigrid):
        rep_bow = self.embedder(obs_minigrid)
        return self.layers(rep_bow)


class Binder_MiniGrid(torch.nn.Module):
    """
    create a local perception field with state_curr and state_targ
    """

    def __init__(self, sample_input, len_rep, norm=True, activation=torch.nn.ReLU, num_heads=1, size_bottleneck=4, size_field=8, type_arch="CP"):
        super(Binder_MiniGrid, self).__init__()
        self.norm = norm
        dim_embed = sample_input.shape[1]
        self.len_rep = len_rep
        self.len_out = 2 * len_rep
        self.activation = activation
        self.local_perception = "local" in type_arch.lower()
        if self.local_perception:
            self.extractor_fields = torch.nn.Conv2d(dim_embed, len_rep, kernel_size=size_field, stride=1, padding=0)
            # self.query = torch.nn.parameter.Parameter(data=torch.nn.init.uniform_(torch.empty(len_rep)).reshape(1, 1, len_rep), requires_grad=True)
            self.register_buffer("query", torch.zeros(1, 1, len_rep))
            if size_bottleneck == 0:
                print("BINDER: size_bottleneck == 0, fall back to standard attention")
                self.attn = torch.nn.MultiheadAttention(embed_dim=len_rep, num_heads=num_heads, kdim=len_rep, vdim=len_rep, batch_first=True, dropout=0.0)
            else:
                # from nocache_attention import AttentionNoCache
                # self.attn = AttentionNoCache(activation=torch.nn.Softmax(-1), size_bottleneck=size_bottleneck)
                self.attn = TopKMultiheadAttention(
                    embed_dim=len_rep,
                    num_heads=num_heads,
                    kdim=len_rep,
                    vdim=len_rep,
                    batch_first=True,
                    dropout=0.0,
                    size_bottleneck=size_bottleneck,
                    no_out_proj=num_heads == 1,
                )
            if self.norm:
                self.layer_norm_1 = torch.nn.LayerNorm(len_rep)
                self.layer_norm_2 = torch.nn.LayerNorm(len_rep)
        else:
            self.flattener = torch.nn.Sequential(
                activation(False),
                torch.nn.Flatten(),
                torch.nn.Linear(sample_input.shape[-1] * sample_input.shape[-2] * sample_input.shape[-3], len_rep),
            )

    def to(self, device):
        super().to(device)
        if self.local_perception:
            self.extractor_fields.to(device)
            self.query = self.query.to(device)
            self.attn.to(device)
            if self.norm:
                self.layer_norm_1.to(device)
                self.layer_norm_2.to(device)
        else:
            self.flattener.to(device)

    def parameters(self):
        if self.local_perception:
            parameters = list(self.extractor_fields.parameters())
            if self.norm:
                parameters += list(self.layer_norm_1.parameters())
                parameters += list(self.layer_norm_2.parameters())
            parameters += list(self.attn.parameters())
            # parameters += [self.query]
            return parameters
        else:
            return list(self.flattener.parameters())

    def extract_local_field(self, state):
        size_batch = state.shape[0]
        fields = self.extractor_fields(state).permute(0, 2, 3, 1).reshape(size_batch, -1, self.len_rep)
        if self.norm:
            fields = self.layer_norm_1(fields)
        state_local, _ = self.attn(self.query.expand(size_batch, 1, self.len_rep), fields, fields, need_weights=False)
        # state_local = self.attn(self.query.expand(size_batch, 1, self.len_rep), fields, fields)
        if self.norm:
            state_local = self.layer_norm_2(state_local)
        state_local = self.activation()(state_local)
        state_local = state_local.reshape(size_batch, self.len_rep)
        return state_local

    def forward(self, state_curr, state_targ, return_curr=False):
        size_batch = state_curr.shape[0]
        states_stacked_curr_targ = torch.cat([state_curr, state_targ], dim=0)
        if self.local_perception:
            state_local_curr_targ = self.extract_local_field(states_stacked_curr_targ)
        else:
            state_local_curr_targ = self.flattener(states_stacked_curr_targ)
        state_local_curr, state_local_targ = torch.split(state_local_curr_targ, [size_batch, size_batch], dim=0)
        state_binded = torch.cat([state_local_curr, state_local_targ], dim=-1)
        if return_curr:
            return state_binded, state_local_curr
        else:
            return state_binded


class Predictor_MiniGrid(torch.nn.Module):
    """
    on top of the extracted states, this predicts interesting values
    """

    def __init__(
        self,
        num_actions,
        len_input,
        depth=3,
        width=256,
        activation=torch.nn.ReLU,
        norm=True,
        dict_head=[{"len_predict": None, "dist_out": True, "value_min": 0.0, "value_max": 1.0, "atoms": 4, "classify": False}],
    ):
        super(Predictor_MiniGrid, self).__init__()
        self.len_input = len_input
        self.num_actions = num_actions
        self.dict_head = dict_head
        self.dist_output = bool(dict_head["dist_out"])
        self.norm = norm
        if dict_head["len_predict"] is None:
            self.len_predict = self.num_actions
        else:
            self.len_predict = dict_head["len_predict"]
        if dict_head["dist_out"]:
            assert "value_min" in dict_head and "value_max" in dict_head
            assert "atoms" in dict_head and "classify" in dict_head
            self.histogram_converter = HistogramConverter(value_min=dict_head["value_min"], value_max=dict_head["value_max"], atoms=dict_head["atoms"])
            self.len_output = self.len_predict * dict_head["atoms"]
            self.atoms = dict_head["atoms"]
            self.classify = dict_head["classify"]
        else:
            self.histogram_converter = None
            self.len_output = self.len_predict
            # self.value_min, self.value_max = dict_head["value_min"], dict_head["value_max"]

        self.layers = []
        for idx_layer in range(depth):
            len_in, len_out = width, width
            if idx_layer == 0:
                len_in = self.len_input
            if idx_layer == depth - 1:
                len_out = self.len_output
            if idx_layer > 0:
                # if self.norm:
                #     self.layers.append(torch.nn.LayerNorm(len_in))
                self.layers.append(activation(True))
            self.layers.append(torch.nn.Linear(len_in, len_out))
        self.layers = torch.nn.Sequential(*self.layers)
        init_weights(self.layers)
        # with torch.no_grad():
        #     torch.nn.init.orthogonal_(self.layers[-1].weight, 0.01)
        #     self.layers[-1].bias.data.zero_()

    def to(self, device):
        super().to(device)
        self.layers.to(device)
        if self.histogram_converter is not None:
            self.histogram_converter.to(device)

    def parameters(self):
        return list(self.layers.parameters())

    def forward(self, input, action=None, scalarize=False):
        size_batch = input.shape[0]
        predicted = self.layers(input.reshape(size_batch, -1))
        if action is not None:
            assert action.device == predicted.device
            predicted = predicted.reshape(size_batch, self.len_predict, -1)
            predicted = predicted[torch.arange(size_batch, device=predicted.device), action.squeeze()]
        if self.dist_output:
            if action is None:
                predicted = predicted.reshape(size_batch, -1, self.atoms).contiguous()
            else:
                predicted = predicted.reshape(size_batch, self.atoms).contiguous()
            if scalarize:
                with torch.no_grad():
                    if self.classify:
                        return self.histogram_converter.support[predicted.argmax(-1)]
                    else:
                        return self.histogram_converter.from_histogram(predicted, logits=True)
            else:
                return predicted
        else:
            return predicted  # .clamp(self.value_min, self.value_max)
