# Single File Import version of GDF found in Stable Cascade

import torch
import numpy as np

# --- Loss Weighting
class BaseLossWeight():
    def weight(self, logSNR):
        raise NotImplementedError("this method needs to be overridden")

    def __call__(self, logSNR, *args, shift=1, clamp_range=None, **kwargs):
        clamp_range = [-1e9, 1e9] if clamp_range is None else clamp_range
        if shift != 1:
            logSNR = logSNR.clone() + 2 * np.log(shift)
        return self.weight(logSNR, *args, **kwargs).clamp(*clamp_range)

class ComposedLossWeight(BaseLossWeight):
    def __init__(self, div, mul):
        self.mul = [mul] if isinstance(mul, BaseLossWeight) else mul
        self.div = [div] if isinstance(div, BaseLossWeight) else div

    def weight(self, logSNR):
        prod, div = 1, 1
        for m in self.mul:
            prod *= m.weight(logSNR)
        for d in self.div:
            div *= d.weight(logSNR)
        return prod/div

class ConstantLossWeight(BaseLossWeight):
    def __init__(self, v=1):
        self.v = v

    def weight(self, logSNR):
        return torch.ones_like(logSNR) * self.v

class SNRLossWeight(BaseLossWeight):
    def weight(self, logSNR):
        return logSNR.exp()

class P2LossWeight(BaseLossWeight):
    def __init__(self, k=1.0, gamma=1.0, s=1.0):
        self.k, self.gamma, self.s = k, gamma, s

    def weight(self, logSNR):
        return (self.k + (logSNR * self.s).exp()) ** -self.gamma

class SNRPlusOneLossWeight(BaseLossWeight):
    def weight(self, logSNR):
        return logSNR.exp() + 1

class MinSNRLossWeight(BaseLossWeight):
    def __init__(self, max_snr=5):
        self.max_snr = max_snr

    def weight(self, logSNR):
        return logSNR.exp().clamp(max=self.max_snr)

class MinSNRPlusOneLossWeight(BaseLossWeight):
    def __init__(self, max_snr=5):
        self.max_snr = max_snr

    def weight(self, logSNR):
        return (logSNR.exp() + 1).clamp(max=self.max_snr)

class TruncatedSNRLossWeight(BaseLossWeight):
    def __init__(self, min_snr=1):
        self.min_snr = min_snr

    def weight(self, logSNR):
        return logSNR.exp().clamp(min=self.min_snr)

class SechLossWeight(BaseLossWeight):
    def __init__(self, div=2):
        self.div = div

    def weight(self, logSNR):
        return 1/(logSNR/self.div).cosh()

class DebiasedLossWeight(BaseLossWeight):
    def weight(self, logSNR):
        return 1/logSNR.exp().sqrt()

class SigmoidLossWeight(BaseLossWeight):
    def __init__(self, s=1):
        self.s = s

    def weight(self, logSNR):
        return (logSNR * self.s).sigmoid()

class AdaptiveLossWeight(BaseLossWeight):
    def __init__(self, logsnr_range=[-10, 10], buckets=300, weight_range=[1e-7, 1e7]):
        self.bucket_ranges = torch.linspace(logsnr_range[0], logsnr_range[1], buckets-1)
        self.bucket_losses = torch.ones(buckets)
        self.weight_range = weight_range

    def weight(self, logSNR):
        indices = torch.searchsorted(self.bucket_ranges.to(logSNR.device), logSNR)
        return (1/self.bucket_losses.to(logSNR.device)[indices]).clamp(*self.weight_range)

    def update_buckets(self, logSNR, loss, beta=0.99):
        indices = torch.searchsorted(self.bucket_ranges.to(logSNR.device), logSNR).cpu()
        self.bucket_losses[indices] = self.bucket_losses[indices]*beta + loss.detach().cpu() * (1-beta)


# --- Noise Conditions
class BaseNoiseCond():
    def __init__(self, *args, shift=1, clamp_range=None, **kwargs):
        clamp_range = [-1e9, 1e9] if clamp_range is None else clamp_range
        self.shift = shift
        self.clamp_range = clamp_range
        self.setup(*args, **kwargs)

    def setup(self, *args, **kwargs):
        pass # this method is optional, override it if required

    def cond(self, logSNR):
        raise NotImplementedError("this method needs to be overriden")

    def __call__(self, logSNR):
        if self.shift != 1:
            logSNR = logSNR.clone() + 2 * np.log(self.shift)
        return self.cond(logSNR).clamp(*self.clamp_range)

class CosineTNoiseCond(BaseNoiseCond):
    def setup(self, s=0.008, clamp_range=[0, 1]): # [0.0001, 0.9999]
        self.s = torch.tensor([s])
        self.clamp_range = clamp_range
        self.min_var = torch.cos(self.s / (1 + self.s) * torch.pi * 0.5) ** 2

    def cond(self, logSNR):
        var = logSNR.sigmoid()
        var = var.clamp(*self.clamp_range)
        s, min_var = self.s.to(var.device), self.min_var.to(var.device)
        t = (((var * min_var) ** 0.5).acos() / (torch.pi * 0.5)) * (1 + s) - s
        return t

class EDMNoiseCond(BaseNoiseCond):
    def cond(self, logSNR):
        return -logSNR/8

class SigmoidNoiseCond(BaseNoiseCond):
    def cond(self, logSNR):
        return (-logSNR).sigmoid()

class LogSNRNoiseCond(BaseNoiseCond):
    def cond(self, logSNR):
        return logSNR

class EDMSigmaNoiseCond(BaseNoiseCond):
    def setup(self, sigma_data=1):
        self.sigma_data = sigma_data

    def cond(self, logSNR):
        return torch.exp(-logSNR / 2) * self.sigma_data

class RectifiedFlowsNoiseCond(BaseNoiseCond):
    def cond(self, logSNR):
        _a = logSNR.exp() - 1
        _a[_a == 0] = 1e-3 # Avoid division by zero
        a = 1 + (2-(2**2 + 4*_a)**0.5) / (2*_a)
        return a

# Any NoiseCond that cannot be described easily as a continuous function of t
# It needs to define self.x and self.y in the setup() method
class PiecewiseLinearNoiseCond(BaseNoiseCond):
    def setup(self):
        self.x = None
        self.y = None

    def piecewise_linear(self, y, xs, ys):
        indices = (len(xs)-2) - torch.searchsorted(ys.flip(dims=(-1,))[:-2], y)  
        x_min, x_max = xs[indices], xs[indices+1]
        y_min, y_max = ys[indices], ys[indices+1]
        x = x_min + (x_max - x_min) * (y - y_min) / (y_max - y_min)
        return x

    def cond(self, logSNR):
        var = logSNR.sigmoid()
        t = self.piecewise_linear(var, self.x.to(var.device), self.y.to(var.device)) # .mul(1000).round().clamp(min=0)
        return t

class StableDiffusionNoiseCond(PiecewiseLinearNoiseCond):
    def setup(self, linear_range=[0.00085, 0.012], total_steps=1000):
        self.total_steps = total_steps
        linear_range_sqrt = [r**0.5 for r in linear_range]
        self.x = torch.linspace(0, 1, total_steps+1)

        alphas = 1-(linear_range_sqrt[0]*(1-self.x) + linear_range_sqrt[1]*self.x)**2
        self.y = alphas.cumprod(dim=-1)

    def cond(self, logSNR):
        return super().cond(logSNR).clamp(0, 1)

class DiscreteNoiseCond(BaseNoiseCond):
    def setup(self, noise_cond, steps=1000, continuous_range=[0, 1]):
        self.noise_cond = noise_cond
        self.steps = steps
        self.continuous_range = continuous_range

    def cond(self, logSNR):
        cond = self.noise_cond(logSNR)
        cond = (cond-self.continuous_range[0]) / (self.continuous_range[1]-self.continuous_range[0])
        return cond.mul(self.steps).long()

# --- Samplers
class SimpleSampler():
    def __init__(self, gdf):
        self.gdf = gdf
        self.current_step = -1

    def __call__(self, *args, **kwargs):
        self.current_step += 1
        return self.step(*args, **kwargs)

    def init_x(self, shape):
        return torch.randn(*shape)

    def step(self, x, x0, epsilon, logSNR, logSNR_prev):
        raise NotImplementedError("You should override the 'apply' function.")

class DDIMSampler(SimpleSampler):
    def step(self, x, x0, epsilon, logSNR, logSNR_prev, eta=0):
        a, b = self.gdf.input_scaler(logSNR)
        if len(a.shape) == 1:
            a, b = a.view(-1, *[1]*(len(x0.shape)-1)), b.view(-1, *[1]*(len(x0.shape)-1))

        a_prev, b_prev = self.gdf.input_scaler(logSNR_prev)
        if len(a_prev.shape) == 1:
            a_prev, b_prev = a_prev.view(-1, *[1]*(len(x0.shape)-1)), b_prev.view(-1, *[1]*(len(x0.shape)-1))

        sigma_tau = eta * (b_prev**2 / b**2).sqrt() * (1 - a**2 / a_prev**2).sqrt() if eta > 0 else 0
        # x = a_prev * x0 + (1 - a_prev**2 - sigma_tau ** 2).sqrt() * epsilon + sigma_tau * torch.randn_like(x0)
        x = a_prev * x0 + (b_prev**2 - sigma_tau**2).sqrt() * epsilon + sigma_tau * torch.randn_like(x0)
        return x

class DDPMSampler(DDIMSampler):
    def step(self, x, x0, epsilon, logSNR, logSNR_prev, eta=1):
        return super().step(x, x0, epsilon, logSNR, logSNR_prev, eta)

class LCMSampler(SimpleSampler):
    def step(self, x, x0, epsilon, logSNR, logSNR_prev):        
        a_prev, b_prev = self.gdf.input_scaler(logSNR_prev)
        if len(a_prev.shape) == 1:
            a_prev, b_prev = a_prev.view(-1, *[1]*(len(x0.shape)-1)), b_prev.view(-1, *[1]*(len(x0.shape)-1))
        return x0 * a_prev + torch.randn_like(epsilon) * b_prev

# --- Scalers
class BaseScaler():
    def __init__(self):
        self.stretched_limits = None

    def setup_limits(self, schedule, input_scaler, stretch_max=True, stretch_min=True, shift=1):
        min_logSNR = schedule(torch.ones(1), shift=shift)
        max_logSNR = schedule(torch.zeros(1), shift=shift)

        min_a, max_b = [v.item() for v in input_scaler(min_logSNR)] if stretch_max else [0, 1]
        max_a, min_b = [v.item() for v in input_scaler(max_logSNR)] if stretch_min else [1, 0]
        self.stretched_limits = [min_a, max_a, min_b, max_b]
        return self.stretched_limits

    def stretch_limits(self, a, b):
        min_a, max_a, min_b, max_b = self.stretched_limits
        return (a - min_a) / (max_a - min_a), (b - min_b) / (max_b - min_b)

    def scalers(self, logSNR):
        raise NotImplementedError("this method needs to be overridden")

    def __call__(self, logSNR):
        a, b = self.scalers(logSNR)
        if self.stretched_limits is not None:
            a, b = self.stretch_limits(a, b)
        return a, b

class VPScaler(BaseScaler):
    def scalers(self, logSNR):
        a_squared = logSNR.sigmoid()
        a = a_squared.sqrt()
        b = (1-a_squared).sqrt()
        return a, b

class LERPScaler(BaseScaler):
    def scalers(self, logSNR):
        _a = logSNR.exp() - 1
        _a[_a == 0] = 1e-3 # Avoid division by zero
        a = 1 + (2-(2**2 + 4*_a)**0.5) / (2*_a)
        b = 1-a
        return a, b

# --- Schedulers
class BaseSchedule():
    def __init__(self, *args, force_limits=True, discrete_steps=None, shift=1, **kwargs):
        self.setup(*args, **kwargs)
        self.limits = None
        self.discrete_steps = discrete_steps
        self.shift = shift
        if force_limits:
            self.reset_limits()

    def reset_limits(self, shift=1, disable=False):
        try:
            self.limits = None if disable else self(torch.tensor([1.0, 0.0]), shift=shift).tolist() # min, max
            return self.limits
        except Exception:
            print("WARNING: this schedule doesn't support t and will be unbounded")
            return None

    def setup(self, *args, **kwargs):
        raise NotImplementedError("this method needs to be overriden")

    def schedule(self, *args, **kwargs):
        raise NotImplementedError("this method needs to be overriden")

    def __call__(self, t, *args, shift=1, **kwargs):
        if isinstance(t, torch.Tensor):
            batch_size = None
            if self.discrete_steps is not None:
                if t.dtype != torch.long:
                    t = (t * (self.discrete_steps-1)).round().long()
                t = t / (self.discrete_steps-1)
            t = t.clamp(0, 1)
        else:
            batch_size = t
            t = None
        logSNR = self.schedule(t, batch_size, *args, **kwargs)
        if shift*self.shift != 1:
            logSNR += 2 * np.log(1/(shift*self.shift))
        if self.limits is not None:
            logSNR = logSNR.clamp(*self.limits)
        return logSNR

class CosineSchedule(BaseSchedule):
    def setup(self, s=0.008, clamp_range=[0.0001, 0.9999], norm_instead=False):
        self.s = torch.tensor([s])
        self.clamp_range = clamp_range
        self.norm_instead = norm_instead
        self.min_var = torch.cos(self.s / (1 + self.s) * torch.pi * 0.5) ** 2

    def schedule(self, t, batch_size):
        if t is None:
            t = (1-torch.rand(batch_size)).add(0.001).clamp(0.001, 1.0)
        s, min_var = self.s.to(t.device), self.min_var.to(t.device)
        var = torch.cos((s + t)/(1+s) * torch.pi * 0.5).clamp(0, 1) ** 2 / min_var
        if self.norm_instead:
            var = var * (self.clamp_range[1]-self.clamp_range[0]) + self.clamp_range[0]
        else:
            var = var.clamp(*self.clamp_range)
        logSNR = (var/(1-var)).log()
        return logSNR

class CosineSchedule2(BaseSchedule):
    def setup(self, logsnr_range=[-15, 15]):
        self.t_min = np.arctan(np.exp(-0.5 * logsnr_range[1]))
        self.t_max = np.arctan(np.exp(-0.5 * logsnr_range[0]))

    def schedule(self, t, batch_size):
        if t is None:
            t = 1-torch.rand(batch_size)
        return -2 * (self.t_min + t*(self.t_max-self.t_min)).tan().log()

class SqrtSchedule(BaseSchedule):
    def setup(self, s=1e-4, clamp_range=[0.0001, 0.9999], norm_instead=False):
        self.s = s
        self.clamp_range = clamp_range
        self.norm_instead = norm_instead

    def schedule(self, t, batch_size):
        if t is None:
            t = 1-torch.rand(batch_size)
        var = 1 - (t + self.s)**0.5
        if self.norm_instead:
            var = var * (self.clamp_range[1]-self.clamp_range[0]) + self.clamp_range[0]
        else:
            var = var.clamp(*self.clamp_range)
        logSNR = (var/(1-var)).log()
        return logSNR

class RectifiedFlowsSchedule(BaseSchedule):
    def setup(self, logsnr_range=[-15, 15]):
        self.logsnr_range = logsnr_range

    def schedule(self, t, batch_size):
        if t is None:
            t = 1-torch.rand(batch_size)
        logSNR = (((1-t)**2)/(t**2)).log()
        logSNR = logSNR.clamp(*self.logsnr_range)
        return logSNR

class EDMSampleSchedule(BaseSchedule):
    def setup(self, sigma_range=[0.002, 80], p=7):
        self.sigma_range = sigma_range
        self.p = p

    def schedule(self, t, batch_size):
        if t is None:
            t = 1-torch.rand(batch_size)
        smin, smax, p = *self.sigma_range, self.p
        sigma = (smax ** (1/p) + (1-t) * (smin ** (1/p) - smax ** (1/p))) ** p
        logSNR = (1/sigma**2).log()
        return logSNR

class EDMTrainSchedule(BaseSchedule):
    def setup(self, mu=-1.2, std=1.2):
        self.mu = mu
        self.std = std

    def schedule(self, t, batch_size):
        if t is not None:
            raise Exception("EDMTrainSchedule doesn't support passing timesteps: t")
        logSNR = -2*(torch.randn(batch_size) * self.std - self.mu)
        return logSNR

class LinearSchedule(BaseSchedule):
    def setup(self, logsnr_range=[-10, 10]):
        self.logsnr_range = logsnr_range

    def schedule(self, t, batch_size):
        if t is None:
            t = 1-torch.rand(batch_size)
        logSNR = t * (self.logsnr_range[0]-self.logsnr_range[1]) + self.logsnr_range[1]
        return logSNR

# Any schedule that cannot be described easily as a continuous function of t
# It needs to define self.x and self.y in the setup() method
class PiecewiseLinearSchedule(BaseSchedule):
    def setup(self):
        self.x = None
        self.y = None

    def piecewise_linear(self, x, xs, ys):
        indices = torch.searchsorted(xs[:-1], x) - 1
        x_min, x_max = xs[indices], xs[indices+1]
        y_min, y_max = ys[indices], ys[indices+1]
        var = y_min + (y_max - y_min) * (x - x_min) / (x_max - x_min)
        return var

    def schedule(self, t, batch_size):
        if t is None:
            t = 1-torch.rand(batch_size)
        var = self.piecewise_linear(t, self.x.to(t.device), self.y.to(t.device))
        logSNR = (var/(1-var)).log()
        return logSNR

class StableDiffusionSchedule(PiecewiseLinearSchedule):
    def setup(self, linear_range=[0.00085, 0.012], total_steps=1000):
        linear_range_sqrt = [r**0.5 for r in linear_range]
        self.x = torch.linspace(0, 1, total_steps+1)

        alphas = 1-(linear_range_sqrt[0]*(1-self.x) + linear_range_sqrt[1]*self.x)**2
        self.y = alphas.cumprod(dim=-1)

class AdaptiveTrainSchedule(BaseSchedule):
    def setup(self, logsnr_range=[-10, 10], buckets=100, min_probs=0.0):
        th = torch.linspace(logsnr_range[0], logsnr_range[1], buckets+1)
        self.bucket_ranges = torch.tensor([(th[i], th[i+1]) for i in range(buckets)])
        self.bucket_probs = torch.ones(buckets)
        self.min_probs = min_probs

    def schedule(self, t, batch_size):
        if t is not None:
            raise Exception("AdaptiveTrainSchedule doesn't support passing timesteps: t")
        norm_probs = ((self.bucket_probs+self.min_probs) / (self.bucket_probs+self.min_probs).sum())
        buckets = torch.multinomial(norm_probs, batch_size, replacement=True)
        ranges = self.bucket_ranges[buckets]
        logSNR = torch.rand(batch_size) * (ranges[:, 1]-ranges[:, 0]) + ranges[:, 0]
        return logSNR

    def update_buckets(self, logSNR, loss, beta=0.99):
        range_mtx = self.bucket_ranges.unsqueeze(0).expand(logSNR.size(0), -1, -1).to(logSNR.device)
        range_mask = (range_mtx[:, :, 0] <= logSNR[:, None]) * (range_mtx[:, :, 1] > logSNR[:, None]).float()
        range_idx = range_mask.argmax(-1).cpu()
        self.bucket_probs[range_idx] = self.bucket_probs[range_idx] * beta + loss.detach().cpu() * (1-beta)

class InterpolatedSchedule(BaseSchedule):
    def setup(self, scheduler1, scheduler2, shifts=[1.0, 1.0]):
        self.scheduler1 = scheduler1
        self.scheduler2 = scheduler2
        self.shifts = shifts

    def schedule(self, t, batch_size):
        if t is None:
            t = 1-torch.rand(batch_size)
        t = t.clamp(1e-7, 1-1e-7) # avoid infinities multiplied by 0 which cause nan
        low_logSNR = self.scheduler1(t, shift=self.shifts[0])
        high_logSNR = self.scheduler2(t, shift=self.shifts[1])
        return low_logSNR * t + high_logSNR * (1-t)

# --- Targets
class EpsilonTarget():
    def __call__(self, x0, epsilon, logSNR, a, b):
        return epsilon

    def x0(self, noised, pred, logSNR, a, b):
        return (noised - pred * b) / a

    def epsilon(self, noised, pred, logSNR, a, b):
        return pred

class X0Target():
    def __call__(self, x0, epsilon, logSNR, a, b):
        return x0

    def x0(self, noised, pred, logSNR, a, b):
        return pred

    def epsilon(self, noised, pred, logSNR, a, b):
        return (noised - pred * a) / b

class VTarget():
    def __call__(self, x0, epsilon, logSNR, a, b):
        return a * epsilon - b * x0

    def x0(self, noised, pred, logSNR, a, b):
        squared_sum = a**2 + b**2
        return a/squared_sum * noised - b/squared_sum * pred

    def epsilon(self, noised, pred, logSNR, a, b):
        squared_sum = a**2 + b**2
        return b/squared_sum * noised + a/squared_sum * pred

class RectifiedFlowsTarget():
    def __call__(self, x0, epsilon, logSNR, a, b):
        return epsilon - x0

    def x0(self, noised, pred, logSNR, a, b):
        return noised - pred * b

    def epsilon(self, noised, pred, logSNR, a, b):
        return noised + pred * a

# --- Main Class
class GDF():
    def __init__(self, schedule, input_scaler, target, noise_cond, loss_weight, offset_noise=0):
        self.schedule = schedule
        self.input_scaler = input_scaler
        self.target = target
        self.noise_cond = noise_cond
        self.loss_weight = loss_weight
        self.offset_noise = offset_noise

    def setup_limits(self, stretch_max=True, stretch_min=True, shift=1):
        stretched_limits = self.input_scaler.setup_limits(self.schedule, self.input_scaler, stretch_max, stretch_min, shift)
        return stretched_limits

    def diffuse(self, x0, epsilon=None, t=None, shift=1, loss_shift=1, offset=None):
        if epsilon is None:
            epsilon = torch.randn_like(x0)
        if self.offset_noise > 0:
            if offset is None:
                offset = torch.randn([x0.size(0), x0.size(1)] + [1]*(len(x0.shape)-2)).to(x0.device)
            epsilon = epsilon + offset * self.offset_noise
        logSNR = self.schedule(x0.size(0) if t is None else t, shift=shift).to(x0.device)
        a, b = self.input_scaler(logSNR) # B
        if len(a.shape) == 1:
            a, b = a.view(-1, *[1]*(len(x0.shape)-1)), b.view(-1, *[1]*(len(x0.shape)-1)) # BxCxHxW
        target = self.target(x0, epsilon, logSNR, a, b)

        # noised, noise, logSNR, t_cond
        return x0 * a + epsilon * b, epsilon, target, logSNR, self.noise_cond(logSNR), self.loss_weight(logSNR, shift=loss_shift)

    def undiffuse(self, x, logSNR, pred):
        a, b = self.input_scaler(logSNR)
        if len(a.shape) == 1:
            a, b = a.view(-1, *[1]*(len(x.shape)-1)), b.view(-1, *[1]*(len(x.shape)-1))
        return self.target.x0(x, pred, logSNR, a, b), self.target.epsilon(x, pred, logSNR, a, b)

    def sample(self, model, model_inputs, shape, unconditional_inputs=None, sampler=None, schedule=None, t_start=1.0, t_end=0.0, timesteps=20, x_init=None, cfg=3.0, cfg_t_stop=None, cfg_t_start=None, cfg_rho=0.7, sampler_params=None, shift=1, device="cpu"):
        sampler_params = {} if sampler_params is None else sampler_params
        if sampler is None:
            sampler = DDPMSampler(self)
        r_range = torch.linspace(t_start, t_end, timesteps+1)
        schedule = self.schedule if schedule is None else schedule
        logSNR_range = schedule(r_range, shift=shift)[:, None].expand(
            -1, shape[0] if x_init is None else x_init.size(0)
        ).to(device)

        x = sampler.init_x(shape).to(device) if x_init is None else x_init.clone()
        if cfg is not None:
            if unconditional_inputs is None:
                unconditional_inputs = {k: torch.zeros_like(v) for k, v in model_inputs.items()}
            model_inputs = {
                k: torch.cat([v, v_u], dim=0) if isinstance(v, torch.Tensor) 
                else [torch.cat([vi, vi_u], dim=0) if isinstance(vi, torch.Tensor) and isinstance(vi_u, torch.Tensor) else None for vi, vi_u in zip(v, v_u)] if isinstance(v, list)
                else {vk: torch.cat([v[vk], v_u.get(vk, torch.zeros_like(v[vk]))], dim=0) for vk in v} if isinstance(v, dict)
                else None for (k, v), (k_u, v_u) in zip(model_inputs.items(), unconditional_inputs.items())
            }
        for i in range(0, timesteps):
            noise_cond = self.noise_cond(logSNR_range[i])
            if cfg is not None and (cfg_t_stop is None or r_range[i].item() >= cfg_t_stop) and (cfg_t_start is None or r_range[i].item() <= cfg_t_start):
                cfg_val = cfg
                if isinstance(cfg_val, (list, tuple)):
                    assert len(cfg_val) == 2, "cfg must be a float or a list/tuple of length 2"
                    cfg_val = cfg_val[0] * r_range[i].item() + cfg_val[1] * (1-r_range[i].item())
                pred, pred_unconditional = model(torch.cat([x, x], dim=0), noise_cond.repeat(2), **model_inputs).chunk(2)
                pred_cfg = torch.lerp(pred_unconditional, pred, cfg_val)
                if cfg_rho > 0:
                    std_pos, std_cfg = pred.std(),  pred_cfg.std()
                    pred = cfg_rho * (pred_cfg * std_pos/(std_cfg+1e-9)) + pred_cfg * (1-cfg_rho)
                else:
                    pred = pred_cfg
            else:
                pred = model(x, noise_cond, **model_inputs)
            x0, epsilon = self.undiffuse(x, logSNR_range[i], pred)
            x = sampler(x, x0, epsilon, logSNR_range[i], logSNR_range[i+1], **sampler_params)
            altered_vars = yield (x0, x, pred)

            # Update some running variables if the user wants
            if altered_vars is not None:
                cfg = altered_vars.get('cfg', cfg)
                cfg_rho = altered_vars.get('cfg_rho', cfg_rho)
                sampler = altered_vars.get('sampler', sampler)
                model_inputs = altered_vars.get('model_inputs', model_inputs)
                x = altered_vars.get('x', x)
                x_init = altered_vars.get('x_init', x_init)