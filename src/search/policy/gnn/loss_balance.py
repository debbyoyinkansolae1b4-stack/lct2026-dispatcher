"""Обрезка градиента по группам параметров, диагностика общих слоёв и баланс критика."""
# refactor: Claude, 21.09.2026 — из src/alt/loss_balance.py; стиль, логика без изменений
import torch

# Общее представление: кодировщики, обмен сообщениями, нормировки, эксперты.
SHARED = ('engineer_in.', 'route_in.', 'request_in.', 'messages.', 'norms.', 'experts.')


def split_parameters(net):
    """(параметры сети, регуляторы потерь). Регуляторы — log_vars и log_alpha.

    log_beta (температура гейта) остаётся в группе сети — так было при всех замерах;
    перенос в регуляторы изменил бы обрезку градиента, то есть траекторию обучения.
    """
    controllers = [net.log_vars, net.log_alpha]
    ids = {id(p) for p in controllers}
    network = [p for p in net.parameters() if id(p) not in ids]
    return network, controllers


def _norm(params):
    return float(torch.sqrt(sum((p.grad.detach().square().sum() for p in params if p.grad is not None),
                                torch.tensor(0.))))


def clip_step_gradients(net, separate=True, limit=1.):
    """Обрезка нормы градиента: сеть и регуляторы отдельно (separate) или вместе."""
    network, controllers = split_parameters(net)
    pre = dict(network=_norm(network), controllers=_norm(controllers))
    if separate:
        torch.nn.utils.clip_grad_norm_(network, limit, error_if_nonfinite=True)
        torch.nn.utils.clip_grad_norm_(controllers, limit, error_if_nonfinite=True)
    else:
        torch.nn.utils.clip_grad_norm_(net.parameters(), limit, error_if_nonfinite=True)
    return dict(before=pre, after=dict(network=_norm(network), controllers=_norm(controllers)),
                separate=separate, network_clipped=pre['network'] > limit,
                controllers_clipped=pre['controllers'] > limit)


def shared_gradient_diagnostics(net, components):
    """Нормы и попарные косинусы градиентов слагаемых потерь по общим слоям.

    Только общее представление: выходные головы своих задач разбавили бы косинусы.
    """
    params = [p for name, p in net.named_parameters() if name.startswith(SHARED)]
    vectors = {}
    for name, loss in components.items():
        grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
        vectors[name] = torch.cat([(g.detach() if g is not None else torch.zeros_like(p)).reshape(-1)
                                   for p, g in zip(params, grads)])
    norms = {k: float(v.norm()) for k, v in vectors.items()}
    cosine = {}
    names = list(vectors)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            denom = norms[a] * norms[b]
            cosine[a + '__' + b] = float(vectors[a] @ vectors[b]) / denom if denom > 1e-12 else None
    return dict(norms=norms, cosine=cosine)


class CriticBalance:
    """Масштаб градиента критика в общих слоях — не в его собственной голове.

    Суррогат с нулевым значением меняет производные, не меняя отчётной потери.
    Опора — актор; скользящее среднее не даёт реагировать на одну почти нулевую партию.
    """
    PREFIXES = SHARED

    def __init__(self, mode='baseline', decay=.9, minimum=.02):
        if mode not in ('baseline', 'fixed', 'adaptive'):
            raise ValueError(mode)
        self.mode = mode
        self.decay = decay
        self.minimum = minimum
        self.actor_ema = None
        self.critic_ema = None
        self.scale = 1.

    @staticmethod
    def scaled(values, scale):
        """Значение вперёд без изменений, градиент назад — с множителем scale.

        Точная замена поправке через отдельный обратный проход: значение равно values,
        производная по values равна scale.
        """
        return scale * values + (1 - scale) * values.detach()

    def correction(self, net, actor_grads, critic_grads):
        rows = [(p, a, c) for (name, p), a, c in zip(net.named_parameters(), actor_grads, critic_grads)
                if name.startswith(self.PREFIXES)]

        def norm(index):
            return float(torch.sqrt(sum((row[index].detach().square().sum() for row in rows
                                         if row[index] is not None), torch.tensor(0.))))

        actor_norm = norm(1)
        critic_norm = .5 * norm(2)
        if self.mode == 'fixed':
            self.scale = .1
        elif self.mode == 'adaptive' and actor_norm > 1e-9 and critic_norm > 1e-9:
            self.actor_ema = (actor_norm if self.actor_ema is None
                              else self.decay * self.actor_ema + (1 - self.decay) * actor_norm)
            self.critic_ema = (critic_norm if self.critic_ema is None
                               else self.decay * self.critic_ema + (1 - self.decay) * critic_norm)
            self.scale = min(1., max(self.minimum, self.actor_ema / self.critic_ema))
        correction = sum(((p - p.detach()) * (.5 * c.detach()) * (self.scale - 1)).sum()
                         for p, a, c in rows if c is not None)
        return correction, dict(mode=self.mode, scale=self.scale, actor_norm=actor_norm,
                                critic_norm=critic_norm, effective_critic_norm=self.scale * critic_norm,
                                actor_ema=self.actor_ema, critic_ema=self.critic_ema)


def diagnostics_due(epoch, epochs, every=10, warmup=3):
    """Эпохи с нуля: диагностика на первых warmup, на каждой every-й и на последней."""
    if every < 1 or warmup < 0:
        raise ValueError('Positive interval and nonnegative warmup required')
    return epoch < warmup or (epoch + 1) % every == 0 or epoch + 1 == epochs
