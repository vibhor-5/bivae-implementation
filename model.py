import torch 

class BiVAE(nn.Module):
    def __init__(
        self,
        k,
        user_encoder_structure,
        item_encoder_structure,
        act_fn,
        likelihood,
        cap_priors,
        feature_dim,
        batch_size,
    ):
        super(BiVAE, self).__init__()

        self.mu_theta = torch.zeros((item_encoder_structure[0], k))  # n_users*k
        self.mu_beta = torch.zeros((user_encoder_structure[0], k))  # n_items*k

        self.theta = torch.randn(item_encoder_structure[0], k) * 0.01
        self.beta = torch.randn(user_encoder_structure[0], k) * 0.01
        torch.nn.init.kaiming_uniform_(self.theta, a=np.sqrt(5))

        self.likelihood = likelihood
        self.act_fn = ACT.get(act_fn, None)
        if self.act_fn is None:
            raise ValueError("Supported act_fn: {}".format(ACT.keys()))

        self.cap_priors = cap_priors
        if self.cap_priors.get("user", False):
            self.user_prior_encoder = nn.Linear(feature_dim.get("user"), k)
        if self.cap_priors.get("item", False):
            self.item_prior_encoder = nn.Linear(feature_dim.get("item"), k)

        # User Encoder
        self.user_encoder = nn.Sequential()
        for i in range(len(user_encoder_structure) - 1):
            self.user_encoder.add_module(
                "fc{}".format(i),
                nn.Linear(user_encoder_structure[i], user_encoder_structure[i + 1]),
            )
            self.user_encoder.add_module("act{}".format(i), self.act_fn)
        self.user_mu = nn.Linear(user_encoder_structure[-1], k)  # mu
        self.user_std = nn.Linear(user_encoder_structure[-1], k)

        # Item Encoder
        self.item_encoder = nn.Sequential()
        for i in range(len(item_encoder_structure) - 1):
            self.item_encoder.add_module(
                "fc{}".format(i),
                nn.Linear(item_encoder_structure[i], item_encoder_structure[i + 1]),
            )
            self.item_encoder.add_module("act{}".format(i), self.act_fn)
        self.item_mu = nn.Linear(item_encoder_structure[-1], k)  # mu
        self.item_std = nn.Linear(item_encoder_structure[-1], k)

    def to(self, device):
        self.beta = self.beta.to(device=device)
        self.theta = self.theta.to(device=device)
        self.mu_beta = self.mu_beta.to(device=device)
        self.mu_theta = self.mu_theta.to(device=device)
        return super(BiVAE, self).to(device)

    def encode_user_prior(self, x):
        h = self.user_prior_encoder(x)
        return h

    def encode_item_prior(self, x):
        h = self.item_prior_encoder(x)
        return h

    def encode_user(self, x):
        h = self.user_encoder(x)
        return self.user_mu(h), torch.sigmoid(self.user_std(h))

    def encode_item(self, x):
        h = self.item_encoder(x)
        return self.item_mu(h), torch.sigmoid(self.item_std(h))

    def decode_user(self, theta, beta):
        h = theta.mm(beta.t())
        return torch.sigmoid(h)

    def decode_item(self, theta, beta):
        h = beta.mm(theta.t())
        return torch.sigmoid(h)

    def reparameterize(self, mu, std):
        eps = torch.randn_like(mu)
        return mu + eps * std

    def forward(self, x, user=True, beta=None, theta=None):

        if user:
            mu, std = self.encode_user(x)
            theta = self.reparameterize(mu, std)
            return theta, self.decode_user(theta, beta), mu, std
        else:
            mu, std = self.encode_item(x)
            beta = self.reparameterize(mu, std)
            return beta, self.decode_item(theta, beta), mu, std

    def loss(self, x, x_, mu, mu_prior, std, kl_beta):
        # Likelihood
        ll_choices = {
            "bern": x * torch.log(x_ + EPS) + (1 - x) * torch.log(1 - x_ + EPS),
            "gaus": -((x - x_) ** 2),
            "pois": x * torch.log(x_ + EPS) - x_,
        }

        ll = ll_choices.get(self.likelihood, None)
        if ll is None:
            raise ValueError("Supported likelihoods: {}".format(ll_choices.keys()))

        ll = torch.sum(ll, dim=1)

        # KL term
        kld = -0.5 * (1 + 2.0 * torch.log(std) - (mu - mu_prior).pow(2) - std.pow(2))
        kld = torch.sum(kld, dim=1)

        return torch.mean(kl_beta * kld - ll)