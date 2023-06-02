import torch
from torch import nn
from torch import distributions
from torch.nn.parameter import Parameter
from EI_calculation import approx_ei
class InvertibleNN(nn.Module):
    def __init__(self, nets, nett, mask, device):
        super(InvertibleNN, self).__init__()
        
        self.device = device
        self.mask = nn.Parameter(mask, requires_grad=False)
        length = mask.size()[0] // 2
        self.t = torch.nn.ModuleList([nett() for _ in range(length)]) #repeating len(masks) times
        self.s = torch.nn.ModuleList([nets() for _ in range(length)])
        self.size = mask.size()[1]
    def g(self, z):
        x = z
        log_det_J = x.new_zeros(x.shape[0], device=self.device)
        for i in range(len(self.t)):
            x_ = x*self.mask[i]
            s = self.s[i](x_)*(1 - self.mask[i])
            t = self.t[i](x_)*(1 - self.mask[i])
            x = x_ + (1 - self.mask[i]) * (x * torch.exp(s) + t)
            log_det_J += s.sum(dim=1)
        return x, log_det_J

    def f(self, x):
        log_det_J, z = x.new_zeros(x.shape[0], device=self.device), x
        for i in reversed(range(len(self.t))):
            z_ = self.mask[i] * z
            s = self.s[i](z_) * (1-self.mask[i])
            t = self.t[i](z_) * (1-self.mask[i])
            z = (1 - self.mask[i]) * (z - t) * torch.exp(-s) + z_
            log_det_J -= s.sum(dim=1)
        return z, log_det_J
class Renorm_Dynamic(nn.Module):
    def __init__(self, sym_size, latent_size, effect_size, hidden_units,normalized_state,device,is_random=False):
        #latent_size: input size
        #effect_size: scale, effective latent dynamics size
        super(Renorm_Dynamic, self).__init__()
        if sym_size % 2 !=0:
            sym_size = sym_size + 1
        self.device = device
        self.latent_size = latent_size
        self.effect_size = effect_size
        self.sym_size = sym_size
        nets = lambda: nn.Sequential(nn.Linear(sym_size, hidden_units), nn.LeakyReLU(), 
                                     nn.Linear(hidden_units, hidden_units), nn.LeakyReLU(), 
                                     nn.Linear(hidden_units, sym_size), nn.Tanh())
        nett = lambda: nn.Sequential(nn.Linear(sym_size, hidden_units), nn.LeakyReLU(), 
                                     nn.Linear(hidden_units, hidden_units), nn.LeakyReLU(), 
                                     nn.Linear(hidden_units, sym_size))
        self.dynamics = nn.Sequential(nn.Linear(latent_size, hidden_units), nn.LeakyReLU(), 
                                 nn.Linear(hidden_units, hidden_units), nn.LeakyReLU(), 
                                 nn.Linear(hidden_units, latent_size))
        self.inv_dynamics = nn.Sequential(nn.Linear(latent_size, hidden_units), nn.LeakyReLU(), 
                                 nn.Linear(hidden_units, hidden_units), nn.LeakyReLU(), 
                                 nn.Linear(hidden_units, latent_size))
        mask1 = torch.cat((torch.zeros(1, sym_size // 2, device=self.device), torch.ones(1, sym_size // 2, device=self.device)), 1)
        mask2 = 1 - mask1
        masks = torch.cat((mask1, mask2, mask1, mask2, mask1, mask2), 0)
        
        prior = distributions.MultivariateNormal(torch.zeros(latent_size), torch.eye(latent_size))
        self.flow = InvertibleNN(nets, nett, masks, self.device)
        self.normalized_state=normalized_state
        self.is_random = is_random
        if is_random:
            self.sigmas = torch.nn.parameter.Parameter(torch.rand(1, latent_size, device=self.device))
    def forward(self, x):
        #state_dim = x.size()[1]
        if len(x.size())<=1:
            x = x.unsqueeze(0)
        
        s = self.encoding(x)
        s_next = self.dynamics(s) + s
        if self.normalized_state:
            s_next = torch.tanh(s_next)
        if self.is_random:
            s_next = s_next + torch.relu(self.sigmas.repeat(s_next.size()[0],1)) * torch.randn(s_next.size(), device=self.device)
        y = self.decoding(s_next)
        return y, s, s_next
    def back_forward(self, x):
        #state_dim = x.size()[1]
        if len(x.size())<=1:
            x = x.unsqueeze(0)
        
        s = self.encoding(x)
        s_next = self.inv_dynamics(s) - s
        if self.normalized_state:
            s_next = torch.tanh(s_next)
        if self.is_random:
            s_next = s_next + torch.relu(self.sigmas.repeat(s_next.size()[0],1)) * torch.randn(s_next.size(), device=self.device)
        y = self.decoding(s_next)
        return y, s, s_next
    def multi_step_forward(self, x, steps):
        batch_size = x.size()[0]
        x_hist = x
        predict, latent, latent_n = self.forward(x)
        z_hist = latent
        n_hist = torch.zeros(x.size()[0], x.size()[1]-latent.size()[1], device = self.device)
        for t in range(steps):    
            z_next, x_next, noise = self.simulate(latent)
            z_hist = torch.cat((z_hist, z_next), 0)
            x_hist = torch.cat((x_hist, self.eff_predict(x_next)), 0)
            n_hist = torch.cat((n_hist, noise), 0)
            latent = z_next
        return x_hist[batch_size:,:], z_hist[batch_size:,:], n_hist[batch_size:,:]
    def decoding(self, s_next):
        sz = self.sym_size - self.latent_size
        if sz>0:
            noise = distributions.MultivariateNormal(torch.zeros(sz), torch.eye(sz)).sample((s_next.size()[0], 1))
            noise = noise.to(self.device)
            #print(noise.size(), s_next.size(1))
            if s_next.size()[0]>1:
                noise = noise.squeeze(1)
            else:
                noise = noise.squeeze(0)
            #print(noise.size())
            zz = torch.cat((s_next, noise), 1)
        else:
            zz = s_next
        y,_ = self.flow.g(zz)
        return y
    def decoding1(self, s_next):
        sz = self.sym_size - self.latent_size
        if sz>0:
            noise = distributions.MultivariateNormal(torch.zeros(sz), torch.eye(sz)).sample((s_next.size()[0], 1))
            noise = noise.to(self.device)
            #print(noise.size(), s_next.size(1))
            if s_next.size()[0]>1:
                noise = noise.squeeze(1)
            else:
                noise = noise.squeeze(0)
            #print(noise.size())
            zz = torch.cat((s_next, noise), 1)
        else:
            noise = distributions.MultivariateNormal(torch.zeros(sz), torch.eye(sz)).sample((s_next.size()[0], 1))
            noise = noise.to(self.device)
            #print(noise.size(), s_next.size(1))
            if s_next.size()[0]>1:
                noise = noise.squeeze(1)
            else:
                noise = noise.squeeze(0)
            zz = s_next
        y,_ = self.flow.g(zz)
        return y, noise
    def encoding(self, x):
        xx = x
        if len(x.size()) > 1:
            if x.size()[1] < self.sym_size:
                xx = torch.cat((x, torch.zeros([x.size()[0], self.sym_size - x.size()[1]], device=self.device)), 1)
        else:
            if x.size()[0] < self.sym_size:
                xx = torch.cat((x, torch.zeros([self.sym_size - x.size()[0]], device=self.device)), 0)
        s, _ = self.flow.f(xx)
        if self.normalized_state:
            s = torch.tanh(s)
        return s[:, :self.latent_size]
    def encoding1(self, x):
        xx = x
        if len(x.size()) > 1:
            if x.size()[1] < self.sym_size:
                xx = torch.cat((x, torch.zeros([x.size()[0], self.sym_size - x.size()[1]], device=self.device)), 1)
        else:
            if x.size()[0] < self.sym_size:
                xx = torch.cat((x, torch.zeros([self.sym_size - x.size()[0]], device=self.device)), 0)
        s, _ = self.flow.f(xx)
        if self.normalized_state:
            s = torch.tanh(s)
        return s[:, :self.latent_size], s[:,self.latent_size:]
    def eff_predict(self, prediction):
        return prediction[:, :self.effect_size]
    def simulate(self, x):
        x_next = self.dynamics(x) + x
        if self.normalized_state:
            x_next = torch.tanh(x_next)
        if self.is_random:
            x_next = x_next + torch.relu(self.sigmas.repeat(x_next.size()[0],1)) * torch.randn(x_next.size(), device=self.device)
        decode,noise = self.decoding1(x_next)
        return x_next, decode, noise
    def multi_step_prediction(self, s, steps):
        s_hist = s
        z_hist = self.encoding(s)
        z = z_hist[:1, :]
        for t in range(steps):    
            z_next, s_next, _ = self.simulate(z)
            z_hist = torch.cat((z_hist, z_next), 0)
            s_hist = torch.cat((s_hist, self.eff_predict(s_next)), 0)
            z = z_next
        return s_hist, z_hist

class Stacked_Renorm_Dynamic(nn.Module):
    def __init__(self, sym_size, latent_size, effect_size, cut_size, hidden_units,normalized_state,device,is_random=False):
        #latent_size: input size
        #effect_size: scale, effective latent dynamics size
        super(Stacked_Renorm_Dynamic, self).__init__()
        if latent_size < 1 or latent_size > sym_size:
            print('Latent Size is too small(<1) or too large(>input_size):', latent_size)
            raise
            return
        self.device = device
        self.latent_size = latent_size
        self.effect_size = effect_size
        self.sym_size = sym_size
        i = sym_size
        flows = []
        
        while i > latent_size:
            input_size = max(latent_size, i)
            if input_size % 2 !=0:
                input_size = input_size + 1
            flow = self.build_flow(input_size, hidden_units)
            flows.append(flow)
            i = i // cut_size
        self.flows = nn.ModuleList(flows)
        self.dynamics = nn.Sequential(nn.Linear(latent_size, hidden_units), nn.LeakyReLU(), 
                                     nn.Linear(hidden_units, hidden_units), nn.LeakyReLU(), 
                                     nn.Linear(hidden_units, latent_size))
        
        self.normalized_state=normalized_state
        self.is_random = is_random
        
        
        
        
        #if sym_size % 2 !=0:
        #    sym_size = sym_size + 1
        #self.device = device
        #self.latent_size = latent_size
        #self.effect_size = effect_size
        #self.sym_size = sym_size
        #i = sym_size
        #flows = []
        #while i > latent_size:
        #    if i // cut_size <= latent_size:
        #        i = latent_size
        #    nets = lambda: nn.Sequential(nn.Linear(i, hidden_units), nn.LeakyReLU(), 
        #                             nn.Linear(hidden_units, hidden_units), nn.LeakyReLU(), 
        #                             nn.Linear(hidden_units, i), nn.Tanh())
        #    nett = lambda: nn.Sequential(nn.Linear(i, hidden_units), nn.LeakyReLU(), 
        #                                 nn.Linear(hidden_units, hidden_units), nn.LeakyReLU(), 
        #                                 nn.Linear(hidden_units, i))
        #    mask1 = torch.cat((torch.zeros(1, i // 2, device=self.device), 
        #                       torch.ones(1, i // 2, device=self.device)), 1)
        #    mask2 = 1 - mask1
        #    masks = torch.cat((mask1, mask2, mask1, mask2, mask1, mask2), 0)
        #    flow = InvertibleNN(nets, nett, masks, self.device)
        #    flows.append(flow)
        #    i = i // cut_size
        #self.flows = nn.ModuleList(flows)
        
        #self.dynamics = nn.Sequential(nn.Linear(latent_size, hidden_units), nn.LeakyReLU(), 
        #                             nn.Linear(hidden_units, hidden_units), nn.LeakyReLU(), 
        #                             nn.Linear(hidden_units, latent_size))
        
        #self.normalized_state=normalized_state
        #self.is_random = is_random
        
        
    def build_flow(self, input_size, hidden_units):
        nets = lambda: nn.Sequential(nn.Linear(input_size, hidden_units), nn.LeakyReLU(), 
                                     nn.Linear(hidden_units, hidden_units), nn.LeakyReLU(), 
                                     nn.Linear(hidden_units, input_size), nn.Tanh())
        nett = lambda: nn.Sequential(nn.Linear(input_size, hidden_units), nn.LeakyReLU(), 
                                     nn.Linear(hidden_units, hidden_units), nn.LeakyReLU(), 
                                     nn.Linear(hidden_units, input_size))

        mask1 = torch.cat((torch.zeros(1, input_size // 2, device=self.device), 
                           torch.ones(1, input_size // 2, device=self.device)), 1)
        mask2 = 1 - mask1
        masks = torch.cat((mask1, mask2, mask1, mask2, mask1, mask2), 0)
        flow = InvertibleNN(nets, nett, masks, self.device)
        return flow
    def build_dynamics(self, mid_size, hidden_units):
        dynamics = nn.Sequential(nn.Linear(mid_size, hidden_units), nn.LeakyReLU(), 
                                     nn.Linear(hidden_units, hidden_units), nn.LeakyReLU(), 
                                     nn.Linear(hidden_units, mid_size))
        return dynamics
        
    def forward(self, x):
        #state_dim = x.size()[1]
        
        if len(x.size())<=1:
            x = x.unsqueeze(0)
        
        s = self.encoding(x)
        s_next = self.dynamics(s) + s
        if self.normalized_state:
            s_next = torch.tanh(s_next)
        if self.is_random:
            s_next = s_next + torch.relu(self.sigmas.repeat(s_next.size()[0],1)) * torch.randn(s_next.size(), device=self.device)
        y = self.decoding(s_next)
        return y, s, s_next
    def decoding(self, s_next):
        y = s_next
        for i in range(len(self.flows))[::-1]:
            flow = self.flows[i]
            end_size = self.latent_size
            if i < len(self.flows)-1:
                flow_n = self.flows[i+1]
                end_size = flow_n.size
            sz = flow.size - end_size
            if sz>0:
                noise = distributions.MultivariateNormal(torch.zeros(sz), torch.eye(sz)).sample((y.size()[0], 1))
                noise = noise.to(self.device)
                #print(noise.size(), s_next.size(1))
                if y.size()[0]>1:
                    noise = noise.squeeze(1)
                else:
                    noise = noise.squeeze(0)
                y = torch.cat((y, noise), 1)
            y,_ = flow.g(y)
        return y
    def encoding(self, x):
        xx = x
        if len(x.size()) > 1:
            if x.size()[1] < self.sym_size:
                xx = torch.cat((x, torch.zeros([x.size()[0], self.sym_size - x.size()[1]], device=self.device)), 1)
        else:
            if x.size()[0] < self.sym_size:
                xx = torch.cat((x, torch.zeros([self.sym_size - x.size()[0]], device=self.device)), 0)
        y = xx
        for i,flow in enumerate(self.flows):
            y,_ = flow.f(y)
            if self.normalized_state:
                y = torch.tanh(y)
            if i < len(self.flows)-1:
                lsize = self.flows[i+1].size
            else:
                lsize = self.latent_size
            y = y[:, :lsize]
        return y
    def eff_predict(self, prediction):
        return prediction[:, :self.effect_size]
    def simulate(self, x):
        x_next = self.dynamics(x) + x
        decode = self.decoding(x_next)
        return x_next, decode
    def multi_step_prediction(self, s, steps):
        s_hist = s
        z_hist = self.encoding(s)
        z = z_hist[:1, :]
        for t in range(steps):    
            z_next, s_next = self.simulate(z)
            z_hist = torch.cat((z_hist, z_next), 0)
            s_hist = torch.cat((s_hist, self.eff_predict(s_next)), 0)
            z = z_next
        return s_hist, z_hist
class Parellel_Renorm_Dynamic(nn.Module):
    def __init__(self, sym_size, latent_size, effect_size, cut_size, hidden_units,normalized_state,device,is_random=False):
        #latent_size: input size
        #effect_size: scale, effective latent dynamics size
        super(Parellel_Renorm_Dynamic, self).__init__()
        if latent_size < 1 or latent_size > sym_size:
            print('Latent Size is too small(<1) or too large(>input_size):', latent_size)
            raise
            return
        
        self.device = device
        self.latent_size = latent_size
        self.effect_size = effect_size
        self.sym_size = sym_size
        i = sym_size
        flows = []
        dynamics_modules = []
        
        while i > latent_size:    
            input_size = max(latent_size, i)
            if i == sym_size:
                mid_size = sym_size
                dynamics = self.build_dynamics(mid_size, hidden_units)
                dynamics_modules.append(dynamics)
                flow = self.build_flow(input_size, hidden_units)
                flows.append(flow)
            
            flow = self.build_flow(input_size, hidden_units)
            flows.append(flow)
            mid_size = max(latent_size, i // cut_size)
            dynamics = self.build_dynamics(mid_size, hidden_units)
            dynamics_modules.append(dynamics)
            i = i // cut_size
        self.flows = nn.ModuleList(flows)
        self.dynamics_modules = nn.ModuleList(dynamics_modules)
        
        self.normalized_state=normalized_state
        self.is_random = is_random
    def build_flow(self, input_size, hidden_units):
        if input_size % 2 !=0 and input_size > 1:
            input_size = input_size - 1
        nets = lambda: nn.Sequential(nn.Linear(input_size, hidden_units), nn.LeakyReLU(), 
                                     nn.Linear(hidden_units, hidden_units), nn.LeakyReLU(), 
                                     nn.Linear(hidden_units, input_size), nn.Tanh())
        nett = lambda: nn.Sequential(nn.Linear(input_size, hidden_units), nn.LeakyReLU(), 
                                     nn.Linear(hidden_units, hidden_units), nn.LeakyReLU(), 
                                     nn.Linear(hidden_units, input_size))

        mask1 = torch.cat((torch.zeros(1, input_size // 2, device=self.device), 
                           torch.ones(1, input_size // 2, device=self.device)), 1)
        mask2 = 1 - mask1
        masks = torch.cat((mask1, mask2, mask1, mask2, mask1, mask2), 0)
        flow = InvertibleNN(nets, nett, masks, self.device)
        return flow
    def build_dynamics(self, mid_size, hidden_units):
        dynamics = nn.Sequential(nn.Linear(mid_size, hidden_units), nn.LeakyReLU(), 
                                     nn.Linear(hidden_units, hidden_units), nn.LeakyReLU(), 
                                     nn.Linear(hidden_units, mid_size))
        return dynamics
        
    def forward(self, x):
        #state_dim = x.size()[1]
        
        if len(x.size())<=1:
            x = x.unsqueeze(0)
        ss = self.encoding(x)
        
        s_nexts = []
        ys = []
        for i,s in enumerate(ss):
            s_next = self.dynamics_modules[i](s) + s
            if self.normalized_state:
                s_next = torch.tanh(s_next)
            if self.is_random:
                s_next = s_next + torch.relu(self.sigmas.repeat(s_next.size()[0],1)) * torch.randn(s_next.size(),
                                                                                                   device=self.device)
            if i > 0:
                y = self.decoding(s_next, i)
            else:
                y = s_next
            s_nexts.append(s_next)
            ys.append(y)
        return ys, ss, s_nexts
    def decoding(self, s_next, level):
        y = s_next
        for i in range(level+1)[::-1]:
            flow = self.flows[i]
            end_size = self.latent_size
            if i < len(self.flows)-1:
                flow_n = self.flows[i+1]
                end_size = max(y.size()[1], flow_n.size)
            #print(flow.size, end_size, y.size()[1])
            sz = flow.size - end_size
            
            if sz>0:
                noise = distributions.MultivariateNormal(torch.zeros(sz), torch.eye(sz)).sample((y.size()[0], 1))
                noise = noise.to(self.device)
                #print(noise.size(), s_next.size(1))
                if y.size()[0]>1:
                    noise = noise.squeeze(1)
                else:
                    noise = noise.squeeze(0)
                y = torch.cat((y, noise), 1)
            y,_ = flow.g(y)
        return y
    def encoding(self, x):
        xx = x
        if len(x.size()) > 1:
            if x.size()[1] < self.sym_size:
                xx = torch.cat((x, torch.zeros([x.size()[0], self.sym_size - x.size()[1]], device=self.device)), 1)
        else:
            if x.size()[0] < self.sym_size:
                xx = torch.cat((x, torch.zeros([self.sym_size - x.size()[0]], device=self.device)), 0)
        y = xx
        ys = []
        for i,flow in enumerate(self.flows):
            if y.size()[1] > flow.size:
                #y = torch.cat((y, y[:,:1]), 1)
                y = y[:, :flow.size]
            y,_ = flow.f(y)
            if self.normalized_state:
                y = torch.tanh(y)
            pdict = dict(self.dynamics_modules[i].named_parameters())
            lsize = pdict['0.weight'].size()[1]
            y = y[:, :lsize]
            ys.append(y)
        return ys
    def loss(self, predictions, real, loss_f):
        losses = []
        sum_loss = 0
        for i, predict in enumerate(predictions):
            loss = loss_f(real, predict)
            losses.append(loss)
            sum_loss += loss
        return losses, sum_loss / len(predictions)
    def calc_EIs(self, real, latent_ps, device):
        sp = self.encoding(real)
        eis = []
        sigmass = []
        scales = []
        for i,state in enumerate(sp):
            latent_p = latent_ps[i]
            flow = self.flows[i]
            dynamics = self.dynamics_modules[i]
            dd = dict(dynamics.named_parameters())
            scale = dd['0.weight'].size()[1]
            
            sigmas = torch.sqrt(torch.mean((state-latent_p)**2, 0))
            sigmas_matrix = torch.diag(sigmas)
            ei = approx_ei(scale, scale, sigmas_matrix.data, lambda x:(dynamics(x.unsqueeze(0))+x.unsqueeze(0)), 
                           num_samples = 1000, L=100, easy=True, device=device)
            eis.append(ei)
            sigmass.append(sigmas)
            scales.append(scale)
        return eis, sigmass, scales
        
    def eff_predict(self, prediction):
        return prediction[:, :self.effect_size]
    def simulate(self, x, level):
        if level > len(self.dynamics_modules) or level<0:
            print('input error: level must be less than', len(self.dynamics_modules))
        dynamics = self.dynamics_modules[level]
        x_next = dynamics(x) + x
        decode = self.decoding(x_next, level)
        return x_next, decode
    def multi_step_prediction(self, s, steps, level):
        if level > len(self.dynamics_modules) or level<0:
            print('input error: level must be less than', len(self.dynamics_modules))
        s_hist = s
        ss = self.encoding(s)
        z_hist = ss[level]
        z = z_hist[:1, :]
        for t in range(steps):    
            z_next, s_next = self.simulate(z, level)
            z_hist = torch.cat((z_hist, z_next), 0)
            s_hist = torch.cat((s_hist, self.eff_predict(s_next)), 0)
            z = z_next
        return s_hist, z_hist