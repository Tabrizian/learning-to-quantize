import torch
import torch.nn
import torch.multiprocessing

from data import InfiniteLoader


class GradientEstimator(object):
    def __init__(self, data_loader, opt, tb_logger=None, *args, **kwargs):
        self.opt = opt
        self.model = None
        self.data_loader = data_loader
        self.tb_logger = tb_logger

    def init_data_iter(self):
        self.data_iter = iter(InfiniteLoader(self.data_loader))
        self.estim_iter = iter(InfiniteLoader(self.data_loader))

    def grad(self, model_new, in_place=False):
        raise NotImplementedError('grad not implemented')

    def grad_estim(self, model):
        # insuring continuity of data seen in training
        # TODO: make sure sub-classes never use any other data_iter, e.g. raw
        dt = self.data_iter
        self.data_iter = self.estim_iter
        ret = self.grad(model)
        self.data_iter = dt
        return ret

    def get_gradient_distribution(self, model, gviter):
        """
        gviter: Number of minibatches to apply on the model
        model: Model to be evaluated
        """
        # estimate grad mean and variance
        mean_estimates = [torch.zeros_like(g) for g in model.parameters()]

        for i in range(gviter):
            minibatch_gradient = self.grad_estim(model)
            for e, g in zip(mean_estimates, minibatch_gradient):
                e += g

        # Calculate the mean
        for e in mean_estimates:
            e /= gviter

        # Fix the randomization seed
        torch.manual_seed(123)
        random_layer = torch.randint(0, len(mean_estimates), (1,))
        random_weight_layer_size = mean_estimates[random_layer].shape
        random_weight_array = []
        for weight in random_weight_layer_size:
            random_weight_array.append(torch.randint(0, weight, (1,)))


        # Number of Weights
        variance_estimates = [torch.zeros_like(g) for g in model.parameters()]
      
        for i in range(gviter):
            minibatch_gradient = self.grad_estim(model)
            v = [(gg-ee).pow(2) for ee, gg in zip(mean_estimates, minibatch_gradient)]
            for e, g in zip(variance_estimates, v):
                e += g

        variance_estimates = variance_estimates[random_layer]
        mean_estimates = mean_estimates[random_layer]

        for weight in random_weight_array:
            variance_estimates = variance_estimates[weight]
            variance_estimates.squeeze_()

            mean_estimates = mean_estimates[weight]
            mean_estimates.squeeze_()
        
        variance_estimates = variance_estimates / gviter
      
        return mean_estimates, variance_estimates

    def get_minibatch_gradients(self, model, gviter):
        

    def get_Ege_var(self, model, gviter):
        # estimate grad mean and variance
        Ege = [torch.zeros_like(g) for g in model.parameters()]

        for i in range(gviter):
            ge = self.grad_estim(model)
            for e, g in zip(Ege, ge):
                e += g

        for e in Ege:
            e /= gviter

        torch.manual_seed(123)
        random_layer = torch.randint(0, len(Ege), (1,))
        random_weight_layer_size = Ege[random_layer].shape
        random_weight_array = []
        for weight in random_weight_layer_size:
            random_weight_array.append(torch.randint(0, weight, (1,)))


        # Number of Weights
        nw = sum([w.numel() for w in model.parameters()])
        var_e = [torch.zeros_like(g) for g in model.parameters()]
        Es = [torch.zeros_like(g) for g in model.parameters()]
        En = [torch.zeros_like(g) for g in model.parameters()]
        for i in range(gviter):
            ge = self.grad_estim(model)
            v = [(gg-ee).pow(2) for ee, gg in zip(Ege, ge)]
            for e, g in zip(var_e, v):
                e += g

        # import ipdb; ipdb.set_trace()
        
        # This layer seems to contain some variance, most other layers are zero
        var_e = var_e[random_layer]
        Ege = Ege[random_layer]

        for weight in random_weight_array:
            var_e = var_e[weight]
            var_e.squeeze_()

            Ege = Ege[weight]
            Ege.squeeze_()
        
        
        print('Variance is ' + str(var_e.item()))

        var_e = var_e / gviter
        print(var_e)
        print('Variance is ' + str(var_e.item()))
        # Division by gviter cancels out in ss/nn
        snr_e = sum(
                [((ss+1e-10).log()-(nn+1e-10).log()).sum()
                    for ss, nn in zip(Es, En)])/nw
        nv_e = sum([(nn/(ss+1e-7)).sum() for ss, nn in zip(Es, En)])/nw
        return Ege, var_e, snr_e, 0.00034

    def state_dict(self):
        return {}

    def load_state_dict(self, state):
        pass
