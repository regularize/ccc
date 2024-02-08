import abc
import torch

from .dataloader_base import DataLoaderBase

class DataLoaderTransformerDecoder(DataLoaderBase):
    """
    An abstract data loader class, 
    for decoder-only transformer models, 
    with a QKV weight matrix, MLP, two layer norms.   
    """
    def __init__(
        self,
        model_name,
        model_args,
        steps_list
        ):

        model_args['att_inner_dim'] = model_args['n_embd'] // model_args['n_head']

        super(
            DataLoaderTransformerDecoder, self).__init__(
            model_name,
            model_args,
            steps_list
            )
            
        self.set_path()
        self.set_modes_list()

    def set_modes_list(self):
        self.modes_list = ['compact', 'full']

    @abc.abstractmethod
    def get_device(self, model):
        pass


    @abc.abstractmethod
    def get_attn_QKV_weight(self, model, i_layer):
        """Provide the QKV-weights data matrix
        This is assumed to be of shape (3*C, C) where C is the embedding dimension,
        and each head in each of Q, K, and V, matrices are a submatrix, 
        with a subset of ROWS and all of columns of this QKV-weights matrix.
        """
        pass

    @abc.abstractmethod
    def get_attn_O_weight(self, model, i_layer):
        """Provide the O-weights data matrix
        This is assumed to be of shape (C, C) where C is the embedding dimension,
        and each head is a submatrix, 
        with a subset of ROWS and all of columns of this O-weights matrix. 

        Note the "ROWS" specification; hence, commonly one has to "transpose" the raw weight matrix.
        """
        pass

    def get_Q_head(self, model, i_layer, i_head):
        r = self.model_args['att_inner_dim']
        return self.get_attn_QKV_weight(model, i_layer)[i_head*r:(i_head+1)*r,:].T

    def get_K_head(self, model, i_layer, i_head):
        n = self.model_args['n_embd']
        r = self.model_args['att_inner_dim']
        return self.get_attn_QKV_weight(model, i_layer)[(n + i_head*r):(n + (i_head+1)*r),:].T

    def get_V_head(self, model, i_layer, i_head):
        n = self.model_args['n_embd']
        r = self.model_args['att_inner_dim']
        return self.get_attn_QKV_weight(model, i_layer)[(2*n + i_head*r):(2*n + (i_head+1)*r),:].T

    def get_O_head(self, model, i_layer, i_head):
        r = self.model_args['att_inner_dim']
        return self.get_attn_O_weight(model, i_layer)[i_head*r:(i_head+1)*r,:].T


    @abc.abstractmethod
    def get_emb_in(self, model):
        """Provide the emb_in matrix, with 0-th dimension of size n_embd, 
        or None to ignore this component.
        """
        pass

    @abc.abstractmethod
    def get_emb_out(self, model):
        """Provide the emb_out matrix, with 0-th dimension of size n_embd, 
        or None to ignore this component.
        """
        pass

    @abc.abstractmethod
    def get_mlp_in(self, model, i_layer):
        """Provide the mlp_in matrix, with 0-th dimension of size n_embd, 
        or None if the model does not have a MLP module in this layer.
        """
        pass

    @abc.abstractmethod
    def get_mlp_out(self, model, i_layer):
        """Provide the mlp_out matrix, with 0-th dimension of size n_embd, 
        or None if the model does not have a MLP module in this layer.
        """
        pass

    @abc.abstractmethod
    def get_ln_1(self, model, i_layer):
        """Provide the weight vector of the 1st layer norm, 
        or None if the model does not have such a module in this layer.
        """
        pass

    @abc.abstractmethod
    def get_ln_2(self, model, i_layer):
        """Provide the weight vector of the 2nd layer norm, 
        or None if the model does not have such a module in this layer.
        """
        pass

    def get_comps_rownorms(self, model, mode):
        """

        Notes: 
        ------
        Change fun1d() and fun2d() to change the aggregation method within each component. 

        The labels in this method (the first element of the 3-tuples in the list "comps"), 
            will later be used in viz_comps_grps() in experiments.py

        BUG: for some reason the layer norms are being ignored; the conditional 
        """
        n_embd = self.model_args['n_embd']
        n_layer = self.model_args['n_layer']
        n_head = self.model_args['n_head']
        r = self.model_args['att_inner_dim']

        device = self.get_device(model)

        #####

        def fun1d(p):
            return (p - torch.median(p)).pow(2)

        def fun2d(p):
            return p.pow(2).mean(dim=1)

        #####

        comps = []

        if self.get_emb_in(model) is not None:
            comps += [(
            '0|0|emb', 
            fun2d(self.get_emb_in(model)), 
            self.get_emb_in(model).shape[1])]
        if self.get_emb_out(model) is not None:
            comps += [(
            '0|1|emb', 
            fun2d(self.get_emb_out(model)), 
            self.get_emb_out(model).shape[1])]
        if self.get_emb_in(model) is not None or self.get_emb_out(model) is not None:
            rowsnum = 1
        else:
            rowsnum = 0

        #####

        if mode == 'compact':

            v = torch.zeros(n_embd, device=device)
            s = 0
            for i_layer in range(n_layer):
                for i_head in range(n_head):
                    v += fun2d(self.get_Q_head(model, i_layer, i_head))
                    s += r
            comps += [('{0}|{1}|Q'.format(rowsnum, 0), v, s)]
            rowsnum += 1

            v = torch.zeros(n_embd, device=device)
            s = 0
            for i_layer in range(n_layer):
                for i_head in range(n_head):
                    v += fun2d(self.get_V_head(model, i_layer, i_head))
                    s += r
            comps += [('{0}|{1}|V'.format(rowsnum, 0), v, s)]
            rowsnum += 1

            v = torch.zeros(n_embd, device=device)
            s = 0
            for i_layer in range(n_layer):
                for i_head in range(n_head):
                    v += fun2d(self.get_O_head(model, i_layer, i_head))
                    s += r
            comps += [('{0}|{1}|O'.format(rowsnum, 0), v, s)]
            rowsnum += 1

            if all([self.get_mlp_in(model, i_layer) is not None for i_layer in range(n_layer)]):
                v = torch.zeros(n_embd, device=device)
                s = 0
                for i_layer in range(n_layer):
                    v += fun2d(self.get_mlp_in(model, i_layer))
                    s += self.get_mlp_in(model, i_layer).shape[1]
                comps += [('{0}|{1}|mlp_1'.format(rowsnum, 0), v, s)]
                rowsnum += 1

            if all([self.get_mlp_out(model, i_layer) is not None for i_layer in range(n_layer)]):
                v = torch.zeros(n_embd, device=device)
                s = 0
                for i_layer in range(n_layer):
                    v += fun2d(self.get_mlp_out(model, i_layer))
                    s += self.get_mlp_out(model, i_layer).shape[1]
                comps += [('{0}|{1}|mlp_2'.format(rowsnum, 0), v, s)]
                rowsnum += 1

            #####

            v = torch.zeros(n_embd, device=device)
            s = 0
            for i_layer in range(n_layer):
                for i_head in range(n_head):
                    v += fun2d(self.get_K_head(model, i_layer, i_head))
                    s += r
            comps += [('{0}|{1}|K'.format(rowsnum, 0), v, s)]
            rowsnum += 1

            if (
                all([self.get_ln_1(model, i_layer) is not None for i_layer in range(n_layer)]) and
                all([self.get_ln_2(model, i_layer) is not None for i_layer in range(n_layer)]) 
            ):
                v = torch.zeros(n_embd, device=device)
                s = 0
                for i_layer in range(n_layer):
                    v += fun1d(self.get_ln_1(model, i_layer))
                    s += 1
                comps += [('{0}|{1}|ln_1'.format(rowsnum, 0), v, s)]

                v = torch.zeros(n_embd, device=device)
                s = 0
                for i_layer in range(n_layer):
                    v += fun1d(self.get_ln_1(model, i_layer))
                    s += 1
                comps += [('{0}|{1}|ln_2'.format(rowsnum, 1), v, s)]
                rowsnum += 1

        elif mode == 'full':

            for i_layer in range(n_layer):
                for i_head in range(n_head):
                    comps += [('{0}|{1}|Q_head{2}'.format(i_head + rowsnum, i_layer, i_head),
                               fun2d(self.get_Q_head(model, i_layer, i_head)),
                               r)]
            rowsnum += n_head

            for i_layer in range(n_layer):
                for i_head in range(n_head):
                    comps += [('{0}|{1}|V_head{2}'.format(i_head + rowsnum, i_layer, i_head),
                            fun2d(self.get_V_head(model, i_layer, i_head)),
                            r)]
            rowsnum += n_head

            for i_layer in range(n_layer):
                for i_head in range(n_head):
                    comps += [('{0}|{1}|O_head{2}'.format(i_head + rowsnum, i_layer, i_head),
                            fun2d(self.get_O_head(model, i_layer, i_head)),
                            r)]
            rowsnum += n_head

            cnt = 0
            for i_layer in range(n_layer):
                if self.get_mlp_in(model, i_layer) is not None:
                    comps += [('{0}|{1}|mlp_1'.format(rowsnum, i_layer),## ????????????????
                               fun2d(self.get_mlp_in(model, i_layer)),
                               self.get_mlp_in(model, i_layer).shape[1])]
                    cnt += 1
            if cnt > 0:
                rowsnum += 1 

            cnt = 0
            for i_layer in range(n_layer):
                if self.get_mlp_out(model, i_layer) is not None:
                    comps += [('{0}|{1}|mlp_2'.format(rowsnum, i_layer),## ????????????????
                               fun2d(self.get_mlp_out(model, i_layer)),
                               self.get_mlp_out(model, i_layer).shape[1])]
                    cnt += 1
            if cnt > 0:
                rowsnum += 1

            #####

            for i_layer in range(n_layer):
                for i_head in range(n_head):
                    comps += [('{0}|{1}|K_head{2}'.format(i_head + rowsnum, i_layer, i_head),
                               fun2d(self.get_K_head(model, i_layer, i_head)),
                               r)]
            rowsnum += n_head

            cnt = 0
            for i_layer in range(n_layer):
                if self.get_ln_1(model, i_layer) is not None:
                    comps += [('{0}|{1}|ln_1'.format(rowsnum, i_layer),
                               fun1d(self.get_ln_1(model, i_layer)),
                               1)]
                    cnt += 1
            if cnt > 0:
                rowsnum += 1

            cnt = 0
            for i_layer in range(n_layer):
                if self.get_ln_2(model, i_layer) is not None:
                    comps += [('{0}|{1}|ln_2'.format(rowsnum, i_layer),
                               fun1d(self.get_ln_2(model, i_layer)),
                               1)]
                    cnt += 1
            if cnt > 0:
                rowsnum += 1

        print('mode:', mode, ' | len(comps) : ', len(comps))
        return comps
