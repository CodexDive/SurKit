#!/usr/bin/env python
# -*- coding:UTF-8 -*-
import yaml


class Pipeline:
    def __init__(self, config_file):
        self.load_default()
        f = open(config_file, "r")
        config = yaml.load(f, yaml.FullLoader)
        for key, value in config.items():
            setattr(self, key, value)
        # print(self)

    def load_default(self):
        self.backend = None
        #### model ####
        self.type = None
        self.activation = 'Tanh'
        self.initializer = 'He normal'
        self.layers = []
        self.transforms = []
        self.excitation = None
        self.n_models = None
        self.prior_mean = 0.
        self.prior_var = 1.
        self.noise_tol = 1.
        #### training ####
        self.loss_function = 'MSE'
        self.optimizer = 'Adam'
        self.lr = 1e-4
        self.max_iteration = 20000
        self.report_interval = 1000
        #### pinn ####
        self.pde = []
        self.icbc = []
        self.constant = {}
        #### save ####
        self.save_path = None
        #### data ####
        self.sampler = []
        self.repeat_sample = None
        self.output_name = []
        self.header = []
        self.train_set = None
        self.val_set = None
        self.test_set = None

