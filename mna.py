import torch
from torch.optim.optimizer import Optimizer
from copy import deepcopy

class MNA(Optimizer):
    """
    Implements Modified Neighbor Annealing algorithm.

    It has been proposed in `Heart murmur detection based on Wavelet
        Transformation and a synergy between Artificial Neural Network and modified
        Neighbor Annealing methods" published in Artificial Intelligence in Medicine.
        https://doi.org/10.1016/j.artmed.2017.05.005
    
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        initTemp     (float, optional): initial temperature  (default: 1)
        schedule     (float, optional): schedule coefficient to decay temperature(default: 0)
        window       (float, optional): this argument control the range in which random numbers will be extracted
        terminateTemp(float, optional): training process will be terminated in this temperature/// will be applied in future versions
        #****************************************************************************************************************************
        subItter (int):number of sub iteration in which algorithm can freely search for better answers.
        prew     (dic):weights from previous iteration
        bestw    (dic):best weights ever found
        bestLoss (float): best loss ever found
        preLoss  (float): loss from previous iteration
        bestTemp (float): best temperature for exploring answers
        
    """

    def __init__(self, params, initTemp=1, schedule=0.99, window=2, terminateTemp=0.00001):
        if not 0.0 < initTemp:
            raise ValueError("Invalid initTemp: {}".format(initTemp))
        if not 0.0 <= schedule:
            raise ValueError("Invalid schedule value: {}".format(schedule))
        if not 0.0 < terminateTemp:
            raise ValueError("Invalid terminateTemp value: {}".format(terminateTemp))
        if not 0.0 < window:
            raise ValueError("Invalid window value{}".format(window))
        #*******************************************************************************
        defaults = dict(initTemp=initTemp, schedule=schedule,
                         window=window, terminateTemp=terminateTemp,subItter=10,bestw=[], prew=[],bestLoss=1000, preLoss=1000,bestTemp=0)
        
        super(MNA, self).__init__(params, defaults)
        
        
        for group in self.param_groups:
            group['prew'] =deepcopy( group['params'])
            group['bestw'] =deepcopy( group['params'])
            
            for p in group['params']:
                state = self.state[p]
                #state['step'] = 0

    #*******************************************************************************
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        
            
        for group in self.param_groups:
            if loss < group['bestLoss']:
                if group['bestLoss']-loss > group['bestLoss']*0.005: # as smaller result larger itterations
                    group['bestTemp'] = group['initTemp']
                    group['initTemp'] = group['initTemp']/group['schedule']
                group['bestLoss'] = loss
                group['bestw'] =deepcopy(group['params'])
            #**************************************************************
            if group['subItter'] == 0:
                group['params']=deepcopy(group['bestw'])
                group['subItter'] =10
            else:
                group['subItter']=group['subItter'] - 1
                if loss > group['preLoss']:
                    delt=torch.Tensor([(loss- group['preLoss'])/group['preLoss']])
                    if torch.rand(1) <= torch.exp(-delt/group['initTemp']):
                        group['prew']=deepcopy(group['params'])
                        group['preLoss'] = loss
                    else:
                        group['params']=deepcopy(group['prew'])
                else:
                    group['prew']=deepcopy(group['params'])
                    group['preLoss']=loss
            group['initTemp']= group['initTemp'] * group['schedule']
            # access param key in group dictionary
            
            for p in group['params']:# walkthrough layer by layer
                #if p.grad is None:
                #    continue
                
            #**************************************************************
                if group['initTemp'] > group['terminateTemp']:
                    #generate random var for updating weights
                    p.data=p.data +(2 * group['window'] * group['initTemp'] * torch.rand(p.data.size())) - group['window'] * group['initTemp']
                    ### TODO: R ke behtarin weight iaft shode ast baiad jaigozin shavad
                    
                else:#terminate
                    pass
                    
        return loss