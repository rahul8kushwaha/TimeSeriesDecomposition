import torch
def get_synthetic_data(n):
    x=torch.linspace(0,1,n)
    x=torch.randint(1,2,(1,))*(x-torch.rand(1))*(x+torch.rand(1))*(x-torch.rand(1))*(x-torch.rand(1))+\
        (torch.rand(1)/10)*torch.randn(n)+(torch.rand(1)/10)*torch.rand(n)+0.1*torch.sin(2*torch.pi*x*6+torch.pi/6)+0.1*torch.sin(2*torch.pi*x*3)
    return 100*x