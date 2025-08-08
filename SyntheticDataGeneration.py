import torch
def get_synthetic_data(n):
    x=torch.linspace(0,1,n)
    x=torch.randint(1,2,(1,))*(x-torch.rand(1))*(x+torch.rand(1))*(x-torch.rand(1))*(x-torch.rand(1))+\
        (torch.rand(1)/10)*torch.randn(n)+(torch.rand(1)/10)*torch.rand(n)+\
            torch.sum(torch.vstack([(0.05+0.1*torch.rand(1))*torch.sin(2*torch.pi*x*torch.randint(5,8,(1,))+torch.rand(1)*torch.pi) for _ in range(3)]),dim=0)
    return 100*x
