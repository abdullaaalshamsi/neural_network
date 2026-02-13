# what is diffrent of MSE vs MAE

import torch

def mae(y_hat: torch.tensor, y:torch.tensor) -> torch.tensor:
    return torch.mean(torch.abs(y - y_hat))

def mse(y_hat: torch.tensor,y: torch.tensor) -> torch.tensor:
    return torch.mean((y - y_hat) ** 2)

def main():
    y = torch.tensor([10.0,5.0,8.0,12.0], device="cuda:1")
    y_hat = torch.tensor([8.0, 7.0,6.0,15.0], device="cuda:1")
    
    L_mae = mae(y_hat, y)
    L_mse = mse(y_hat, y)
    
    print("y     = ",y)
    print("y_hat = ",y_hat)
    print("MAE   =", L_mae.item())
    print("MSE   =", L_mse.item())    

if __name__ == "__main__":
    main()