import torch

def train_one_weight(x_val=3.0, y_val=10.0, eta=0.01,steps=200,device="cuda:1"):
    x = torch.tensor(x_val,dtype=torch.float32, device=device)
    y = torch.tensor(y_val,dtype=torch.float32, device=device)
    
    w = torch.tensor(0.0, dtype=torch.float32, device=device)
    
    w_hist = []
    l_hist = []
    yhat_hist = []
    g_hist = []
    
    for t in range(steps):
        y_hat = w * x
        L = (y - y_hat) ** 2
        g = 2 * x * (w * x - y)
        
        w_hist.append(w.item())
        l_hist.append(L.item())
        yhat_hist.append(y_hat.item())
        g_hist.append(g.item())
        
        w = w - eta * g
        
    print(f"Final w     = {w.item():.6f}")
    print(f"Final y_hat = {(w*x).item():.6f}")
    print(f"Final Loss  = {((y - w*x)**2).item():.6f}")

    return w_hist, l_hist, yhat_hist, g_hist

if __name__ == "__main__":
    train_one_weight(device="cuda:1")